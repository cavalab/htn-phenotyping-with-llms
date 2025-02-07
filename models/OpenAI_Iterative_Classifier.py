"""Base for openAI based methods"""

import os
import traceback
import importlib
from dotenv import load_dotenv, find_dotenv

import ast

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score

from openai import OpenAI

from .openai_cfg import cfg, default_cp_function

import sys

sys.path.append("..")
from feature_descriptions import htn_variable_dict

load_dotenv(find_dotenv())

# setting up the environment ---------------------------------------------------
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


# Implementing the class -------------------------------------------------------
class OpenAI_Iterative_Classifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        phenotype="htn_dx_ia",
        model="gpt-4o-mini",
        # recs from blogpost:
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        filename=None,
        max_iterations=5,
        max_stall_count=2,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.phenotype = phenotype
        self.filename = filename
        self.max_iterations = max_iterations
        self.max_stall_count = max_stall_count

    def set_prompt(self, target="", richness=False):
        # after finishing implementing, try to use different models
        # mean --> of all visits. explain that in the description. try with all variables (appendix of the paper have their descriptions).
        if target in ["htn_dx_ia", "HTN_heuristic"]:
            if richness:
                # self.phenotype = 'hypertension, otherwise known as high blood pressure, which can be intuited based on high blood pressure measurements, number of encounters with high blood pressure measurements, hypertension Dx codes, and/or counts in clinical notes'
                self.phenotype = "hypertension, which we will define as 2 or more hypertension Dx codes"
            else:
                self.phenotype = "hypertension"
        elif target in ["res_htn_dx_ia", "res_HTN_heuristic"]:
            if richness:
                # self.phenotype = 'treatment resistant hypertension, defined as a high blood pressure measurements while prescribed 3 or more hypertension medications or requiring prescription of 4 or more hypertentsion medications'
                self.phenotype = "treatment resistant hypertension, which we will define as 2 or more high blood pressure measurements while prescribed 3 or more hypertension medications"
            else:
                self.phenotype = "treatment resistant hypertension"
        elif target in ["htn_hypok_dx_ia", "hypoK_heuristic_v4"]:
            if richness:
                # self.phenotype = 'hypertension with hypokalemia, defined as the cooccurrence of high blood pressure, which can be intuited based on high blood pressure measurements, number of encounters with high blood pressure measurements, hypertension Dx codes, and/or counts in clinical notes, and low potassium, which can be defined as either 2 or more low potassium test results, 2 or more potassium supplementation prescriptions, or 2 or more hypokalemia diagnosis code'
                self.phenotype = "hypertension with hypokalemia, which we will define as 2 or more hypertension Dx codes and either 2 or more low potassium test results, 2 or more potassium supplementation prescriptions, or 2 or more hypokalemia diagnosis codes"
            else:
                self.phenotype = "hypertension with hypokalemia"
        else:
            raise Exception("Unknown target")

        print("set phenotype prompt description to: ", self.phenotype)

        pass

    def set_file(self, filename):
        # Because fold doesnt matter, we have the option to save or load previous programs (avoid excessive openAPI calls).
        # So we are going to save the result and future calls will load it for a given seed (different seeds will load different programs,
        # so make sure to have temperature!=0.0 if running several repetitions)
        self.filename = filename

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "X must be a dataframe for openAI to work."

        self.threshold_ = 0.5

        X_variable_dict = {c: htn_variable_dict[c] for c in X.columns}

        # Asking for pred_proba
        func_return = "floats representing the probability"
        # func_return = 'booleans representing the classification'

        self.messages_ = cfg["init_messages"](
            self.phenotype, func_return, X_variable_dict
        )

        self.str_rep = None
        best_auprc = -1
        stall_count = 0
        best_cp = None
        auprcs = []
        improvement = None
        if (self.filename is not None) and (not os.path.exists(self.filename)):
            for i in range(self.max_iterations):
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                # generates text competion
                completion = client.chat.completions.create(
                    messages=self.messages_,
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                cp = completion.choices[0].message.content
                if cp.startswith("```python"):
                    cp = cp.split("```python")[1][:-3]

                print("iteration", i, "best_auprc:", best_auprc)

                print(cp)

                self.messages_.append(
                    {
                        "role": "assistant",
                        "content": cp,
                    }
                )

                # evaluate and refine
                instruct_message, auroc, auprc = self.evaluate_and_instruct(
                    cp, X, y, improvement
                )
                print(instruct_message)
                auprcs.append(auprc)
                # keep best
                if best_auprc < 0:
                    best_auprc = auprc
                    best_cp = cp
                if auprc > best_auprc:
                    best_auprc = auprc
                    improvement = True
                    best_cp = cp
                else:
                    stall_count += 1
                    improvement = False
                if stall_count > self.max_stall_count:
                    print("max stall count reached, breaking")
                    break

                self.messages_.append(
                    {
                        "role": "user",
                        "content": instruct_message,
                    }
                )

            # write final cp
            print("Best Computable Phenotype Found:\n", best_cp)
            self.str_rep = best_cp
            with open(self.filename, "w") as file:
                file.write(self.str_rep)

            print(f"Content written to {self.filename}")
        else:
            try:
                with open(self.filename, "r") as file:
                    self.str_rep = file.read()
                print(f"Content loaded from {self.filename}")
            except FileNotFoundError:
                print(f"File {self.filename} not found")

        # print(self.str_rep)

        # self.set_cp_function(self.str_rep)
        try:
            self.cp_function_ = self.get_cp_function_from_file()
            self.train_success_ = True
        except Exception:
            self.cp_function_ = default_cp_function
            self.train_success_ = False

        # self._find_threshold(X, y)

        print("auprcs:", auprcs)

        return self

    def evaluate_and_instruct(self, cp, X, y_true, improvement):
        """Evaluate a computable phenotype and return an instruction message"""
        target = self.phenotype.split(",")[0]
        # try to predict; if it fails, capture error message and append
        try:
            cp_function = self.get_cp_function(cp)
            y_pred = self.predict_proba(X, cp_function=cp_function)[:, 1]
        except Exception:
            # make custom message with error traceback
            message = "\n".join(
                [
                    "Python encountered an error when trying to execute the function.",
                    f"Error Message:\n{traceback.format_exc()}",
                    "\nPlease try again.",
                    "**MAKE ABSOLUTELY SURE TO RETURN A SYNTACTICALLY VALID PYTHON FUNCTION**.",
                ]
            )
            return message, 0.5, 0

        # subset features to those in cp
        # subset only in case of large feature set?
        subset_features = [c for c in X.columns if c in cp]
        Xs = X[subset_features]
        # import ipdb
        # ipdb.set_trace()
        y_true = y_true.astype(bool)
        # get overall performance
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred > self.threshold_).ravel()
        # get worst cases
        # TODO: give random or close call cases instead of worst performance

        use_fp_example = np.max(y_pred[~y_true]) > self.threshold_
        if use_fp_example:
            # idx_worst_FP = np.argmax(y_pred[~y_true]) # np.random.choice
            idxs_worst_FP = np.random.randint(0, len(Xs.loc[~y_true]), size=3)
            X_max_fp = Xs.loc[~y_true].iloc[idxs_worst_FP]
            y_max_fp = y_pred[~y_true][idxs_worst_FP]

        use_fn_example = np.max(1 - y_pred[y_true]) > self.threshold_
        if use_fn_example:
            idx_worst_FN = np.argmax(1 - y_pred[y_true])
            X_min_fn = Xs.loc[y_true].iloc[idx_worst_FN]
            y_min_fn = y_pred[y_true][idx_worst_FN]

        if improvement:
            improve_msg = "Good News! The updated Python function you created outerperformed the previous version. Let's keep those improvements coming."
        elif improvement is None:
            improve_msg = ""
        else:
            improve_msg = "The updated Python function you created did not outperform the previous version you provided. Let's try to do better this time."

        n_pos = y_true.sum()
        n_neg = len(y_true) - y_true.sum()

        message = "\n".join(
            [
                f"We evaluated the prediction function you provided on a set of {len(Xs)} patients.",
                improve_msg,
                "Using the performance feedback below, please refine the Python function. ",
                "\n# Overall Performance\n",
                f"Area Under the Receiver-Operating Curve (AUROC): {auroc:.3f}",
                f"Area under the precision-recall curve (AUPRC): {auprc:.3f}",
                # f"True Negatives: {tn} ({tn / n_neg * 100:.1f}%)",
                f"The False Positive Rate is {fp / n_neg * 100:.1f}%",
                f"The False Negative Rate is {fn / n_pos * 100:.1f}%",
                # f"False Negatives: {fn} ({fn / n_pos * 100:.1f}%)",
                # f"True Positives: {tp} ({tp / n_pos * 100:.1f}%)",
                "",
                "",
            ]
        )
        if use_fp_example:
            message += (
                f"# False Positives\n"
                f"Please refine the function so that some of the {fp} False Positives have lower predicted probabilities.\n"
            )
            X_max_fp["Correct Label"] = 0
            X_max_fp["Predicted Probability"] = y_max_fp
            message += "\n".join(
                [
                    f"Here is a table of {len(X_max_fp)} examples of false positives.",
                    f"{X_max_fp[['Correct Label', 'Predicted Probability'] + subset_features].reset_index().T.to_markdown()}",
                    "\n",
                ]
            )
        if use_fn_example:
            message += f"Please refine the function to so that some of the {fn} False Negatives have higher predicted probabilities.\n"
            X_min_fn["Correct Label"] = 0
            X_min_fn["Predicted Probability"] = y_min_fn
            message += "\n".join(
                [
                    f"Here is a table of {len(X_min_fn)} examples of false negatives.",
                    f"{X_min_fn[['Correct Label', 'Predicted Probability'] + subset_features].reset_index().T.to_markdown()}",
                    "\n",
                ]
            )
        message += "\n".join(
            [
                "Please create an updated Python function named `predict_hypertension` that achieves fewer false positives and fewer false negatives than the one you previously provided."
                "Where possible, rather than adding additional logic and calculations, try to reuse and refine the existing logic in the previous function."
                f"The function should assess whether each patient (represented as rows) has evidence of {self.phenotype}. "
                f"As before, the function takes a pandas DataFrame named `df` as input. "
                f"Recall that the available columns and their meanings are provided as key value pairs in a dictionary previously provided. "
                "As before, you may only use the features whose names appear in this dictionary. "
                "As before, your response must contain only a Python function, with no comments or explanations, that strictly follows the given description. ",
            ]
        )

        return message, auroc, auprc

    def get_cp_function_from_file(self, filename=None):
        # alternative TODO: use import lib to import from the filename
        if filename is None:
            filename = self.filename
        cp_module = importlib.__import__(
            filename[:-3].replace("/", "."),
            globals(),
            locals(),
            ["predict_hyptertension"],
        )
        return cp_module.predict_hypertension

    def get_cp_function(self, str_rep):
        filename = self.filename.replace(".py", "[tmp].py")
        with open(filename, "w") as file:
            file.write(str_rep)
        return self.get_cp_function_from_file(filename)

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] > self.threshold_, 1, 0)

    def predict_proba(self, X, cp_function=None):
        check_is_fitted(self)

        assert isinstance(X, pd.DataFrame), "X must be a dataframe for openAI to work."

        try:
            if cp_function:
                prob = cp_function(X)
            else:
                prob = self.cp_function_(X)
        except Exception:
            prob = np.zeros(size=len(X))

        prob = np.hstack(
            (np.ones(X.shape[0]).reshape(-1, 1), np.array(prob).reshape(-1, 1))
        )
        prob[:, 0] -= prob[:, 1]

        return prob

    def get_model(self):
        return self.str_rep

    def _find_threshold(self, X, y):
        print("Optimizing threshold... ", end="")
        probas = self.predict_proba(X)
        probas_sorted = np.unique(np.sort(probas))

        # maximizing metric
        best_threshold = 0.0
        best_metric = 0.0
        for i in range(len(probas_sorted)):
            mid = probas_sorted[i]
            preds = np.where(probas[:, 1] > mid, 1, 0)

            # calculate gain
            mid_metric = balanced_accuracy_score(y, preds)

            # update best
            if mid_metric >= best_metric:
                best_threshold = mid
                best_metric = mid_metric

        self.threshold_ = best_threshold
        print(f" Done! threshold is {self.threshold_}, with metric of {best_metric}")

    def _count_operations(self):
        # Extract the function definition
        if self.train_success_:
            tree = ast.parse(self.str_rep)

            return sum([1 for x in ast.walk(tree)])
        else:
            return np.nan
