"""Base for openAI based methods"""

import os
import traceback
import importlib
from dotenv import load_dotenv, find_dotenv
import uuid
import ast

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score

import openai

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
        max_iterations=10,
        max_stall_count=3,
        random_state=None,
        max_tokens=cfg["max_tokens"],
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.phenotype = phenotype
        self.filename = filename
        self.max_iterations = max_iterations
        self.max_stall_count = max_stall_count
        self.random_state = random_state
        self.max_tokens = max_tokens

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
        func_return = cfg["func_return"]
        # func_return = 'booleans representing the classification'

        self.messages_ = cfg["init_messages"](
            self.phenotype, func_return, X_variable_dict
        )

        self.str_rep = None
        best_auprc = -1
        stall_count = 0
        best_cp = None
        auprcs = []
        cps = []
        improvement = None
        if (self.filename is not None) and (not os.path.exists(self.filename)):
            for i in range(self.max_iterations):
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                # generates text competion
                api_request = dict(
                    messages=self.messages_,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.random_state,
                )
                try:
                    completion = client.chat.completions.create(**api_request)
                except openai.BadRequestError:
                    print(traceback.format_exc())
                    break
                except openai.RateLimitError:
                    print(traceback.format_exc())
                    import time

                    time.sleep(np.random.randint(low=1, high=5))
                    completion = client.chat.completions.create(**api_request)
                    pass

                response = completion.choices[0].message.content
                cp = response
                if cp.startswith("```python"):
                    cp = cp.split("```python")[1][:-3]

                print("iteration", i, "best_auprc:", best_auprc)

                print(cp)
                cps.append(cp)

                self.messages_.append(
                    {
                        "role": "assistant",
                        "content": response,
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
                elif auprc > best_auprc:
                    print("AURPC improved from", best_auprc, "to", auprc)
                    best_auprc = auprc
                    best_cp = cp
                    improvement = True
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
        print("cps:")
        for cpi in cps:
            print(40 * "#")
            print(cpi)

        return self

    def evaluate_and_instruct(self, cp, X, y_true, improvement):
        """Evaluate a computable phenotype and return an instruction message"""
        # try to predict; if it fails, capture error message and append
        try:
            cp_function = self.get_cp_function(cp)

            y_pred = self.predict_proba(X, cp_function=cp_function)[:, 1]

            # get overall performance
            y_true = y_true.astype(bool)
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            auroc = round(auroc, 3)
            auprc = round(auprc, 3)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred > self.threshold_).ravel()
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
        n_features = min(10, X.shape[1])
        subset_features = [c for c in X.columns if c in cp]
        if len(subset_features) < n_features:
            n_add = n_features - len(subset_features)
            for i in range(n_add):
                candidates = [c for c in X.columns if c not in subset_features]
                subset_features.append(np.random.choice(candidates))
        elif len(subset_features) > n_features:
            subset_features = list(
                np.random.choice(subset_features, size=n_features, replace=False)
            )
        Xs = X[subset_features]

        # get example cases

        use_fp_example = np.max(y_pred[~y_true]) > self.threshold_
        if use_fp_example:
            fp_mask = (y_pred > self.threshold_) & (~y_true)
            # idx_worst_FP = np.argmax(y_pred[fp_mask])  # np.random.choice
            # idx_rand_FP = np.random.randint(low=0, high=len(y_pred[fp_mask]))
            # idx_best_FP = np.argmin(y_pred[fp_mask])  # np.random.choice

            X_fps = Xs.loc[fp_mask]
            X_fps["True Label"] = 0
            X_fps["Predicted Probability"] = y_pred[fp_mask]
            X_fps = X_fps.reset_index()
            X_fps.index.name = "Patient"
            if len(X_fps) > 10:
                X_fps = X_fps.sample(10, replace=False)
            X_fps = X_fps[["True Label", "Predicted Probability"] + subset_features]

        use_fn_example = np.max(1 - y_pred[y_true]) > self.threshold_
        if use_fn_example:
            fn_mask = (y_pred <= self.threshold_) & (y_true)
            X_fns = Xs.loc[fn_mask]
            X_fns["True Label"] = 1
            X_fns["Predicted Probability"] = y_pred[fn_mask]
            X_fns = X_fns.reset_index()
            X_fns.index.name = "Patient"
            if len(X_fns) > 10:
                X_fns = X_fns.sample(10, replace=False)
            X_fns = X_fns[["True Label", "Predicted Probability"] + subset_features]

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
                f"The False Positive Rate is {fp / n_neg * 100:.1f}%",
                f"The False Negative Rate is {fn / n_pos * 100:.1f}%",
                "",
                "",
            ]
        )
        if use_fp_example:
            message += (
                f"# Analysis of False Positives\n\n"
                f"Please refine the function so that the {fp} False Positives have lower predicted probabilities.\n\n"
                f"Below you will find {len(X_fps)} example patients with false positive assessments to assist you in prescribing changes to the predict_hypertension function:\n"
            )
            message += f"\n{X_fps.to_json(orient='records', lines=True)}\n"

        if use_fn_example:
            message += (
                f"# Analysis of False Negatives\n\n"
                f"Please refine the function so that the {fn} False Negatives have higher predicted probabilities.\n\n"
                f"Below you will find {len(X_fns)} example patients with false negative assessments to assist you in prescribing changes to the predict_hypertension function:\n"
            )
            message += f"\n{X_fns.to_json(orient='records', lines=True)}\n"

        message += "\n".join(
            [
                "# Summary of Request\n",
                "Please create an updated Python function named `predict_hypertension` that achieves fewer false positives and fewer false negatives than the one you previously provided. "
                # "Where possible, rather than adding additional logic and calculations, try to reuse and refine the existing logic in the previous function."
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
        filename = self.filename.replace(".py", f"{uuid.uuid4()}.py")
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
            prob = np.zeros(len(X))

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
