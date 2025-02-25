"""Base for openAI based methods"""

import os
from dotenv import load_dotenv, find_dotenv
import importlib

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


# Implementing the class -------------------------------------------------------
class OpenAI_HTN_Classifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        phenotype="htn_dx_ia",
        model="gpt-4o-mini",
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        filename=None,
        random_state=None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.phenotype = phenotype
        self.filename = filename
        self.random_state = random_state

    def set_prompt(self, target="", richness=False):
        self.phenotype = cfg['phenotype_description'](target, richness)

    def set_file(self, filename):
        # Because fold doesnt matter, we have the option to save or load previous programs (avoid excessive openAPI calls).
        # So we are going to save the result and future calls will load it for a given seed (different seeds will load different programs,
        # so make sure to have temperature!=0.0 if running several repetitions)
        self.filename = filename

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

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "X must be a dataframe for openAI to work."

        X_variable_dict = {c: htn_variable_dict[c] for c in X.columns}

        # Asking for pred_proba
        func_return = cfg["func_return"]

        messages = cfg["init_messages"](self.phenotype, func_return, X_variable_dict)

        self.str_rep = None
        if (self.filename is not None) and (not os.path.exists(self.filename)):
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # generates text competion
            completion = client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=cfg["max_tokens"],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.random_state,
            )

            cp = completion.choices[0].message.content
            print(cp)

            self.str_rep = cp.split("```python")[1][:-3]

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

        print(self.str_rep)

        # def predict_proba_hypertension(df):
        #     raise Exception("Not parsed yet.")

        # Remove the ``` wrapping the definition
        self.messages_ = messages
        self.threshold_ = 0.5

        try:
            self.cp_function_ = self.get_cp_function_from_file()
            self.train_success_ = True
        except Exception:
            self.cp_function_ = default_cp_function
            self.train_success_ = False

        # self._find_threshold(X, y)

        return self

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] > self.threshold_, 1, 0)

    def predict_proba(self, X):
        check_is_fitted(self)

        assert isinstance(X, pd.DataFrame), "X must be a dataframe for openAI to work."

        try:
            prob = self.cp_function_(X)
            assert len(prob) == len(X)
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
