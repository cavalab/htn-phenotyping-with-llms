# Import packages
import pandas as pd
import seaborn as sns

import os

import matplotlib

matplotlib.rc("pdf", fonttype=42)
matplotlib.rc("ps", fonttype=42)


# pd.set_option("display.max_colwidth", None)
sns.set_style("ticks", {"legend.frameon": True})  # style='whitegrid', palette='magma'
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1.1})

data_dir = '../data'
results_path = "../results"
paper_dir = "../paper"
poster_dir = "../poster"

for p in [paper_dir, poster_dir, f"{paper_dir}/final_models"]:
    if not os.path.exists(p):
        os.makedirs(p)

targets = {
    "htn_dx_ia": "Htndx",
    "res_htn_dx_ia": "ResHtndx",
    "htn_hypok_dx_ia": "HtnHypoKdx",
    "HTN_heuristic": "HtnHeuri",
    "res_HTN_heuristic": "ResHtnHeuri",
    "hypoK_heuristic_v4": "HtnHypoKHeuri",
}
heuristics = {
    "Htndx": "HTN_heuristic",
    "ResHtndx": "res_HTN_heuristic",
    "HtnHypoKdx": "hypoK_heuristic_v4",
}
targets_rev = {v: k for k, v in targets.items()}

dnames = [
    "HtnHeuri",
    "HtnHypoKHeuri",
    "ResHtnHeuri",
    "Htndx",
    "HtnHypoKdx",
    "ResHtndx",
]
dnames_nice = [
    "HTN Heuristic",
    "Htn-Hypokalemia Heuristic",
    "Resistant HTN Heuristic",
    "HTN Diagnosis",
    "HTN-Hypokalemia Diagnosis",
    "Resistant HTN Diagnosis",
]
dnames_to_nice = {k: v for k, v in zip(dnames, dnames_nice)}
dnames_to_ugly = {v: k for k, v in zip(dnames, dnames_nice)}

# folds = ['ALL']
folds = ["A", "B", "C", "D", "E"]

nice_model_labels = {  # Comment out the ones you dont want to load
    "GPT_35_Classifier": "gpt-3.5-turbo",
    "GPT_35_iterative_Classifier": "gpt-3.5-turbo-iter",
    # 'GPT_4_turbo_Classifier':'gpt-4-turbo',
    # 'GPT_4_turbo_iterative_Classifier':'gpt-4-turbo-iter',
    "GPT_4o_Classifier": "gpt-4o",
    "GPT_4o_iterative_Classifier": "gpt-4o-iter",
    "GPT_4o_mini_Classifier": "gpt-4o-mini",
    "GPT_4o_mini_iterative_Classifier": "gpt-4o-mini-iter",
}
model_nice = list(nice_model_labels.values())
models = list(nice_model_labels.keys())
nice_to_ugly = {v: k for k, v in nice_model_labels.items()}

markers = (
    "^",
    "o",
    "s",
    "S",
    "p",
    "P",
    "h",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
    "*",
)

# Comment out a model here to hide it in the plots
order = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-iter",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4o-mini-iter",
    "gpt-4o-iter",
    "gpt-4-turbo-iter",
]

marker_choice = {
    "gpt-3.5-turbo": "v",
    "gpt-3.5-turbo-iter": ".",
    "gpt-4-turbo": "^",
    "gpt-4-turbo-iter": "X",
    "gpt-4o": "o",
    "gpt-4o-iter": "D",
    "gpt-4o-mini": "d",
    "gpt-4o-mini-iter": "p",
}

# Use models generated with or without scaled data

results = []
