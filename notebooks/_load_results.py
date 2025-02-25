# Import packages
import pandas as pd
import seaborn as sns

import os

import matplotlib

matplotlib.rc("pdf", fonttype=42)
matplotlib.rc("ps", fonttype=42)


# pd.set_option("display.max_colwidth", None)
sns.set(style="ticks", palette="colorblind")  # style='whitegrid', palette='magma'
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1.1})

results_path_not_llms = "../results_not-llms"
results_path_llms = "../results_paper_rebuttal"
paper_dir = "../paper_rebuttal"

if not os.path.exists(paper_dir):
    os.makedirs(paper_dir)

if not os.path.exists(f"{paper_dir}/final_models"):
    os.makedirs(f"{paper_dir}/final_models")

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

folds = ["A", "B", "C", "D", "E"]

nice_model_labels = {  # Comment out the ones you dont want to load
    "RandomForest": "RF",
    "DecisionTree": "DT",
    # 'GaussianNaiveBayes':'GNB',
    "FeatBoolean": "FEAT",
    "LogisticRegression_L2": "LR L2",
    "LogisticRegression_L1": "LR L1",
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
# models = [ # Comment out the ones you dont want to load into the results (here and in model_nice)
#     'RandomForest',
#     'DecisionTree',
#     # 'GaussianNaiveBayes',
#     'FeatBoolean',
#     # 'LogisticRegression_L2',
#     'LogisticRegression_L1',

#     'GPT_4_turbo_Classifier',
#     'GPT_4_turbo_iterative_Classifier',

#     'GPT_4o_Classifier',
#     'GPT_4o_iterative_Classifier',

#     'GPT_4o_mini_Classifier',
#     'GPT_4o_mini_iterative_Classifier',
# ]
# model_nice = [
#     'RF',
#     'DT',
#     # 'GNB',
#     'FEAT',
#     # 'LR L2',
#     'LR L1',

#     'gpt-4-turbo',
#     'gpt-4-turbo-iter',

#     'gpt-4o',
#     'gpt-4o-iter',

#     'gpt-4o-mini',
#     'gpt-4o-mini-iter',
# ]
# nice_model_labels = {k:v for k,v in zip(models,model_nice)}
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
    "gpt-4o-mini-iter",
    "gpt-4o",
    "gpt-4o-iter",
    # 'gpt-4-turbo',
    # 'gpt-4-turbo-iter',
    # 'GNB',
    "DT",
    # 'LR L2',
    "LR L1",
    "RF",
    "FEAT",
]

marker_choice = {
    "GNB": "^",
    "DT": "o",
    # 'LR L2'          : 's',
    "LR L1": "s",
    "RF": "p",
    "FEAT": "P",
    "gpt-3.5-turbo": "v",
    "gpt-3.5-turbo-iter": ".",
    "gpt-4-turbo": "^",
    "gpt-4-turbo-iter": "X",
    "gpt-4o": "o",
    "gpt-4o-iter": "D",
    "gpt-4o-mini": "d",
    "gpt-4o-mini-iter": "p",
}

# # Use models generated with or without scaled data
# scaled_data = False

# # Setting this as false so we can replicate feat's paper results
# icd_only = True

# # Using best setting for LLMs based on notebook 05 analysis
# prompt_richness = True

# results = []
# for target in targets:
#     for model in models:
#         for fold in folds:
#             for results_path in results_paths:
#                 if "GPT" in model:
#                     globby = glob(
#                         f"{results_path}/{target}/{model}/*_{fold}_{scaled_data}_{icd_only}_{prompt_richness}.json"
#                     )
#                 else:
#                     globby = glob(
#                         f"{results_path}/{target}/{model}/*_{fold}_{scaled_data}_{False}_True.json"
#                     )
#                 for file in globby:
#                     # skipping llm results --> they are trained with fold ALL
#                     df = pd.read_json(file, typ="series")

#                     indxs = df.index
#                     # indxs = [indx for indx in indxs if indx not in ['pred', 'pred_proba']]

#                     results.append(df[indxs])
# for results_path in [results_path_llms, results_path_not_llms]:
#     for file in glob(
#         f"{results_path}/**/*.json",
#         recursive=True,
#     ):
#         df = pd.read_json(file, typ="series")

#         indxs = df.index
#         # indxs = [indx for indx in indxs if indx not in ['pred', 'pred_proba']]

#         results.append(df[indxs])

# for file in glob(
#     f"{results_path_llms}/**/*{scaled_data}_{icd_only}_{prompt_richness}*.json",
#     recursive=True,
# ):
#     df = pd.read_json(file, typ="series")

#     indxs = df.index
#     # indxs = [indx for indx in indxs if indx not in ['pred', 'pred_proba']]

#     results.append(df[indxs])

# results_df = pd.DataFrame(data=results, columns=indxs)

# # Beautifying it
# results_df["model"] = results_df["model"].apply(lambda m: nice_model_labels[m])
# results_df["target"] = results_df["target"].apply(lambda t: dnames_to_nice[t])

# results_df = results_df[results_df["model"].isin(order)]

# print(results_df.shape)
# print(results_df["model"].unique())
# print(results_df["target"].unique())