import json
import os
import sys
import pickle
import time
import re
import argparse
import importlib

import pandas as pd
import numpy as np

# import sage

from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import feature_selection

targets = {
    "htn_dx_ia": "Htndx",
    "res_htn_dx_ia": "ResHtndx",
    "htn_hypok_dx_ia": "HtnHypoKdx",
    "HTN_heuristic": "HtnHeuri",
    "res_HTN_heuristic": "ResHtnHeuri",
    "hypoK_heuristic_v4": "HtnHypoKHeuri",
}


def get_top_k_features(X, y, k=10):
    if y.ndim == 2:
        y = y[:, 0]
    if X.shape[1] <= k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])


# Function to count leaves
def count_sklearn_tree_leaves(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    def is_leaf(node_id):
        if children_left[node_id] == -1 and children_right[node_id] == -1:
            return 1
        return 0

    leaf_count = sum(is_leaf(i) for i in range(n_nodes))

    return leaf_count


def logistic_regression_summary(estimator, feature_names=None):
    # coefs = estimator.coef_[0]
    coefs = estimator.named_steps["est"].coef_[0]

    # Combine feature names and coefficients
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(len(coefs))]

    feature_coef_pairs = list(zip(feature_names, coefs))

    sorted_pairs = sorted(feature_coef_pairs, key=lambda x: abs(x[1]), reverse=True)

    summary = "Logistic reg. weights:"
    for feature, coef in sorted_pairs:
        summary += f"{coef:.2f}*{feature},"

    return summary


def read_data(
    target,
    fold,
    repeat,
    scale,
    few_feature,
    data_dir,
    random_state=42,
    shuffle_with_rs=True,
):
    """Read in data, setup training and test sets"""

    # Removing targets
    drop_cols = ["UNI_ID"] + list(targets.keys())

    # print("targets:", targets)
    # setup target

    target_new = targets[target]

    ddir = os.path.join(data_dir, "Dataset" + str(repeat), target_new)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    if fold == "ALL":
        for f in ["A", "B", "C", "D", "E"]:
            df_train = pd.concat(
                [df_train, pd.read_csv(ddir + "/" + target_new + f + "Train.csv")]
            )
            df_test = pd.concat(
                [df_test, pd.read_csv(ddir + "/" + target_new + f + "Test.csv")]
            )
    else:
        df_train = pd.read_csv(ddir + "/" + target_new + fold + "Train.csv")
        df_test = pd.read_csv(ddir + "/" + target_new + fold + "Test.csv")

    # Training
    df_y = df_train[target].astype(int)

    # setup predictors
    df_X = df_train.drop(drop_cols, axis=1)
    feature_names = df_X.columns

    # print("feature names:", feature_names)

    # label encode
    # print("X train info:")
    # df_X.info()

    assert not df_X.isna().any().any()

    X_train = df_X
    y_train = df_y

    # Using less data to train (faster execution for sanity checks)
    subsample = None
    if subsample is not None:
        rng = np.random.default_rng(seed=random_state)
        indexes = rng.choice(
            int(subsample * X_train.shape[0]),
            replace=False,
            size=int(subsample * X_train.shape[0]),
        )
        X_train = X_train.iloc[indexes, :]
        y_train = y_train.iloc[indexes]

    if shuffle_with_rs:
        rng = np.random.default_rng(seed=random_state)
        indexes = rng.choice(y_train.shape[0], replace=False, size=y_train.shape[0])
        X_train = X_train.iloc[indexes, :]
        y_train = y_train.iloc[indexes]

    # Testing
    df_y = df_test[target].astype(int)

    # setup predictors
    df_X = df_test.drop(drop_cols, axis=1)

    feature_names = df_X.columns
    # print("feature names:", feature_names)
    # label encode

    # print("X test info:")
    # df_X.info()
    
    assert not df_X.isna().any().any()

    X_test = df_X
    y_test = df_y

    # Duplicate column
    X_train = X_train.drop(columns=["HTN_MED_days_ALDOSTERONE_ANTAGONIST"])
    X_test = X_test.drop(columns=["HTN_MED_days_ALDOSTERONE_ANTAGONIST"])

    # Filtering subset of features
    if few_feature:
        htn_variables = [
            "mean_systolic",
            "mean_diastolic",
            "median_systolic",
            "median_diastolic",
            "bp_n",
            "high_bp_n",
            "high_BP_during_htn_meds_1",
            "high_BP_during_htn_meds_2",
            "high_BP_during_htn_meds_3",
            "high_BP_during_htn_meds_4_plus",
            "sum_enc_during_htn_meds_4_plus",
            "low_K_N",
            "test_K_N",
            "Med_Potassium_N",
            "Dx_HypoK_N",
            "re_htn_sum",
        ]

        X_train = X_train[htn_variables]
        X_test = X_test[htn_variables]

    if scale:
        # scl = StandardScaler().fit(X_train)
        scl = MinMaxScaler().fit(X_train)

        X_train = scl.transform(X_train.values)
        X_test = scl.transform(X_test.values)

        X_train = pd.DataFrame(X_train, columns=df_X.columns)
        X_test = pd.DataFrame(X_test, columns=df_X.columns)

    # X_train.dtypes.to_csv(f'{target}_dtypes.csv')

    return X_train, y_train, X_test, y_test


def evaluate_model(
    estimator,
    name,
    target,
    fold,
    random_state,
    rdir,
    repeat,
    scale,
    few_feature,
    prompt_richness,
    data_dir="./",
):
    """Evaluates estimator by training and predicting on the target."""

    os.makedirs(rdir, exist_ok=True)

    X_train, y_train, X_test, y_test = read_data(
        target, fold, repeat, scale, few_feature, data_dir, random_state
    )
    feature_names = X_train.columns
    # stripping the dataframe
    if "ps-tree" in name.lower():
        X_train, X_test = X_train.values, X_test.values
        y_train, y_test = y_train, y_test

    # set random states
    if hasattr(estimator, "random_state"):
        estimator.random_state = random_state
    elif hasattr(estimator, "seed"):
        estimator.seed = random_state

    if "GPT" in name:
        estimator.set_prompt(target=target, richness=prompt_richness)

        # To avoid loading previously generated models, save the file name with
        # the fold, and dont use the t_group (use the actual target instead), so
        # the names won't match

        # fold and scale does not make difference here, since it is not based on data to generate the models.
        t_group = None
        if target in ["htn_dx_ia", "HTN_heuristic"]:
            t_group = "HTN"
        elif target in ["res_htn_dx_ia", "res_HTN_heuristic"]:
            t_group = "res_HTN"
        elif target in ["htn_hypok_dx_ia", "hypoK_heuristic_v4"]:
            t_group = "hypok_HTN"
        else:
            raise Exception("Unknown target")

        for f in [
            target,  # use 't_group' to recycle dx/htn models. Otherwise use 'target'
            name,
            # str(repeat),
            str(random_state),
            # str(fold),
            str(few_feature),
            str(prompt_richness),
            "program",
        ]:
            print(f)
        # This one will share programs across folds, and dx/heuri targets
        estimator.set_file(
            os.path.join(
                rdir,
                "_".join(
                    [
                        target,  # use 't_group' to recycle dx/htn models. Otherwise use 'target'
                        name,
                        # str(repeat),
                        str(random_state),
                        (str(fold) if "iterative" in name else ""), # for the iter versions the folds are important, they depend on the data
                        str(few_feature),
                        str(prompt_richness),
                        "program",
                    ]
                )
                + ".py",
            )
        )

    print("fitting to all data...")
    t0t = time.time()

    estimator.fit(X_train, y_train)

    fit_time = time.time() - t0t
    print("Training time measure:", fit_time)

    y_pred = estimator.predict(X_test)
    y_pred_proba = estimator.predict_proba(X_test)[:, 1]

    # models.append(estimator)
    # estimator.fit(X,y)
    if type(estimator).__name__ == "GridSearchCV":
        estimator = estimator.best_estimator_

    model = model_fmt = 0

    ### estimator-specific routines
    if "feat" in name.lower():
        pattern = r"[+-]?\d+(?:\.\d+)?\*"

        def count_terminal_weights(text):
            matches = re.findall(pattern, text)
            valid_floats = []
            print(matches)
            for match in matches:
                try:
                    float(match[:-1])
                    valid_floats.append(match)
                except ValueError:
                    pass

            return len(valid_floats)

        print("representation:\n", estimator.get_representation())
        print("model:\n", estimator.get_model())
        model = estimator.get_model()
        model_fmt = estimator.get_model()

        size = (
            estimator.get_n_nodes()  # actual nodes in the expression
            + 2 * count_terminal_weights(estimator.get_representation())  # w * <>
            + 2 * len(estimator.get_coefs())  # + coef * <>
            + 2
        )  # + offset
        complexity = size  # estimator.get_complexity()

        # older versions of feat (i.e. the git checkout in setup)
        # size = estimator.stats_['med_size'][-1]
        # complexity = estimator.stats_['med_complexity'][-1]

        filename = (
            rdir
            + "/"
            + "_".join(
                [
                    targets[target],
                    name,
                    str(repeat),
                    str(random_state),
                    str(fold),
                    str(scale),
                    str(few_feature),
                    str(prompt_richness),
                    ".pkl",
                ]
            )
        )

        if "GPT" not in name:
            pickle.dump(estimator, open(filename, "wb"))

        if "LogisticRegression" in name:
            print("best C:", estimator.named_steps["est"].C_)
            # Multiply by 3: w * feature
            size = 3 * np.count_nonzero(estimator.named_steps["est"].coef_[0])

            # Add sum between each feature
            size = size + np.count_nonzero(estimator.named_steps["est"].coef_[0]) - 1

            model = model_fmt = logistic_regression_summary(estimator, feature_names)
        elif "RandomForest" in name:
            size = 0
            for i in estimator.estimators_:
                # Each inner node has size 2: the feature and the comparison operator,
                size += 2 * i.tree_.node_count

                # each leaf has size 1, so we will deduct 1
                size -= count_sklearn_tree_leaves(i)
        elif "DecisionTree" in name:
            size = 2 * estimator.tree_.node_count
            size = size - count_sklearn_tree_leaves(estimator)
        elif "GPT" in name:
            model = model_fmt = estimator.get_model()
            size = complexity = estimator._count_operations()
        else:
            size = len(X_train.columns)

        complexity = size

    # get scores
    results = {}
    scorers = [
        accuracy_score,
        precision_score,
        average_precision_score,
        roc_auc_score,
        balanced_accuracy_score,
    ]
    for X, y, part in zip([X_train, X_test], [y_train, y_test], ["_train", "_test"]):
        for scorer in scorers:
            col = scorer.__name__ + part
            print(col)

            try:
                if scorer in [average_precision_score, roc_auc_score]:
                    results[col] = scorer(y, estimator.predict_proba(X)[:, 1])
                else:
                    results[col] = scorer(y, estimator.predict(X))
            except ValueError:
                results[col] = np.nan

    # Explaining with SAGE
    # imputer = sage.MarginalImputer(estimator.predict, X_test.values)
    # sampler = sage.PermutationEstimator(imputer, 'cross entropy')
    # sage_values = sampler(X_test.values, Y=y_test.values)

    if "GPT" in name:
        results['messages'] = estimator.messages_
    
    results["model"] = name
    results["target"] = targets[target]
    results["fold"] = fold
    results["RunID"] = repeat
    results["random_state"] = random_state
    results["representation"] = model
    results["representation_fmt"] = r"{}".format(model_fmt)
    results["size"] = size
    results["complexity"] = complexity
    results["scale"] = scale
    results["few_feature"] = few_feature
    results["prompt_richness"] = prompt_richness
    results["time"] = fit_time
    results["pred"] = y_pred.tolist()
    results["pred_proba"] = y_pred_proba.tolist()
    # results['sage_mean'] = sage_values.values.tolist()
    # results['sage_std'] = sage_values.std.tolist()

    print("results:", results)

    filename = (
        rdir
        + "/"
        + "_".join(
            [
                targets[target],
                name,
                str(repeat),
                str(random_state),
                str(fold),
                str(scale),
                str(few_feature),
                str(prompt_richness),
            ]
        )
    )

    with open(f"{filename}.json", "w") as out:
        json.dump(results, out, indent=4)

    return estimator, results


################################################################################
# main entry point
################################################################################

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=True
    )
    # parser.add_argument('-h', '--help', action='help',
    #                     help='Show this help message and exit.')
    parser.add_argument(
        "-ml",
        action="store",
        dest="ALG",
        default=None,
        type=str,
        help="Name of estimator (with matching file in methods/)",
    )
    parser.add_argument(
        "-rdir",
        action="store",
        dest="RDIR",
        default=None,
        type=str,
        help="Name of save file",
    )
    parser.add_argument(
        "-seed",
        action="store",
        dest="RANDOM_STATE",
        default=None,
        type=int,
        help="Seed / trial",
    )
    parser.add_argument(
        "-repeat",
        action="store",
        dest="REPEAT",
        default=1,
        type=int,
        help="repetition number",
    )
    parser.add_argument(
        "-fold", action="store", dest="FOLD", default=None, type=str, help="CV fold"
    )
    parser.add_argument(
        "--scale_data",
        action="store_true",
        dest="SCALE",
        help="Wether if we should scale the data",
    )
    parser.add_argument(
        "--icd-only",
        action="store_true",
        dest="few_feature",
        help="Use only ICD-related features for hypertension problems",
    )
    parser.add_argument(
        "--prompt-richness",
        action="store_true",
        dest="PROMPT_RICHNESS",
        help="Whether if the prompt should contain detailed information about the phenotype. Ignored if the model is not an OpenAI classifier",
    )
    parser.add_argument(
        "-target",
        action="store",
        dest="TARGET",
        default=None,
        type=str,
        help="endpoint name",
        choices=[
            "htn_dx_ia",
            "res_htn_dx_ia",
            "htn_hypok_dx_ia",
            "HTN_heuristic",
            "res_HTN_heuristic",
            "hypoK_heuristic_v4",
        ],
    )
    parser.add_argument(
        "-datadir",
        action="store",
        dest="DDIR",
        default="./",
        type=str,
        help="input data directory",
    )

    args = parser.parse_args()
    print(args)

    # import algorithm
    print("import from", "models." + args.ALG)
    algorithm = importlib.__import__(
        "models." + args.ALG, globals(), locals(), ["clf", "name"]
    )

    print("algorithm:", algorithm.name, algorithm.clf)
    evaluate_model(
        algorithm.clf,
        algorithm.name,
        args.TARGET,
        args.FOLD,
        args.RANDOM_STATE,
        args.RDIR,
        args.REPEAT,
        args.SCALE,
        args.few_feature,
        args.PROMPT_RICHNESS,
        data_dir=args.DDIR,
    )
