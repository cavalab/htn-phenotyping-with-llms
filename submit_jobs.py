import os
import subprocess
import itertools as it
import fire

from joblib import Parallel, delayed


def run(
    targets=[
        "htn_dx_ia",
        "res_htn_dx_ia",
        "htn_hypok_dx_ia",
        "HTN_heuristic",
        "res_HTN_heuristic",
        "hypoK_heuristic_v4",
    ],
    folds=["A", "B", "C", "D", "E"],
    models=["GPT_4o_mini_Classifier", "GPT_4o_Classifier", "GPT_35_Classifier"],
    long=False,
    seeds=[
        14724,
        24284,
        31658,
        6933,
        1318,
        16695,
        27690,
        8233,
        24481,
        6832,
        13352,
        4866,
        12669,
        12092,
        15860,
        19863,
        6654,
        10197,
        29756,
        14289,
        4719,
        12498,
        29198,
        10132,
        28699,
        32400,
        18313,
        26311,
        9540,
        20300,
        6126,
        5740,
        20404,
        9675,
        22727,
        25349,
        9296,
        22571,
        2917,
        21353,
        871,
        21924,
        30132,
        10102,
        29759,
        8653,
        18998,
        7376,
        9271,
        9292,
    ],
    results_dir="results/",
    data_dir="./",
    n_trials=1,
    n_jobs=1,
    scale_datas=[False],
    few_features=[False, True],
    prompt_richnesses=[False, True],
    local=False,
    job_limit=5000,
    script="evaluate_model",
    m=4096,
    time="01:00",
    max_jobs=3500,
    queue="",
):
    n_trials = len(seeds) if n_trials < 1 else n_trials

    print("n_trials: ", n_trials)
    print("n_jobs: ", n_jobs)

    seeds = seeds[:n_trials]

    print("using these seeds:", seeds)
    print("for folds:", folds)

    print("for models:", models)

    print("and these targets:", targets)

    print(f"scale data is {scale_datas}")
    print(f"icd only is {few_features}")
    print(f"Prompt richness is {prompt_richnesses}")

    # name of the column
    targets = {
        "htn_dx_ia": "Htndx",
        "res_htn_dx_ia": "ResHtndx",
        "htn_hypok_dx_ia": "HtnHypoKdx",
        "HTN_heuristic": "HtnHeuri",
        "res_HTN_heuristic": "ResHtnHeuri",
        "hypoK_heuristic_v4": "HtnHypoKHeuri",
    }

    current_jobs = []
    if not local:
        lpc_options = ""
        res = subprocess.check_output(['squeue -o "%j"'], shell=True)
        current_jobs = res.decode().split("\n")

    all_commands = []
    job_info = []
    jobs_w_results = []
    queued_jobs = []

    for target, ml, few_feature, prompt_richness, seed, fold, scale in it.product(
        targets, models, few_features, prompt_richnesses, seeds, folds, scale_datas
    ):
        filepath = "/".join([results_dir, target, ml]) + "/"
        if not os.path.exists(filepath):
            print("WARNING: creating path", filepath)
            os.makedirs(filepath)
        random_state = str(seed)
        save_file = ""
        if script == "evaluate_model":
            save_file = filepath + "_".join(
                [
                    targets[target],
                    ml,
                    random_state,
                    str(fold),
                    str(scale),
                    str(few_feature),
                    str(prompt_richness),
                ]
            )
        else:
            save_file = filepath + "_".join(
                [
                    targets[target],
                    ml,
                    random_state,
                    str(scale),
                    str(prompt_richness),
                ]
            )
        print(save_file)
        # check if there is already a result for this experiment
        if os.path.exists(save_file + ".json"):
            jobs_w_results.append([save_file, "exists"])
            continue
        # check if there is already a queued job for this experiment
        if save_file.split("/")[-1] in current_jobs:
            queued_jobs.append([save_file, "queued"])
            continue

        all_commands.append(
            f" python {script}.py "
            f" -ml {ml}"
            f" -target {target}"
            f" -seed {random_state}"
            f" -rdir {filepath}"
            f" -fold {fold}"
            f" -datadir {data_dir}"
            f" {'--scale_data' if scale else ''}"
            f" {'--icd-only' if few_feature else ''}"
            f" {'--prompt-richness' if prompt_richness else ''}"
        )
        job_info.append(
            {
                "ml": ml,
                "target": target,
                "fold": fold,
                "scale": str(scale),
                "few_feature": str(few_feature),
                "prompt_richness": str(prompt_richness),
                "results_path": filepath,
                "seed": random_state,
            }
        )

        # print(job_info[-1])

    print("skipped", len(jobs_w_results), "jobs with results.")
    print("skipped", len(queued_jobs), "queued jobs.")

    if len(all_commands) > job_limit:
        print("shaving jobs down to job limit ({})".format(job_limit))
        all_commands = all_commands[:job_limit]

    print("submitting", len(all_commands), "jobs...")
    input("Press Enter to continue...")

    if local:  # run locally
        Parallel(n_jobs=n_jobs)(delayed(os.system)(run_cmd) for run_cmd in all_commands)
    # delayed(print)(run_cmd) for run_cmd in all_commands)
    else:
        # sbatch
        for i, run_cmd in enumerate(all_commands):
            job_name = None
            if script == "evaluate_model":
                job_name = "_".join(
                    [
                        job_info[i]["target"],
                        job_info[i]["ml"],
                        job_info[i]["seed"],
                        job_info[i]["fold"],
                        job_info[i]["scale"],
                        job_info[i]["few_feature"],
                        job_info[i]["prompt_richness"],
                    ]
                )
            else:
                job_name = "_".join(
                    [
                        job_info[i]["target"],
                        job_info[i]["ml"],
                        job_info[i]["seed"],
                        job_info[i]["scale"],
                        job_info[i]["few_feature"],
                        job_info[i]["prompt_richness"],
                    ]
                )

            out_file = job_info[i]["results_path"] + job_name + ".%J.out"

            batch_script = "\n".join(
                [
                    "#!/usr/bin/bash ",
                    f"#SBATCH -o {out_file} ",
                    "#SBATCH -N 1 ",
                    f"#SBATCH -n {n_jobs} ",
                    f"#SBATCH -J {job_name} ",
                    f"#SBATCH -p {queue} ",
                    f"#SBATCH --ntasks-per-node=1 --time={time}:00 ",
                    f"#SBATCH --mem-per-cpu={m} ",
                    "",
                    "source .openai_api_key",
                    "",
                    f"{run_cmd}",
                ]
            )
            with open("tmp_script", "w") as f:
                f.write(batch_script)

            # print(batch_script)
            print(job_name)
            sbatch_response = subprocess.check_output(
                ["sbatch tmp_script"], shell=True
            ).decode()  # submit jobs
            print(sbatch_response)

    print("Finished submitting", len(all_commands), "jobs.")


if __name__ == "__main__":
    fire.Fire(run)
