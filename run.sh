#!/bin/bash
# source activate base
# conda activate htn-cp-llm

queue="bch-compute" # your slurm queue name, if applicable. to run locally --- instead of using slurm --- just use the --local flag
rdir="results"

# Switch the `if false` clausules to run the experiments

# Running the LLM experiment ---------------------------------------------------
# Run these first. Then, when running the comparison with other methods (next experiment),
# just specify the models and it will load them from this execution.
if true; then
    # GPT_4_turbo_Classifier,GPT_o3_mini_Classifier
    # Dont run it with several jobs --- they will share the solution between different folders
    # (as openAI models doesnt rely on data, so folds or scaling doesnt affect the generated model).
    # Running sequentially will allow it to reuse a previously found model, avoiding several
    # calls to the API.
    # For slurm, run first with Fold A. When they are done, run with other Folds
    python submit_jobs.py \
        -n_trials 10 \
        -n_jobs 1 \
        -folds A,B,C,D,E \
        -models "GPT_4o_mini_Classifier,GPT_4o_Classifier,GPT_35_Classifier" \
        -data-dir $(pwd)/data/ \
        -results_dir $rdir \
        -few_features [True,False] \
        -prompt_richnesses [True,False] \
        -time "1:00" \
        -queue $queue 
fi;

if true; then
    # This one can actually run in parallel, as they dont share  solutions
    # for the iter versions the folds are important, they depend on the data
    python submit_jobs.py \
        -n_trials 10 \
        -n_jobs 1 \
        -folds A,B,C,D,E \
        -models "GPT_4o_mini_iterative_Classifier,GPT_4o_iterative_Classifier,GPT_35_iterative_Classifier" \
        -data-dir $(pwd)/data/ \
        -results_dir $rdir \
        -few_features [True,False] \
        -prompt_richnesses [True,False] \
        -time "1:00" \
        -queue $queue 
fi;

models=()

# # Feat (reference model)
models+=("FeatBoolean")

# # Other ML models
# out of official experiments: GaussianNaiveBayes,LogisticRegression_L2
models+=("DecisionTree,LogisticRegression_L1,RandomForest")

if false; then
    for model in "${models[@]}"
    do
        # n trials X n folds X n targets X n ml models
        # n_trials --> one random seed for each trial
        # n_jobs   --> number of parallel jobs submitted. If a model is using more than 1 thread, this is not controlled here.
        # folds    --> different train-test splits.
        
        # --scale_data --> will do what it says

        # prompt-richness is redundant for non-llm models. Avoid running experiments 
        # with this flag if there is no LLM in the benchmark.

        # To load previous models for the LLMs, we need to use the same results folder.

        # to run locally --- instead of using slurm --- just use the --local flag
        python submit_jobs.py -models "$model" -n_trials 5 -n_jobs 20 \
            -data-dir $(pwd)/data/ -results $rdir \
            -folds A,B,C,D,E --prompt-richness --local
        
        # slurm (use n_jobs=1 in this case)
        # python submit_jobs.py -models "$model" -n_trials 10 -n_jobs 1 \
        #     -data-dir ./data/ -folds A -time 48:00 -m 6000 --slurm 
    done;
fi;
