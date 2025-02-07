# Learning Computational Phenotypes for Treatment Resistant Hypertension with Large Language Models

Generating hypertension computable phenotypes with LLMs

-----

## Setting up the environment

Simply set the conda environment

```bash
source activate base

# set our conda environment
if conda info --envs | grep -q htn-cp-llm;
    then echo "htn-cp-llm env already exists";
    else conda env create -f environment.yml;
fi
```

## Running the experiments

```bash
conda activate htn-cp-llm
bash run.sh
```
