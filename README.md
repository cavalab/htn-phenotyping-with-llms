# Iterative Learning of Computable Phenotypes for Treatment Resistant Hypertension using Large Language Models

Generating hypertension computable phenotypes with LLMs.

Here we investigate whether LLMs can generate accurate and concise CPs for six clinical phenotypes of varying complexity, which could be leveraged to enable scalable clinical decision support to improve care for patients with hypertension.
In addition to evaluating zero-short performance, we propose and test a synthesize, execute, debug, instruct strategy that uses LLMs to generate and iteratively refine CPs using data-driven feedback.

Repository for the paper [Iterative Learning of Computable Phenotypes for Treatment Resistant Hypertension using Large Language Models. Guilherme Seidyo Imai Aldeia, Daniel S Herman, William La Cava Proceedings of the 10th Machine Learning for Healthcare Conference, PMLR 298, 2025.](https://proceedings.mlr.press/v298/aldeia25a.html).

The preprint version is available in [ArXiv](https://arxiv.org/abs/2508.05581).

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

## Post-processing analysis

Once you fish running the experiments, the notebooks inside `./notebooks/` folder can be used to generate the figures in the paper.
