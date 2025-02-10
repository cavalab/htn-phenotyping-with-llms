def set_init_messages(phenotype, func_return, X_variable_dict):
    return [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that generates Python code based on a plain-text description of a function's purpose. "
                "You will receive a statement describing what the function should do. "
                "Your response must contain only a Python function, with no comments or explanations, that strictly follows the given description."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please create a Python function named `predict_hypertension` that takes a pandas DataFrame named `df` as input. "
                f"The function should assess whether each patient (represented as rows) has evidence of {phenotype}. "
                f"The DataFrame `df` only contains potential risk factors for {phenotype.split(',')[0]}.\n"
                f"The function must return an array of {func_return} for each row. "
                "The available columns and their meanings are provided as key value pairs in the dictionary included below."
                "The function must ONLY use the features whose names appear in this dictionary."
                "The function MUST NOT assume access to any addtional inputs, such as training labels.\n"
                f"The dictionary of available columns is as follows:\n`{str(X_variable_dict)}`"
            ),
        },
    ]


cfg = dict(
    temperature=0.5,
    top_p=1.0,
    max_tokens=4096,
    init_messages=set_init_messages,
    func_return="floats representing the probability, between 0 and 1,",
)
import numpy as np


def default_cp_function(df):
    return np.zeros(len(df))
