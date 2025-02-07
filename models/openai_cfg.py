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
                f"\nThe function MUST NOT assume access to the target variable; "
                f"the DataFrame `df` only contains potential risk factors for {phenotype.split(',')[0]}.\n"
                f"The function must return an array of {func_return} for each row. "
                f"The available columns and their meanings are provided as key value pairs in the following dictionary: `{str(X_variable_dict)}`. "
                "You may only use the features whose names appear in this dictionary."
            ),
        }
    ]

cfg = dict(
    temperature=0.3,
    top_p=0.25,
    max_tokens=4096,
    init_messages = set_init_messages
)
import numpy as np
def default_cp_function(df):
    return np.zeros(len(df))