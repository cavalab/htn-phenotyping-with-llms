This is where your data is suposed to be.

As stated in the paper, ''We use data from EHR clinical data repository Penn Data Store and EPIC Clarity reporting database.
The data in this article cannot be shared publicly to protect the privacy of the subjects.
However, upon request and subject to appropriate approvals, it will be shared by the corresponding author.``

However, to run with your own data, you need to change some things:

1. The contents of the `data` folder;
2. the `feature_descriptions.py` file, wiith descriptions for your data;
3. The `submit_jobs.py` file, to change which features will be used with the `few_features` flag;
4. Use the flag `-targets` when calling the `submiit_jobs.py` from `run.sh`;
5. The prompt messages in `models/opena_cfg.py`, `models/OpenAI_HTN_Classifier.py`, , `models/OpenAI_HTN_Iterative_Classifier.py` to match your definitions.

When changing the content of this folder, data is expected in the following format:

A folder with the target col name will contain the train and test partitions. To create the folds, you can specify the name of the fold in the filename. 

Look at how we load the data:

```python
ddir = os.path.join(data_dir, "Dataset" + str(repeat), target_name)

df_train = pd.DataFrame()
df_test = pd.DataFrame()
if fold == "ALL":
    for f in ["A", "B", "C", "D", "E"]:
        df_train = pd.concat(
            [df_train, pd.read_csv(ddir + "/" + target_name + f + "Train.csv")]
        )
        df_test = pd.concat(
            [df_test, pd.read_csv(ddir + "/" + target_name + f + "Test.csv")]
        )
else:
    df_train = pd.read_csv(ddir + "/" + target_name + fold + "Train.csv")
    df_test = pd.read_csv(ddir + "/" + target_name + fold + "Test.csv")
```

