import numpy as np                                                                   
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = Pipeline([
            ('prep', StandardScaler()),
            ('est', LogisticRegressionCV(Cs = np.logspace(-6,3,10),
                           penalty='l2',
                           solver = 'liblinear')
                           )
            ])

name = 'LogisticRegression_L2'
