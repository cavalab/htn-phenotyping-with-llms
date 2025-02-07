import numpy as np                                                                   
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

name = 'DecisionTree'

max_features = ['auto', 'sqrt']
max_features.append(None) 
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
dt = DecisionTreeClassifier()
inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
clf = GridSearchCV(estimator = dt, param_grid = grid, 
                    cv = inner_cv, verbose=2, n_jobs = 1, refit=True)
