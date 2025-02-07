import numpy as np                                                                   
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

name = 'RandomForest'

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2100, num = 6)]
max_features = ['auto', 'sqrt']
max_features.append(None)
max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
max_depth.append(None)

# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 3)]
# max_features = [None]
# max_depth = [int(x) for x in np.linspace(3, 10, num = 3)]

grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth':    max_depth}

rf = RandomForestClassifier()
inner_cv = StratifiedKFold(n_splits=3, shuffle=True)
clf = GridSearchCV(estimator = rf, param_grid = grid, 
                    cv = inner_cv, verbose=2, n_jobs = 1, refit=True)
