import numpy as np                                                                   
from feat import FeatClassifier                                                                

clf = FeatClassifier(
           max_depth=6,                                                              
           max_dim = 10,
           objectives= ["fitness","size"], # "complexity"
           sel='lexicase',
           gens = 200,
           pop_size = 1000,
           # max_stall = 20,
           stagewise_xo = True,
           scorer='log',
           verbosity=2,
           shuffle=True,
           ml='LR',
           fb=0.5,
           n_jobs=1,
           classification=True,
           functions= ["split","and","or","not","b2f"],
           split=0.8, # percentage of data used in train split
           normalize=False,
           corr_delete_mutate=True, 
           simplify=0.005) 

name = 'FeatBoolean'