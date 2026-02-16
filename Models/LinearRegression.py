#%%
import numpy.linalg as alg
import numpy as np
class LinearRegression:
    def __init__(self):
        self.beta=None
    
    def fit(self,X,y):
        self.beta = alg.pinv(X.T@X)@X.T@y
    
    def predict(self,X):
        return X@self.beta
# %%
