#%%
import numpy.linalg as alg
import numpy as np

class LinearRegression_grad:
    def __init__(self):
        self.beta=None
        
    
    def fit(self,X,y,lr=0.001,n_iter=6):
        self.beta = np.zeros(X.shape[1])
        for k in range(n_iter):
            grad = (2*X.T@X@self.beta - 2*X.T@y)
            self.beta = self.beta - lr*grad
        
    def predict(self,X):
        return X@self.beta
# %%
