#%%
import numpy.linalg as alg
import numpy as np

class MultinomialLogisticRegression:
    def __init__(self):
        self.beta=None

    def fit(self,X,y,lr,n_iter):
        y_2=set(list(y)) # on garde que les labels qu'on suppose numérique
        n_class = len(y_2)
        self.beta=np.zeros((X.shape[1],n_class))
        


        y_onehot = np.zeros((y.shape[0], n_class))
        y_onehot[np.arange(y.shape[0]), y] = 1 # indicator matrix
        
        for n in range(n_iter):
            Z = X @ self.beta 
            sum_exp_Z = np.sum(np.exp(Z), axis=1, keepdims=True)
            P = np.exp(Z) / sum_exp_Z # proba matrix
            grad = X.T @ (y_onehot - P)
            self.beta = self.beta + lr*grad
        
    def predict_proba(self,X):
        Z = X @ self.beta
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))   # to avoid overflow
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return P 

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1) 




# %%
