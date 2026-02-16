#%%
import numpy as np
import numpy.linalg as alg

class LogisticRegression:
    def __init__(self):
        self.beta=None
    
    def fit(self,X,y,lr,n_iter):
        self.beta=np.zeros(np.zeros(X.shape[1]))
        for k in range(n_iter):
            # minimizing the log-loss :
            # l(beta) = sum_i->n ( y_i*log(X_i@beta) + (1-y_i)*log(1-X_i@beta))
            grad = X.T@np.sigmoid(X@self.beta-y)
            self.beta = self.beta - lr*grad
    
    def predict(self,X):
        pred= np.sigmoid(X@self.beta) # probability for each sample that the label is 1.
        return (pred >= 0.5).astype(int) 
    

# %%
