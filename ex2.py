# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.
import numpy as np
import matplotlib.pyplot as plt
class Ols(object):
  def __init__(self):
    self.w = None
    
  @staticmethod
  def pad(X):
    return np.append(np.ones([X.shape[0],1]),X,axis=1)
  
  def fit(self, X, Y):
    #remember pad with 1 before fitting
    self._fit(self.pad(X), Y)    
  
  def _fit(self, X, Y):
    # optional to use this
    self.w = np.matmul(np.linalg.pinv(X), Y)
  
  def predict(self, X):
    #return w
    return self._predict(self.pad(X))

  def _predict(self, X):
    #return w
    # optional to use this
    return np.matmul(X , self.w )
    
  def score(self, X, Y):
    #return MSE
    return ((X-Y)**2).mean()


# Write a new class OlsGd which solves the problem using gradinet descent. 
# The class should get as a parameter the learning rate and number of iteration. 
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate? 
# How would you find number of iteration automatically? 
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.
class Normalizer():
  def __init__(self):
    self._x_max = None
    self._x_min = None

  def fit(self, X):
    self._x_max = X.max(axis=0)
    self._x_min = X.min(axis=0)
    self._x_min[self._x_max == self._x_min] = 0

  def predict(self, X):
    #apply normalization
    return (X - self._x_min)/(self._x_max - self._x_min) 
    
class OlsGd(Ols):
  
  def __init__(self, learning_rate=.0005, 
               num_iteration=1000, 
               normalize=True,
               early_stop=True,
               verbose=True,
               track_loss=False):
    
    super(OlsGd, self).__init__()
    self.learning_rate = learning_rate
    self.num_iteration = num_iteration
    self.early_stop = early_stop
    self.normalize = normalize
    self.normalizer = Normalizer()    
    self.verbose = verbose
    self.track_loss = track_loss
    
  def _fit(self, X, Y, reset=True):
    #remember to normalize the data before starting
    losses =[]
    if self.normalize:
      self.normalizer.fit(X)
      X = self.normalizer.predict(X)
    self.w = np.zeros(X.shape[1])
    # self.w = np.random.rand(X.shape[1])
    for epoch in range(self.num_iteration):
      losses.append(self.score(self.__predict(X),Y))
      self._step(X,Y)
      if (epoch > 2) & (self.early_stop):
        if losses[-1] > losses[-2]:
          break 
      if self.verbose:
        print(f'Epoch={epoch}, MSE={losses[-1]:.6f}')
    if self.track_loss:
      fig, ax = plt.subplots(figsize=(6,2))
      fig.suptitle(f'Loss function with learning_rate = {self.learning_rate}')
      ax.set_xlabel('num_iteration')
      ax.set_ylabel('score')
      plt.plot(losses)   

  def __predict(self, X):
      return np.matmul(X , self.w )


  def _predict(self, X):
    #remember to normalize the data before starting
      if self.normalize:
        X = self.normalizer.predict(X)
      return self.__predict(X)
    
      
  def _step(self, X, Y):
    # use w update for gradient descent
    N = X.shape[0]
    self.w = self.w -  (2*self.learning_rate/N)*(np.matmul(X.transpose(), self.__predict(X) - Y))


class RidgeLs(Ols):
  def __init__(self, ridge_lambda, *wargs, **kwargs):
    super(RidgeLs,self).__init__(*wargs, **kwargs)
    self.ridge_lambda = ridge_lambda
    
  def _fit(self, X, Y):
    #Closed form of ridge regression
    part1 = np.linalg.inv(np.matmul(X.T,X) + self.ridge_lambda * np.identity(X.shape[1]))
    self.w = np.matmul(np.matmul(part1,X.T),Y)