from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from scipy import optimize
from scipy.special import expit
import numpy as np
import math


from few_shot.utils import pairwise_distances

def Sigmoid(x):
  return expit(x)


class NNO():

  def __init__(self, **model_hyper_parameters):
    super().__init__()

    self.tau = None
    #self.m = len(prototype)
    self.threshold = 0.5


  def fit(self, X, Y=None):
    """
    Fit global model on X features to minimize
     a given function on Y

    @param X
    @param Y
    """

    return self

  def predict(self, X):
    """
    @param X: feature vector the model will be evaluated on
    """
    y_pred = self._model(X, self.prototype, self.tau)
    return np.where(
        y_pred == 0,
        1,
        0
    )

  def _model(self, embeddings, prototype, tau):
    #m_dim_ball_volume = ( math.gamma( (self.m/2)+1 ) ) / (  (np.power(np.pi,(self.m/2)))*(np.power(tau,self.m)) )
    m_dim_ball_volume = 1#(  (np.power(np.pi,(self.m/2)))*(np.power(tau,self.m)) ) / ( math.gamma( (self.m/2)+1 ) )
    #print("Volume: ", m_dim_ball_volume)
    f = m_dim_ball_volume * tau * (1 - (1/tau)*(np.sqrt(np.power(embeddings-prototype, 2).sum(axis=1))) )
    f = np.maximum(0,f)
    #print("Pre Sigmoid", f)
    #f = Sigmoid(f)
    return f

  def _loss(self, y_obs, y_pred):
    '''
    return np.where(
        y_obs == 0 ,
        -y_pred,
        y_obs-y_pred
    ).sum()
    '''
    #return (y_obs-y_pred).sum()
    #return log_loss(y_obs, y_pred)
    y_pred_binary = np.where(
        y_pred == 0,
        1,
        0
    )
    #print('Y_pred_binary: ', y_pred_binary)
    return np.where(
      y_obs == y_pred_binary,
      0,
      1
    ).sum()

  def _f(self, tau, *args):

    embeddings = self._train_data
    y_obs = self._train_target
    y_pred = self._model(embeddings, self.prototype, tau)
    l = self._loss(y_obs, y_pred)
    #print("Tau: ", tau)
    #print("Y:obs", y_obs)
    #print("Y_pred_class: ", np.where(y_pred > 0,0,1))
    #print("y_pred: ", y_pred)
    #print("Loss: ", l)
    #print('---------------------------------------')
    return l

  def fit(self, X, Y, prototype):
    #print('NNO SUPPORT:', X)
    self.prototype = prototype
    self.m = len(prototype)
    self._train_data = X
    self._train_target = Y
    x_dist = np.sqrt(np.power(X-prototype, 2).sum(axis=1))- 0.001
    #print("NNO DISTANCES: ", x_dist)
    #print("NNO PROTOTYPE: ", self.prototype)
    #x_dist = np.power(X-prototype, 2).sum(axis=1)+ 0.001
    #x_dist = pairwise_distances(X, prototype, 'l2')

    self.tau = x_dist[0]
    lowest_loss = self._f(x_dist[0])
    for i in range(len(X)):
      current_tau = x_dist[i]
      loss = self._f(current_tau)
      if loss < lowest_loss:
        lowest_loss = loss
        self.tau = current_tau


    '''
    tau_initial_value = np.array([50])
    res = optimize.minimize(
        self._f,
        x0=tau_initial_value,
        method='Nelder-Mead'
    )
    print("Res success: ", res.success)
    if res.success:
      print("Res.x: ", res.x)
      self.tau = res.x
      '''

    return self
