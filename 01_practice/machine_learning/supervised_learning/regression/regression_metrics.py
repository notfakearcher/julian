# import libraries
import pandas as pd
import numpy as np

# master function to get all regression metrics
def jra_regression_metrics(y_true, y_pred, p):
    
  # bias - inability of model causing error between predicted and actual values [-inf, inf]
  def jra_ml_bias(y_true, y_pred):
    x = (np.mean(y_pred) - y_true).sum()
    return(x)
  
  # r squared - goodness of fit; amount of variability explained [0, 1]
  def jra_ml_rsquared(y_true, y_pred):
    a = np.power((y_pred - y_true), 2).sum()
    b = np.power((np.mean(y_true) - y_true), 2).sum()
    x = 1 - (a / b)
    return(x)
  
  # adjusted r squared - goodness of fit; amount of variability explained while penalizing for excess variabels [0, 1]
  def jra_ml_arsquared(y_true, y_pred, p):
    a = np.power((y_pred - y_true), 2).sum()
    b = np.power((np.mean(y_true) - y_true), 2).sum()
    r2 = 1 - (a / b)
    n = len(y_true)
    c = (1 - r2) * (n - 1)
    d = n - p - 1
    x =  1 - (c / d)
    return(x)
  
  # mean squared error - goodness of fit; how close predicted matches true values; gives more weight to larger errors [0, Inf] [0, Inf]
  def jra_ml_mse(y_true, y_pred):
    a = y_pred - y_true
    x = np.power(a, 2).mean()
    return(x)
  
  # root mean squared error - like mse but smaller values and units match original [0, Inf]
  def jra_ml_rmse(y_true, y_pred):
    a = y_pred - y_true
    b = np.power(a, 2).mean()
    x = np.sqrt(b)
    return(x)
  
  # mean squared log error - like mse but offsets large deviations between predicted values and true values [0, Inf]
  def jra_ml_msle(y_true, y_pred):
    a = np.log((y_pred + 1)) - np.log((y_true + 1))
    x = np.power(a, 2).mean()
    return(x)
  
  # root mean squared log error - like rmse but used when predicted values largely deviate true values [0, Inf]
  def jra_ml_rmsle(y_true, y_pred):
    a = np.log((y_pred + 1)) - np.log((y_true + 1))
    b = np.power(a, 2).mean()
    x = np.sqrt(b)
    return(x)
  
  # relative root mean squared error - relative/ratio error of model compared to naive/average model [0, Inf]
  def jra_ml_rrmse(y_true, y_pred):
    a = y_pred - y_true
    b = np.power(a, 2).mean()
    c = np.sqrt(b)
    d = y_pred - np.mean(y_true)
    e = np.power(d, 2).mean()
    f = np.sqrt(e)
    x = c / f
    return(x)
  
  # normalized mean squared error - relative/ratio error of model normalized for cross-model comparisons [0, Inf]
  def jra_ml_nmse(y_true, y_pred):
    a = y_pred - y_true
    b = np.power(a, 2).mean()
    x = b / np.var(y_true)
    return(x)
  
  # normalized root mean squared error - like nmse but smaller values and units match original [0, Inf] [0, Inf]
  def jra_ml_nrmse(y_true, y_pred):
    a = y_pred - y_true
    b = np.power(a, 2).mean()
    x = np.sqrt(b) / np.std(y_true)
    return(x)
  
  # mean absolute error - goodness of fit; how close predicted matches true values not taking direcion into account; gives equal weight to errors [0, Inf]
  def jra_ml_mae(y_true, y_pred):
    a = np.abs(y_pred - y_true)
    x = a.mean()
    return(x)
  
  # mean absolute percent error - goodness of fit; relative magnitue of error produced on average [0, Inf]
  def jra_ml_mape(y_true, y_pred):
    a = np.abs(y_pred - y_true)
    b = a / np.abs(y_true) 
    x = b.mean() * 100
    return(x)
  
  # symmetric mean absolute percent error - goodness of fit; normalized relative magnitue of error produced on average [0, 200]
  def jra_ml_smape(y_true, y_pred):
    a = np.abs(y_pred - y_true) * 2
    b = np.abs(y_pred) + np.abs(y_true)
    x = (a / b).mean() * 100
    return(x)
  
  # mean relative absolute error - relative/ratio absolute error of model compared to naive/average model [0, Inf]
  def jra_ml_mrae(y_true, y_pred):
    a = np.abs(y_pred - y_true)
    b = y_true - np.mean(y_true)
    c = np.abs(b)
    x = (a.mean() / c.mean())
    return(x)
  
  # median absolute error - goodness of fit; median differences between predicted and true values not taking direcion into account; gives equal weight to errors [0, Inf]
  def jra_ml_mdae(y_true, y_pred):
    a = np.abs(y_pred - y_true)
    x = a.median()
    return(x)
  
  # median relative absolute error - median relative/ratio absolute error of model compared to naive/average model [0, Inf]
  def jra_ml_mdrae(y_true, y_pred):
    a = np.abs(y_pred - y_true)
    b = y_true - np.mean(y_true)
    c = np.abs(b)
    x = (a.median() / c.median())
    return(x)
  
  df_metrics = pd.DataFrame({
      'bias': jra_ml_bias(y_true, y_pred),
      'rsquared': jra_ml_rsquared(y_true, y_pred),
      'arsquared': jra_ml_arsquared(y_true, y_pred, p),
      'mse': jra_ml_mse(y_true, y_pred),
      'rmse': jra_ml_rmse(y_true, y_pred),
      'rmsle': jra_ml_rmsle(y_true, y_pred),
      'rrmse': jra_ml_rrmse(y_true, y_pred),
      'nrmse': jra_ml_nrmse(y_true, y_pred), 
      'mae': jra_ml_mae(y_true, y_pred),
      'mape': jra_ml_mape(y_true, y_pred),
      'smape': jra_ml_smape(y_true, y_pred),
      'mrae': jra_ml_mrae(y_true, y_pred),
      'mdae': jra_ml_mdae(y_true, y_pred),
      'mdrae': jra_ml_mdrae(y_true, y_pred),
    }, index = ['Metrics'])
  df_metrics.columns = [x.upper() for x in df_metrics.columns]
  df_metrics = df_metrics.apply(lambda x: round(x, 4)).T
  
  return(df_metrics)