import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from skopt import gp_minimize
from functools import partial
from logzero import logger
import lightgbm as lgb
import matplotlib.pylab as plt
from skopt.plots import plot_convergence
import sklearn
import skopt
import json
import os


class LGBMTuner():
    def __init__(self, configs):
        self.configs = configs
        
    def plot(self):
        plt.figure()
        plot_convergence(self.result)
        plt.tight_layout()
        plt.savefig(self.configs.path2plot)
        plt.close('all')

    def get_model(self):
        params = self.load_hyp()
        if self.configs.task == 'regression':
            estimator = LGBMRegressor(**params)
        elif self.configs.task == 'classification':
            estimator = LGBMClassifier(**params)
        else:
            logger.error(f"{self.configs.task} is not recognized!")
        return estimator
        
        
    def load_hyp(self):
        params = {}
        if os.path.exists(self.configs.path2hyp):
            with open(self.configs.path2hyp) as file:
                params = json.load(file)      
        return params
       
    def tune(self, X, y, groups=None, **kwargs):
        if os.path.exists(self.configs.path2hyp) and (not self.configs.overwrite):
            logger.info(f"parameters are loaded from {self.configs.path2hyp}")
            return self.load_hyp()
        
        logger.info("hyper-parameter tunning in progress ...")
        optimization_function = partial(self.train_cv,
                                        X=X, y=y, groups=groups, 
                                        **kwargs)
        self.result = gp_minimize(
                         optimization_function,
                         dimensions=self.configs.param_space,
                         n_calls=self.configs.n_calls,
                         n_initial_points=self.configs.n_initial_points,
                         verbose=False,
                         )        
        self.plot()

        best_params = dict(zip(self.configs.param_names, self.result.x))

        logger.info(f"parameters are stored in {self.configs.path2hyp}")
        with open(self.configs.path2hyp, 'w') as file:
            best_params_json = json.dumps(best_params, default=self.np_encoder, indent=4)
            file.write(best_params_json) 
        return best_params
    
    def train_cv(self, params_values, X, y, groups, **kwargs):
        params = dict(zip(self.configs.param_names, params_values))
        
        if np.any(groups):
            skf = sklearn.model_selection.GroupKFold(n_splits=self.configs.n_splits)
        else:
            skf = sklearn.model_selection.KFold(n_splits=self.configs.n_splits)
        
        metrics_val = []
        for train_idx, val_idx in skf.split(X, y, groups):
            x_train, x_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if self.configs.task == 'regression':
                estimator = LGBMRegressor(**params)
            elif self.configs.task == 'classification':
                estimator = LGBMClassifier(**params)
            else:
                logger.error(f"{self.configs.task} is not recognized!")

            fit_params = dict(callbacks = [lgb.log_evaluation(period=self.configs.log_period)],
                          eval_set = [(x_val, y_val)],
                          eval_names = ['val'],
                          eval_metric = self.configs.eval_metric,
                          early_stopping_rounds = self.configs.early_stopping_rounds,
                          feature_name = kwargs.get('feature_name', 'auto'),
                          categorical_feature= kwargs.get('categorical_feature', None),
                         )        
            estimator.fit(x_train, y_train, **fit_params)
            mm_val = estimator.best_score_['val'][self.configs.eval_metric[0]]
            metrics_val.append(mm_val)

        sign = 1 if self.configs.metric_decreasing == True else -1
        return np.mean(metrics_val) * sign
    
    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()    
        
