# LightGBM Tuner
Simple automatic hyper-parameter optimization for LightGBM using Bayesian optimization (skopt) 


## Example


Genereting some sample data
```python
n_rows, n_cols, n_grps = 1000, 10, 5

x_train = np.random.rand(n_rows, n_cols)
x_test = np.random.rand(n_rows, n_cols)

y_train = np.random.randint(2, size=n_rows)
y_test = np.random.randint(2, size=n_rows)

groups_train = np.random.randint(n_grps, size=n_rows)
groups_test = np.random.randint(n_grps, size=n_rows)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

```

Check configurations in ```config_tuner``` 



Automatic tunning
```python
import warnings
import tuner_config
from auto_lgbm import LGBMTuner

warnings.filterwarnings("ignore")

lgbm_tr = LGBMTuner(configs=tuner_config)
params_opt = lgbm_tr.tune(X=x_train,
                        y=y_train,
                        groups=groups_train,
                        categorical_feature='auto',
                        feature_name=None)
                        
```

Create and fit an object of LightGBM with the optimized parameters
```python
estmtr_opt = LGBMClassifier(**params_opt)
estmtr_opt.get_params()
```

```python
fit_params = dict(callbacks = [lgb.log_evaluation(period=50)],
              eval_set = [(x_train, y_train),(x_test, y_test)],
              eval_names = ['train','val'],
              eval_metric = ['binary_logloss', 'auc'],
              early_stopping_rounds = 10,
              feature_name = 'auto',)

estmtr_opt.fit(x_train, y_train, **fit_params)
```

For comparison, create and fit an object of LightGBM with the default parameters
```python
estmtr_default = LGBMClassifier()
estmtr_default.get_params()

estmtr_default.fit(x_train, y_train, **fit_params)
estmtr_default.best_score_
```





