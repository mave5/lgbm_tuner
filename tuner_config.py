from skopt import space

# Define task
task = 'classification' # classification, regression

# optimization params
n_calls = 15
n_initial_points = 5


# lightgbm params
eval_metric = ['binary_logloss'] # rmse, auc, binary_logloss
metric_decreasing = True
log_period = -1
n_splits = 5
learning_rate = 0.1
n_estimators = 100
subsample_freq = 1
early_stopping_rounds = 10

# storing parameters
overwrite = True
path2hyp = 'auto_lgbm.json'

# storing convergence plot
path2plot = 'convergence_lgbm.png'


param_space = [
    [learning_rate],
    [n_estimators],
    [subsample_freq],
    
    space.Integer(5, 12, name="max_depth"),
    space.Integer(10, 1000, name="min_data_in_leaf"),
    space.Integer(21, 51, name="num_leaves"),
   
    space.Real(0.7, 1, prior="uniform", name="subsample"),
    space.Real(0.7, 1.0, prior="uniform", name="feature_fraction"),
    space.Integer(0.0, 100, prior="uniform", name="lambda_l1"),
    space.Integer(0.0, 100, prior="uniform", name="lambda_l2"),
    space.Real(0.0, 15, prior="uniform", name="min_gain_to_split"),
    space.Real(1e-5, 1e4, prior="log-uniform", name="min_child_weight"),
]    


param_names = [
               "learning_rate",
               "n_estimators", 
                "subsample_freq",
               "max_depth",
               'min_data_in_leaf',
               'num_leaves',
               "subsample",
               'feature_fraction',
               'lambda_l1',
               'lambda_l2',
               'min_gain_to_split',
               'min_child_weight',]