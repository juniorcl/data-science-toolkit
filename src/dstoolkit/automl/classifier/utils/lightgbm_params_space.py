def get_lightgbm_params_space(trial, random_state=42):
    """
    Get the hyperparameter search space for the LightGBM classification model.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object for suggesting hyperparameters.
    random_state : int, optional
        The random state for reproducibility (default is 42).

    Returns
    -------
    params_space : dict
        The hyperparameter search space for the model.
    """
    return {
        'objective': trial.suggest_categorical('objective', ['binary']),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
        'metric': trial.suggest_categorical('metric', ['auc', 'binary_logloss']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'random_state': trial.suggest_categorical('random_state', [random_state]),
        'verbose': trial.suggest_categorical('verbose', [-1]),
        'n_jobs': trial.suggest_categorical('n_jobs', [-1])
    }