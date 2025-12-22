def get_catboost_params_space(trial, random_state=42):
    """
    Get the hyperparameter search space for the CatBoost classification model.

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
        'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'CrossEntropy']),
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Plain']),
        'random_seed': trial.suggest_categorical('random_seed', [random_state]),
        'verbose': trial.suggest_categorical('verbose', [0])
    }