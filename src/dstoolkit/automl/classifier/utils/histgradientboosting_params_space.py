def get_histgradientboosting_params_space(trial, random_state=42):
    """
    Get the hyperparameter search space for the HistGradientBoosting classification model.

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
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'early_stopping': trial.suggest_categorical('early_stopping', [False]),
        'scoring': trial.suggest_categorical('scoring', ['roc_auc', 'neg_brier_score']),
        'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.4),
        'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
        'random_state': trial.suggest_categorical('random_state', [random_state])
    }