def get_kmeans_params_space(trial, random_state=42):
    """
    Get the hyperparameter search space for the specified clustering model.

    Parameters
    ----------
    model_name : str
        The name of the clustering model to use.
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
        'n_clusters': trial.suggest_int('n_clusters', 2, 20),
        'init': trial.suggest_categorical('init', ['k-means++', 'random']),
        'n_init': trial.suggest_int('n_init', 5, 20),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'algorithm': trial.suggest_categorical('algorithm', ['lloyd', 'elkan']),
        'random_state': trial.suggest_categorical('random_state', [random_state])
    }