def get_gaussian_mixture_params_space(trial, random_state):
    return {
        'n_components': trial.suggest_int('n_components', 2, 20),
        'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical']),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'reg_covar': trial.suggest_float('reg_covar', 1e-8, 1e-3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'n_init': trial.suggest_int('n_init', 1, 10),
        'init_params': trial.suggest_categorical('init_params', ['kmeans', 'random']),
        'random_state': trial.suggest_categorical('random_state', [random_state]),
        'warm_start': trial.suggest_categorical('warm_start', [False])
    }