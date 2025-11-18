def get_params_space(model_name, trial, random_state=42):
    match model_name:
        case 'KMeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                'n_init': trial.suggest_int('n_init', 5, 20),
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'algorithm': trial.suggest_categorical('algorithm', ['lloyd', 'elkan']),
                'random_state': trial.suggest_categorical('random_state', [random_state])
            }

        case 'MiniBatchKMeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
                'batch_size': trial.suggest_int('batch_size', 100, 2000, step=100),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'max_no_improvement': trial.suggest_int('max_no_improvement', 5, 50),
                'n_init': trial.suggest_int('n_init', 5, 20),
                'random_state': trial.suggest_categorical('random_state', [random_state]),
                'reassignment_ratio': trial.suggest_float('reassignment_ratio', 0.001, 0.05)
            }

        case 'Birch':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'threshold': trial.suggest_float('threshold', 0.1, 1.0),
                'branching_factor': trial.suggest_int('branching_factor', 10, 100),
                'compute_labels': trial.suggest_categorical('compute_labels', [True]),
                'copy': trial.suggest_categorical('copy', [True])
            }

        case 'GaussianMixture':
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

        case 'BayesianGaussianMixture':
            return {
                'n_components': trial.suggest_int('n_components', 2, 20),
                'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical']),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'reg_covar': trial.suggest_float('reg_covar', 1e-8, 1e-3, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
                'weight_concentration_prior_type': trial.suggest_categorical('weight_concentration_prior_type', ['dirichlet_process', 'dirichlet_distribution']),
                'init_params': trial.suggest_categorical('init_params', ['kmeans', 'random']),
                'mean_precision_prior': trial.suggest_float('mean_precision_prior', 0.1, 10.0, log=True),
                'random_state': trial.suggest_categorical('random_state', [random_state]),
                'warm_start': trial.suggest_categorical('warm_start', [False])
            }

        case _:
            raise ValueError(f"Model '{model_name}' is not supported.")