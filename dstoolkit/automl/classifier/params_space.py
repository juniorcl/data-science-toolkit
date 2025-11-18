def get_params_space(model_name, trial, random_state=42):
    match model_name:
        case 'LightGBM':
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
        case 'CatBoost':
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
        case 'HistGradientBoosting':
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
        case _:
            raise ValueError(f"Model '{model_name}' is not supported.")