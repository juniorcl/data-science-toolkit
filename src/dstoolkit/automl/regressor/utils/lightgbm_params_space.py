def get_lightgbm_params_space(trial, random_state=42):
    return {
        'objective': trial.suggest_categorical('objective', ['regression']),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
        'metric': trial.suggest_categorical('metric', ['rmse', 'mae', 'mape', 'mse']),
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
        'verbose': trial.suggest_categorical('verbose', [-1])
    }