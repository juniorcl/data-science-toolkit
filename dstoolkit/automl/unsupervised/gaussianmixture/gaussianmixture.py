import optuna
import pandas as pd
from umap import UMAP
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from ..analysis import (
    get_umap_params,
    get_pca_params,
    get_results
)

from ..metrics import get_cluster_metrics


MODEL_NAME = "GaussianMixture"


def get_params(trial, n_features):
    
    return {
        'n_components': trial.suggest_int("n_components", 2, n_features),
        'covariance_type': trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"]),
        'n_init': trial.suggest_int("n_init", 1, 10),
        'tol': trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        'reg_covar': trial.suggest_float("reg_covar", 1e-6, 1e-2, log=True)
    }


class AutoMLGaussianMixture:

    def __init__(self, X, n_trials=100, random_state=42):
        
        self.X = X
        self.n_trials = n_trials
        self.y = pd.DataFrame()
        self.random_state = random_state

    def _get_best_params(self) -> dict:

        def objective(trail):

            params = get_params(trail, self.X.shape[1])

            model = GaussianMixture(**params)

            labels = model.fit_predict(self.X)

            return silhouette_score(self.X, labels)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def _get_best_pca_model_params(self) -> dict:

        def objective(trail):

            pca_params = get_pca_params(trail, self.X.shape[1], self.random_state)

            reducer = PCA(**pca_params)
            X_reduced = reducer.fit_transform(self.X)

            params = get_params(trail, X_reduced.shape[1])

            model = GaussianMixture(**params)

            labels = model.fit_predict(X_reduced)

            return silhouette_score(X_reduced, labels)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def _get_best_umap_model_params(self) -> dict:

        def objective(trail):

            umap_params = get_umap_params(trail)

            reducer = UMAP(**umap_params)
            X_reduced = pd.DataFrame(reducer.fit_transform(self.X))

            params = get_params(trail, X_reduced.shape[1])

            model = GaussianMixture(**params)

            labels = model.fit_predict(X_reduced)

            return silhouette_score(X_reduced, labels)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_base_model(self) -> tuple:

        best_params = self._get_best_params()
        
        self.base_model = GaussianMixture(**best_params)

        self.y['base_model_labels'] = self.base_model.fit_predict(self.X)
    
        return get_cluster_metrics(self.X, self.y['base_model_labels'])
    
    def _train_pca_model(self) -> tuple:

        best_params = self._get_best_pca_model_params()

        self.pca = PCA(
            n_components=best_params['n_components'], 
            random_state=best_params['random_state']
        )
        
        X_reduced = self.pca.fit_transform(self.X)

        self.X_pca = pd.DataFrame(X_reduced, columns=[f"pca_{i}" for i in range(X_reduced.shape[1])])
        
        self.pca_model = GaussianMixture(
            n_components=best_params['n_components'],
            covariance_type=best_params['covariance_type'],
            n_init=best_params['n_init'],
            tol=best_params['tol'],
            reg_covar=best_params['reg_covar']
        )   
        
        self.y['pca_model_labels'] = self.pca_model.fit_predict(self.X_pca)

        return get_cluster_metrics(self.X_pca, self.y['pca_model_labels'])
    
    def _train_umap_model(self) -> tuple:

        best_params = self._get_best_umap_model_params()

        self.umap = UMAP(
            n_components=best_params['n_components'],  
            n_neighbors=best_params['n_neighbors'],  
            min_dist=best_params['min_dist']  
        )
        
        X_reduced = self.umap.fit_transform(self.X)

        self.X_umap = pd.DataFrame(X_reduced, columns=[f"umap_{i}" for i in range(X_reduced.shape[1])])
        
        self.umap_model = GaussianMixture(
            n_components=best_params['n_components'],
            covariance_type=best_params['covariance_type'],
            n_init=best_params['n_init'],
            tol=best_params['tol'],
            reg_covar=best_params['reg_covar']
        )
        
        self.y['umap_model_labels'] = self.umap_model.fit_predict(self.X_umap)

        return get_cluster_metrics(self.X_umap, self.y['umap_model_labels'])

    def train(self) -> None:

        self.result_train_base_model = self._train_base_model()
        self.result_pca_model = self._train_pca_model()
        self.result_umap_model = self._train_umap_model()

    def get_metrics(self) -> dict:

        return pd.DataFrame(
            {
                f"Base {MODEL_NAME} Model": self.result_train_base_model,
                f"PCA {MODEL_NAME} Model": self.result_pca_model,
                f"UMAP {MODEL_NAME} Model": self.result_umap_model
            }
        )
    
    def get_result_analysis(self) -> None:

        results = self.get_metrics()

        print(f"Base {MODEL_NAME} Model")
        display(results[f"Base {MODEL_NAME} Model"])
        get_results(self.X, self.y, "base_model_labels")

        print(f"PCA {MODEL_NAME} Model")
        display(results[f"PCA {MODEL_NAME} Model"])
        get_results(self.X_pca, self.y, "pca_model_labels")
        
        print(f"UMAP {MODEL_NAME} Model")
        display(results[f"UMAP {MODEL_NAME} Model"])
        get_results(self.X_umap, self.y, "umap_model_labels")


