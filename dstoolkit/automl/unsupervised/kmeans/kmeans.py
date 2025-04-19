import optuna
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from ..analysis import (
    get_umap_params,
    get_pca_params,
    get_results
)

from ..metrics import get_cluster_metrics


MODEL_NAME = "KMeans"


def get_params(trial, random_state=42):
    
    return {
        'n_clusters': trial.suggest_int('n_clusters', 2, 10),
        'n_init': trial.suggest_int('n_init', 5, 20),
        'random_state': trial.suggest_categorical('random_state', [random_state])
    }


class AutoMLKMeans:

    def __init__(self, X, n_trials=100, random_state=42):
        
        self.X = X
        self.n_trials = n_trials
        self.y = pd.DataFrame()
        self.random_state = random_state

    def _get_best_params(self) -> dict:

        def objective(trail):

            params = get_params(trail, self.random_state)

            model = KMeans(**params)

            labels = model.fit_predict(self.X)

            return silhouette_score(self.X, labels)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def _get_best_pca_model_params(self) -> dict:

        def objective(trail):

            params = get_params(trail, self.random_state)

            pca_params = get_pca_params(trail, self.X.shape[1], self.random_state)

            reducer = PCA(**pca_params)
            X_reduced = reducer.fit_transform(self.X)

            kmeans = KMeans(**params)

            labels = kmeans.fit_predict(X_reduced)

            return silhouette_score(X_reduced, labels)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def _get_best_umap_model_params(self) -> dict:

        def objective(trail):

            params = get_params(trail, self.random_state)

            umap_params = get_umap_params(trail)

            reducer = UMAP(**umap_params)
            X_reduced = reducer.fit_transform(self.X)

            kmeans = KMeans(**params)

            labels = kmeans.fit_predict(X_reduced)

            return silhouette_score(X_reduced, labels)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_base_model(self) -> tuple:

        best_params = self._get_best_params()
        
        self.base_model = KMeans(**best_params)

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
        
        self.pca_model = KMeans(
            n_clusters=best_params['n_clusters'], 
            n_init=best_params['n_init'], 
            random_state=best_params['random_state']
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
        
        self.umap_model = KMeans(
            n_clusters=best_params['n_clusters'], 
            n_init=best_params['n_init'], 
            random_state=best_params['random_state']
        )
        
        self.y['umap_model_labels'] = self.umap_model.fit_predict(self.X_umap)

        return get_cluster_metrics(self.X_umap, self.y['umap_model_labels'])

    def train(self) -> None:

        self.result_train_base_model = self._train_base_model()
        self.result_train_pca_model = self._train_pca_model()
        self.result_train_umap_model = self._train_umap_model()

    def get_metrics(self) -> dict:

        return pd.DataFrame(
            {
                f"Base {MODEL_NAME} Model": self.result_train_base_model,
                f"PCA {MODEL_NAME} Model": self.result_train_pca_model,
                f"UMAP {MODEL_NAME} Model": self.result_train_umap_model
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


