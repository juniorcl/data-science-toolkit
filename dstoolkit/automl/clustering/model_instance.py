from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def get_model_instance(model_name):
    match model_name:
        case 'KMeans':
            return KMeans
        case 'MiniBatchKMeans':
            return MiniBatchKMeans
        case 'Birch':
            return Birch
        case 'GaussianMixture':
            return GaussianMixture
        case 'BayesianGaussianMixture':
            return BayesianGaussianMixture
        case _:
            raise ValueError(f"Model '{model_name}' is not supported.")