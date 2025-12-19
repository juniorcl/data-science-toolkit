from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def get_model_instance(model_name):
    """
    Retrieve a clustering model instance based on the provided model name.

    Parameters
    ----------
    model_name : str
        The name of the clustering model to use.

    Returns
    -------
    model : object
        The clustering model instance.

    Raises
    ------
    ValueError
        If the input data is not valid for clustering.
    """
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