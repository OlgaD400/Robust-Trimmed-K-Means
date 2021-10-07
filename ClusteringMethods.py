import numpy as np
from oct2py import octave
from KMOR import KMOR
from typing import Optional, Tuple
from scipy.cluster.vq import vq

def run_kmor(data: np.ndarray, k: int, percent_outliers: float, gamma: float, init_centers: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run KMOR and return weight assignment matrix and a vector identifying outliers.

    KMOR needs the data input to be size Nxm and initial centers to be of size kxm or mxk?????.

    Args:
        data (np.ndarray): Array containing data points. Array should be of size mxN, where m is the number of
            dimensions and N is the number of data points.
        k (int): Number of clusters.
        percent_outliers (float): Expected percentage of outliers in the dataset.
        gamma (float): KMOR parameter specifying...
        init_centers (np.ndarray): Optional array giving a set of initial centers for the KMOR algorithm.

    Returns:
        pred_labels (np.ndarray): Array containing weights determining extent of point membership to a cluster of shape
        kxN.
        pred_outliers (np.ndarray): 1D array containing 0 for points that are outliers and 1 for points that are not.
    """
    n = data.shape[1]

    if init_centers is None:
        labels, centers, pval = KMOR(data.T, k=k, gamma=gamma, n0=percent_outliers*n)
    else:
        labels, centers, pval = KMOR(data.T, k=k, gamma = gamma, n0=percent_outliers*n, Z=init_centers.T)

    KMOR_weights = np.zeros((k+1, n))
    KMOR_weights[labels.astype(int), np.arange(n)] = 1

    pred_labels = KMOR_weights[:-1]
    pred_outliers = KMOR_weights[-1:]
    pred_outliers = np.where(pred_outliers == 0, 1, 0)
    pred_outliers = np.squeeze(pred_outliers)

    return pred_labels, pred_outliers

def run_neo_parameters(data: np.ndarray, path_to_parameter_estimation: str,
                       init_centers: np.ndarray, alpha_delta: float, beta_delta:float) -> None:
    """
    Run the NEO k means parameter estimation script and print alpha and beta.

    Args:
        data (np.ndarray): Array containing data points. Array should be of size mxN, where m is the number of
            dimensions and N is the number of data points.
        path_to_parameter_estimation (str): Path to where file "neo_kmeans.m" is located.
        init_centers (Optional[np.ndarray]): Optional array giving a set of initial centers for the NEO k means algorithm.
        alpha_delta (float):
        beta_delta (float):
    """

    octave.addpath(path_to_parameter_estimation)
    octave.feval(path_to_parameter_estimation, data.T, init_centers.T, alpha_delta, beta_delta)

    return None

def run_neo(data: np.ndarray, path_to_NEO: str, k: int,
            percent_outliers: float, sigma: float, init_centers: np.ndarray = None)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the NEO k means algorithm and return weight assignment matrix and a vector identifying outliers.

    Requires input of percent_outliers and alpha.  "run_NEO_parameters" can be used to estimate these parameters
    if they are unknown.

    Args:
        data (np.ndarray): Array containing data points. Array should be of size mxN, where m is the number of
            dimensions and N is the number of data points.
        path_to_NEO (str): Path to where file "estimate_alpha_beta.m" is located.
        k (int): Number of clusters.
        percent_outliers (float): Expected percentage of outliers in the dataset.
        sigma (float): Parameter controlling extent of cluster overlap.  (n+sigma) point assignments are made. Sigma must be between 0 and k-1.
        init_centers (Optional[np.ndarray]): Optional array giving a set of initial centers for the NEO k means algorithm.
    """
    n = data.shape[1]
    assert 0 <= sigma <= k-1, "sigma must be between 0 and k-1"

    if init_centers is None:
        init_centers = data[:, np.random.choice(n, k)]

    init_assign, dist = vq(data.T, init_centers.T)
    init_U = np.zeros((k, n))
    init_U[init_assign, np.arange(n)] = 1

    octave.addpath(path_to_NEO)
    pred_clusters = octave.feval(path_to_NEO, data.T, k, sigma, percent_outliers, init_U.T)

    outliers_NEO = np.where(np.sum(pred_clusters, axis=1) == 0)[0]
    pred_outliers = np.ones(n)
    pred_outliers[outliers_NEO] = 0
    pred_outliers = pred_outliers.astype(int)

    return pred_clusters.T, pred_outliers
