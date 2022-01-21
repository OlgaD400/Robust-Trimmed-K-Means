"""Perform Robust Trimmed K Means to cluster data."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from proxlib.operators import proj_csimplex


def SimplexProx(z: np.ndarray, a: float) -> np.ndarray:
    """
    Project onto simplex.

    Args:
        z:  Variable to be projected. Assume ``x`` is a matrix, each row will be projected onto a simplex.
        a:  Simplex to be projected onto.
    Returns:
        np.maximum(z + lam, 0.0) (np.ndarray): Projected variable.
    """
    u = z.copy()

    if u.ndim == 1:
        u = u[np.newaxis, :]

    u[:, ::-1].sort(axis=1)

    j = np.arange(u.shape[1])
    v = (a - np.cumsum(u, axis=1)) / (j + 1)

    i = np.repeat(j[None, :], u.shape[0], axis=0)
    rho = np.max(i * (u + v > 0), axis=1)
    lam = v[np.arange(u.shape[0]), rho][:, None]
    return np.maximum(z + lam, 0.0)


class RTKM:
    """Create a class to perform Robust Trimmed k Means clustering."""

    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize RTKM.

        Args:
            data (np.ndarray): Array containing data points. Array should be of size mxN, where m is the number of
            dimensions and N is the number of data points.

        Attributes:
            self.centers (np.ndarray): Array containing calculated cluster centers.
            self.weights (np.ndarray): Array containing weights determining extent of point membership to a cluster.
            self.outliers (np.ndarray): 1D Array containing 0 for points that are outliers and 1 for points that are not.
            self.obj_hist: tbd
            self.err_hist: tbd
        """
        self.data = data
        self.centers = None
        self.weights = None
        self.outliers = None
        self.obj_hist = None
        self.err_hist = None

    def perform_clustering(self, k: int, tol: float = 1e-6, max_iter: int = 100,
                           init_centers: Optional[np.ndarray] = None, num_members: int = 1.0) -> None:
        """
        Perform Robust Trimmed k Means algorithm and set values for RTKM attributes.

        Sets values for predicted cluster centers, cluster membership weights, outliers, objective function values over
        iterations, and error values over iterations.

        Args:
            k (int): Number of clusters.
            percent_outliers (float): Expected percentage of outliers in the dataset.
            tol (float): Error tolerance for convergence of algorithm.
            max_iter (int): Max number of iterations for algorithm.
            init_centers (np.ndarray): Initial centers for algorithm.
            num_members (int): Maximum number of clusters a point can belong to.
        """
        n = self.data.shape[1]

        if init_centers is None:
            centers = self.data[:, np.random.choice(n, k), :]
        else:
            centers = init_centers

        lam = .5
        dk = 1.1
        ek = 1.1

        iter_count = 0
        err = tol + 1.0

        obj_hist = []
        err_hist = []

        centers_old = centers.copy()

        weights = np.zeros((k, n))

        outliers = np.ones(n)

        while err >= tol:
            prod1 = (weights).T
            #@ for each time slice...
            centers_new = (self.data @ prod1) + centers_old / (np.sum(prod1, axis=0) + n)

            data_norm = np.linalg.norm(self.data, 2, axis=0) ** 2
            centers_norm = np.linalg.norm(centers_new, 2, axis=0) ** 2

            if num_members == 1:
                weights_new = SimplexProx(weights.T - 1 / dk * outliers[:, np.newaxis] * (
                        data_norm[:, np.newaxis] - 2 * self.data.T @ centers_new + centers_norm[np.newaxis, :]),
                                          num_members)
            else:
                weights_new = weights.T - 1 / dk * outliers[:, np.newaxis] * (
                        data_norm[:, np.newaxis] - 2 * self.data.T @ centers_new + centers_norm[np.newaxis, :])
                proj_csimplex(weights_new, num_members, 0.0, 1.0)

            weights_new = weights_new.T
            outliers_new = (outliers - 1 / ek * np.linalg.norm(self.data - centers_new @ weights_new, 2, axis=0) ** 2)[
                           np.newaxis, :]
            proj_csimplex(outliers_new, h, 0.0, 1.0)

            centers_err = np.linalg.norm(centers - centers_new)
            weights_err = np.linalg.norm(weights - weights_new)
            outliers_err = np.linalg.norm(outliers - outliers_new)

            np.copyto(centers_old, centers)
            np.copyto(centers, centers_new)
            np.copyto(weights, weights_new)
            np.copyto(outliers, outliers_new)

            err = weights_err * dk + outliers_err * ek + centers_err

            obj = np.sum(outliers * np.linalg.norm(self.data - centers @ weights, 2, axis=0) ** 2)

            obj_hist.append(obj)
            err_hist.append(err)

            iter_count += 1

            if iter_count % 100 == 0:
                print('Iteration', iter_count)

            if iter_count >= max_iter:
                print('PALM reached maximum number of iterations')
                self.centers = centers
                self.weights = weights
                self.outliers = outliers.astype(int)
                self.obj_hist = obj_hist
                self.err_hist = err_hist
                break

        self.centers = centers
        self.weights = weights
        self.outliers = outliers.astype(int)
        self.obj_hist = obj_hist
        self.err_hist = err_hist

    def return_clusters(self):
        """Returns predicted cluster labels and outliers from RTKM.  Only used for single-cluster membership."""
        pred_clusters = np.argmax(self.weights, axis=0)

        pred_outliers = np.where(self.outliers == 0)[0]

        pred_clusters[pred_outliers] = self.weights.shape[0]

        return pred_clusters, pred_outliers
