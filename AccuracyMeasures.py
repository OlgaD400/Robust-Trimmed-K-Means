import numpy as np
from collections import Counter
from sklearn.metrics import f1_score
from typing import Literal, Optional, Tuple, Union, List
from ClusteringMethods import run_neo, run_kmor
from RTKM import RTKM

def label_to_weight_mat(label, num_points, num_clusters):
    '''
    Create weight matrix from labels
    '''
    
    true_label_vec = np.zeros(num_points)
    
    new_val = 0
    for val in np.unique(label):
        true_label_vec = np.where(label == val, new_val, true_label_vec)
        new_val +=1
    
    if num_clusters == len(np.unique(label)):
        #no outliers
        true_label_mat = np.zeros((num_clusters, num_points))
    else:
        #last row corresponds to outliers
        true_label_mat = np.zeros((num_clusters+1, num_points))
    
    true_label_mat[true_label_vec.astype(int), np.arange(num_points)] = 1

    return true_label_mat


def ROCDist(true_outliers: np.ndarray, pred_outliers: np.ndarray, n: int) -> float:
    """
    Calculate the Me score.

    Args:
        true_outliers (np.ndarray): Array containing 0 for points that are outliers and 1 for points that are not.
        pred_outliers (np.ndarray):  Array containing 0 for points that are predicted to be outliers and 1 for points
        n (int): Number of total data points.
        that are not.

    Returns: me (float): Me score.
    """

    true_out = np.where(true_outliers == 1)[0]
    pred_out = np.where(pred_outliers == 0)[0]

    tp = len(np.intersect1d(true_out, pred_out))

    fp = len(pred_out) - tp

    fn = len(true_out) -tp

    tn = n - (tp + fp + fn)

    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)

    me = np.sqrt(fp_rate ** 2 + (1 - tp_rate) ** 2)

    return me


def align_labels(true_clusters: np.ndarray, pred_clusters: np.ndarray) -> np.ndarray:
    """Align the true and predicted point assignment weights for comparison.

    Return an array storing the indices that align the rows of the predicted clusters with those of the true clusters
    in order to maximizer f1 score.

    Args:
        true_clusters (np.ndarray): Matrix of true point assignment weights with size kxN if there are no outliers
        and k+1xN if there are outliers.
        pred_clusters (np.ndarray) Predicted point assignment weight matrix (should not have outliers attached).

    Returns:
        current_assignment (np.ndarray): Array storing the indices that align the rows of the predicted clusters with
        those of the true clusters in order to maximizer f1 score
    """
    k = pred_clusters.shape[0]
    percent_overlap = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            percent_overlap[i, j] = f1_score(pred_clusters[i, :], true_clusters[j, :])

    sorted_mat_i = np.argsort(percent_overlap, axis=1)
    sorted_mat_j = np.argsort(percent_overlap, axis=0)

    counts_dict = Counter(sorted_mat_i[:, -1])
    current_assignment = sorted_mat_i[:, -1]

    while not np.all(np.array(list(counts_dict.values())) == 1):
        for key in counts_dict.keys():
            if counts_dict[key] != 1:
                cols = np.where(current_assignment == key)[0]
                order = sorted_mat_j[:, key]

                max_ind = []
                for col_val in cols:
                    max_ind.append(np.where(order == col_val)[0][0])

                cols_to_change_ind = np.argsort(max_ind)[:-1]
                cols_to_change = np.array(cols)[cols_to_change_ind]

                for col_to_change in cols_to_change:
                    index_to_change = np.where(sorted_mat_i[col_to_change] == key)[0][0] - 1
                    current_assignment[col_to_change] = sorted_mat_i[col_to_change, index_to_change]

        counts_dict = Counter(current_assignment)

    return current_assignment


def final_clusters(true_clusters: np.ndarray, pred_clusters: np.ndarray,
                     membership_option: Literal['single', 'multi'], pred_outliers: Optional[np.ndarray] = None):
    """
    Calculate final predicted clusters for output of clustering that maximizes the f1 score of each cluster.

    Args:
        true_clusters (np.ndarray): Matrix of true point assignment weights with size kxN if there are no outliers
        and k+1xN if there are outliers.  Outlier assignments should be in the final row of true_clusters.
        pred_clusters (np.ndarray) Predicted point assignment weight matrix (should not have outliers attached).
        membership_option (Literal['single', 'multi']): String specifying 'single' or 'multi' depending on whether each
        point only belongs to one cluster or can belong to multiple clusters.
        pred_outliers (Optional[np.ndarray]): Array containing 0 for points that are predicted outliers and 1 for
        points that are not.

    """
    assert membership_option in ['single', 'multi'], "Not a valid membership option."

    if pred_outliers is None:
        k, n = true_clusters.shape
    else:
        k_plus, n = true_clusters.shape
        k = k_plus - 1

    if membership_option == 'single':
        max_ind = np.argmax(pred_clusters, axis = 0)
        pred_clusters = np.zeros((k,n))
        pred_clusters[max_ind, np.arange(n)] = 1
                
    elif membership_option == 'multi':
        pred_clusters = np.where(pred_clusters>0, 1, 0)

    current_assignment = align_labels(true_clusters, pred_clusters)

    if pred_outliers is not None:
        current_assignment = np.hstack((current_assignment, k))
        outliers = np.where(pred_outliers == 0)[0]
        pred_clusters[:, pred_outliers] = 0
        outlier_row = np.zeros(n)
        outlier_row[outliers] = 1
        pred_clusters = np.vstack((pred_clusters, outlier_row))

    final_pred_clusters = pred_clusters[np.argsort(current_assignment), :]

    return final_pred_clusters

def accuracy_measures(true_clusters: np.ndarray, pred_clusters: np.ndarray,
                      membership_option: Literal['single', 'multi'],
                      pred_outliers: Optional[np.ndarray] = None) ->Union[Tuple[List[int], List[int], List[int],
                                                                                List[int], List[int], float],
                                                                          Tuple[List[int],List[int], List[int],
                                                                                List[int], List[int]]]:
    """
    Return accuracy measures of clustering method.

    Args:
        true_clusters (np.ndarray): Matrix of true point assignment weights with size kxN if there are no outliers
        and k+1xN if there are outliers.
        pred_clusters (np.ndarray) Predicted point assignment weight matrix (should not have outliers attached).
        membership_option (Literal['single', 'multi']): String specifying 'single' or 'multi' depending on whether each
        point only belongs to one cluster or can belong to multiple clusters.
        pred_outliers (Optional[np.ndarray]): Array containing 0 for points that are predicted outliers and 1 for
        points that are not.

    Returns:
        tp (np.ndarray): Number of points assigned to each cluster that belong in the cluster.
        fp (np.ndarray): Number of points assigned to each cluster that do not belong in the cluster.
        tn (np.ndarray): Number of points not assigned to each cluster that do not belong in the cluster.
        fn (np.ndarray): Number of points not assigned to each cluster that belong in the cluster.
        f1_scores (np.ndarray): Array containing the F1 score for each cluster.
        me (float): Me score.
    """
    assert membership_option in ['single', 'multi'], "Not a valid membership option."

    tp = []
    fp = []
    tn = []
    fn = []
    me = None

    final_pred_clusters = final_clusters(true_clusters, pred_clusters, membership_option, pred_outliers)
    tot_rows, n = final_pred_clusters.shape
    
    for row in range(tot_rows):
        ind_true_member = np.where(true_clusters[row, :] == 1)[0]
        ind_true_nonmember = np.where(true_clusters[row, :] == 0)[0]
        ind_pred_member = np.where(final_pred_clusters[row, :] == 1)[0]
        ind_pred_nonmember = np.where(final_pred_clusters[row, :] == 0)[0]
        
        tp.append(len(np.intersect1d(ind_pred_member, ind_true_member)))
        fp.append(len(np.intersect1d(ind_pred_member, ind_true_nonmember)))
        tn.append(len(np.intersect1d(ind_pred_nonmember, ind_true_nonmember)))
        fn.append(len(np.intersect1d(ind_pred_nonmember, ind_true_member)))
        # cluster_size.append(np.count_nonzero(true_clusters[row, :]))

    f1_scores = f1_score(true_clusters.T, final_pred_clusters.T, average=None)

    if pred_outliers is not None:
        me = ROCDist(true_clusters[-1,:], pred_outliers, n)

    return tp, fp, tn, fn, f1_scores, me


def sensitivity_to_alpha(data: np.ndarray, k: int, kmor_gamma: float, neo_sigma: float, path_to_NEO: 'str',
                         alpha_vals: np.ndarray, iterations: int, true_labels: np.ndarray,
                         membership_option: Literal['single', 'multi']):
    """
    Run RTKM, KMOR, and NEO-k-means and test the sensitivity of the accuracy measures of all three methods to the
    value for the predicted number of outliers.
    """
    assert membership_option in ['single', 'multi'], "Not a valid membership option."


    n = data.shape[1]

    me_avg_rtkm = np.zeros(len(alpha_vals))
    me_avg_kmor = np.zeros(len(alpha_vals))
    me_avg_neo = np.zeros(len(alpha_vals))

    f1_avg_rtkm = np.zeros(len(alpha_vals))
    f1_avg_kmor = np.zeros(len(alpha_vals))
    f1_avg_neo = np.zeros(len(alpha_vals))

    min_rtkm = np.zeros((2, len(alpha_vals)))
    max_rtkm = np.zeros((2, len(alpha_vals)))

    min_kmor = np.zeros((2, len(alpha_vals)))
    max_kmor = np.zeros((2, len(alpha_vals)))

    min_neo = np.zeros((2, len(alpha_vals)))
    max_neo = np.zeros((2, len(alpha_vals)))

    for i, alpha_val in enumerate(alpha_vals):
        me_scores_rtkm = np.zeros(iterations)
        me_scores_kmor = np.zeros(iterations)
        me_scores_neo = np.zeros(iterations)

        f1_scores_rtkm = np.zeros(iterations)
        f1_scores_kmor = np.zeros(iterations)
        f1_scores_neo = np.zeros(iterations)

        for j in range(iterations):
            init_centers = data[:, np.random.choice(n,k)]

            RTKM_WBC = RTKM(data)
            RTKM_WBC.perform_clustering(k, percent_outliers=alpha_val, max_iter=500, init_centers=init_centers)
            tp, fp, tn, fn, f1_rtkm, me_rtkm = accuracy_measures(true_clusters=true_labels,
                                                                 pred_clusters=RTKM_WBC.weights,
                                                              pred_outliers=RTKM_WBC.outliers,
                                                              membership_option=membership_option)

            pred_labels_kmor, pred_outliers_kmor = run_kmor(data, k, percent_outliers=alpha_val, gamma=kmor_gamma,
                                                            init_centers=init_centers)
            tp, fp, tn, fn, f1_kmor, me_kmor = accuracy_measures(true_clusters=true_labels,
                                                                 pred_clusters=pred_labels_kmor,
                                                                 pred_outliers=pred_outliers_kmor,
                                                                 membership_option=membership_option)

            pred_labels_neo, pred_outliers_neo = run_neo(data, path_to_NEO, k, percent_outliers=alpha_val,
                                                         sigma = neo_sigma, init_centers=init_centers)
            tp, fp, tn, fn, f1_neo, me_neo = accuracy_measures(true_clusters=true_labels,
                                                               pred_clusters=pred_labels_neo,
                                                              pred_outliers=pred_outliers_neo,
                                                              membership_option=membership_option)

            f1_scores_rtkm[j] = np.average(f1_rtkm)
            f1_scores_kmor[j] = np.average(f1_kmor)
            f1_scores_neo[j] = np.average(f1_neo)

            me_scores_rtkm[j] = me_rtkm
            me_scores_kmor[j] = me_kmor
            me_scores_neo[j] = me_neo

        f1_avg_rtkm[i] = np.average(f1_scores_rtkm)
        f1_avg_kmor[i] = np.average(f1_scores_kmor)
        f1_avg_neo[i] = np.average(f1_scores_neo)

        me_avg_kmor[i] = np.average(me_scores_kmor)
        me_avg_rtkm[i] = np.average(me_scores_rtkm)
        me_avg_neo[i] = np.average(me_scores_neo)


        min_rtkm[0, i] = np.min(f1_scores_rtkm)
        max_rtkm[0, i] = np.max(f1_scores_rtkm)

        min_kmor[0, i] = np.min(f1_scores_kmor)
        max_kmor[0, i] = np.max(f1_scores_kmor)

        min_neo[0, i] = np.min(f1_scores_neo)
        max_neo[0, i] = np.max(f1_scores_neo)

        min_rtkm[1, i] = np.min(me_scores_rtkm)
        max_rtkm[1, i] = np.max(me_scores_rtkm)

        min_kmor[1, i] = np.min(me_scores_kmor)
        max_kmor[1, i] = np.max(me_scores_kmor)

        min_neo[1, i] = np.min(me_scores_neo)
        max_neo[1, i] = np.max(me_scores_neo)

    return f1_avg_rtkm, f1_avg_kmor, f1_avg_neo, me_avg_rtkm, me_avg_kmor, me_avg_neo,\
           min_rtkm, max_rtkm, min_kmor, max_kmor, min_neo, max_neo





