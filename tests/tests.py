import unittest
from AccuracyMeasures import label_to_weight_mat, rocdist, align_labels, final_clusters, accuracy_measures
import numpy as np


class Tests(unittest.TestCase):
    """
    Test class for RTKM
    """

    def test_label_to_weight(self):
        """
        Test the creation of the assignment matrix from a list of labels.
        """
        #is num points not simply the len of label????
        #I don't think I ever even call this function???
        #Don't forget to test the else in this loop

        label = [1, 2, 1, 1, 2, 1]
        true_label_mat = [[1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0]]
        label_mat = label_to_weight_mat(label=label, num_points=6, num_clusters=2)

        np.testing.assert_array_equal(label_mat, true_label_mat)

    def test_roc_dist(self):
        """Test that outlier roc distance is being calculated properly."""
        #is n not the length of this outlier vector????
        true_outliers = np.array([1, 0, 0, 0, 0, 1])
        pred_outliers = np.array([1, 0, 1, 1, 1, 0])
        true_me = np.sqrt((1/4) ** 2 + (1/2) ** 2)

        assert rocdist(true_outliers=true_outliers, pred_outliers=pred_outliers, n=6) == true_me

    def test_align_labels(self):
        """
        Test that test and prediction labels are being aligned properly.

        First test exact labels, then text labels that maximize the f1 score.
        """
        true_clusters = np.array([[1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        pred_clusters = np.array([[0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 1]])

        assignment = align_labels(true_clusters=true_clusters, pred_clusters=pred_clusters)

        self.assertListEqual(list(assignment), [1, 2, 0])

        true_clusters = np.array([[1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        pred_clusters = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1]])
        assignment = align_labels(true_clusters=true_clusters, pred_clusters=pred_clusters)

        self.assertListEqual(list(assignment), [1, 2, 0])

    def test_final_pred_clusters(self):
        """Test correct assignment of final predicted clusters."""
        # Test single membership data with outleirs
        true_clusters = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1]])

        pred_clusters = np.array([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0]])
        pred_outliers = np.array([1, 1, 1, 1, 1, 1, 0])

        final_pred_clusters = final_clusters(true_clusters=true_clusters, pred_clusters=pred_clusters,
                                             membership_option='single', pred_outliers=pred_outliers)

        np.testing.assert_array_equal(final_pred_clusters, np.array([[1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0],
                                                                     [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]]))

        # Test single membership data without outliers
        true_clusters = np.array([[1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        pred_clusters = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1]])

        final_pred_clusters = final_clusters(true_clusters=true_clusters, pred_clusters=pred_clusters,
                                             membership_option='single', pred_outliers=None)

        np.testing.assert_array_equal(final_pred_clusters, np.array([[1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0],
                                                                     [0, 0, 0, 1, 0, 0]]))

        # Test multi membership data without outliers

        true_clusters = np.array([[1, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]])

        pred_clusters = np.array([[.50, .20, .80, 0, 0, .20], [.50, 0, 0, 0, 1, .80], [0, .80, .20, 1, 0, 0]])

        final_pred_clusters = final_clusters(true_clusters=true_clusters, pred_clusters=pred_clusters,
                                             membership_option='multi', pred_outliers=None)

        np.testing.assert_array_equal(final_pred_clusters, np.array([[1, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1],
                                                                     [0, 1, 1, 1, 0, 0]]))

        # Test multi membership data with outliers
        true_clusters = np.array([[1, 0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0],
                                  [0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]])

        pred_clusters = np.array([[.50, .20, .80, 0, 0, .20, 0], [.50, 0, 0, 0, 1, .80, 0],
                                  [0, .80, .20, 1, 0, 0, 0]])
        pred_outliers = np.array([1, 1, 1, 1, 1, 1, 0])

        final_pred_clusters = final_clusters(true_clusters=true_clusters, pred_clusters=pred_clusters,
                                             membership_option='multi', pred_outliers=pred_outliers)

        np.testing.assert_array_equal(final_pred_clusters, np.array([[1, 0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 1, 0],
                                                                     [0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]]))

    def test_accuracy_measures(self):
        """Test calculation of accuracy measures."""

        true_clusters = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1]])

        pred_clusters = np.array([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0]])
        pred_outliers = np.array([1, 1, 1, 1, 1, 1, 0])

        tp, fp, tn, fn, f1_scores, me = accuracy_measures(true_clusters=true_clusters, pred_clusters=pred_clusters,
                                       membership_option='single', pred_outliers=pred_outliers)

        np.testing.assert_array_equal(tp, [3, 1, 1, 1])
        np.testing.assert_array_equal(fp, [1, 0, 0, 0])
        np.testing.assert_array_equal(tn, [3, 5, 6, 6])
        np.testing.assert_array_equal(fn, [0, 1, 0, 0])
        self.assertEqual(me, 1.0)



