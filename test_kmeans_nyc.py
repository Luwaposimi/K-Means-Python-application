import unittest
from kmeans_nyc_taxi import X_scaled, optimal_k, kmeans  # ensure script is run or imported properly

#Unit test
class TestKMeansTaxi(unittest.TestCase):

    def test_data_not_empty(self):
        self.assertGreater(X_scaled.shape[0], 1000)

    def test_optimal_k_positive(self):
        self.assertGreater(optimal_k, 1)

    def test_cluster_labels_correct_length(self):
        self.assertEqual(len(kmeans.labels_), X_scaled.shape[0])

    def test_silhouette_score_reasonable(self):
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_scaled, kmeans.labels_)
        self.assertGreater(score, 0.2)  # reasonable clustering


if __name__ == "__main__":
    unittest.main()

