import numpy as np
from scipy.spatial.distance import cdist

class HausdorffDistance:
    def __init__(self):
        pass
    
    def directed_hausdorff(self, set_A, set_B):
        distances = cdist(set_A, set_B)
        min_distances = np.min(distances, axis=0)
        max_distance = np.max(min_distances)
        return max_distance
    
    def hausdorff_distance(self, set_A, set_B):
        h1 = self.directed_hausdorff(set_A, set_B)
        h2 = self.directed_hausdorff(set_B, set_A)
        return max(h1, h2)

set_A = np.array([[1, 2], [3, 4], [5, 6]])
set_B = np.array([[2, 3], [4, 5], [6, 7]])

class MeanVectorDistanceCalculator:
    def __init__(self, vectors1, vectors2):
        self.vectors1 = vectors1
        self.vectors2 = vectors2
        self.mean_vector1 = None
        self.mean_vector2 = None
        self.distance = None
    
    def calculate_mean(self):
        self.mean_vector1 = np.mean(self.vectors1, axis=0)
        self.mean_vector2 = np.mean(self.vectors2, axis=0)
    
    def calculate_distance(self):
        self.distance = np.linalg.norm(self.mean_vector1 - self.mean_vector2)
    
    def get_distance(self):
        return self.distance

class RandomVectorDistanceCalculator:
    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2
        self.selected_vector_set1 = None
        self.selected_vector_set2 = None
        self.euclidean_distance = None
        self.mean_distance = None
    
    def select_random_vectors(self):
        self.selected_vector_set1 = np.random.choice(self.set1)
        self.selected_vector_set2 = np.random.choice(self.set2)
    
    def calculate_euclidean_distance(self):
        self.euclidean_distance = np.linalg.norm(self.selected_vector_set1 - self.selected_vector_set2)

    def get_euclidean_distance(self):
        return self.euclidean_distance


# hausdorff = HausdorffDistance()
# distance = hausdorff.hausdorff_distance(set_A, set_B)


# print("Hausdorff distance:", distance)