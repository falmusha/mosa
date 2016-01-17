import algorithm

class FeatureDescriptor(algorithm.Algorithm):

    def __init__(self, algorithm_name):
        super(FeatureDescriptor, self).__init__(algorithm_name)

    def compute(self, img, features):
        return self.algorithm.compute(img, features)
