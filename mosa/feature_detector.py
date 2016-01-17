import algorithm

class FeatureDetector(algorithm.Algorithm):

    def __init__(self, algorithm_name):
        super(FeatureDetector, self).__init__(algorithm_name)

    def detect(self, img):
        return self.algorithm.detect(img, None)
