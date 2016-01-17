import cv2

class AlgorithmFactory:

    @staticmethod
    def get(algorithm_type):
        try:
            return getattr(AlgorithmFactory, algorithm_type)()
        except AttributeError as e:
            return None

    @staticmethod
    def surf():
        return cv2.SURF()

    @staticmethod
    def sift():
        return cv2.SIFT()

    @staticmethod
    def fast():
        return cv2.FastFeatureDetector()

    @staticmethod
    def flann():
        FLANN_INDEX_KDTREE = 0
        index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
