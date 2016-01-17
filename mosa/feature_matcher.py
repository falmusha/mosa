import algorithm

class FeatureMatcher(algorithm.Algorithm):

    def __init__(self, algorithm_name, filter_ratio=0.6):
        super(FeatureMatcher, self).__init__(algorithm_name)
        self.filter_ratio = filter_ratio

    def filter_knn_matches(self, matches):
        filtered = []
        for m in matches:
            if len(m) == 2 \
                    and m[0].distance < m[1].distance * self.filter_ratio:
                filtered.append(m[0])
        return filtered

    def match(self, query_descriptions, train_descriptions):
        return self.algorithm.match(query_descriptions, train_descriptions)

    def knn_match(self, query_descriptions, train_descriptions, k=2):
        knn_matches = self.algorithm.knnMatch(
                query_descriptions,
                train_descriptions,
                k
            )
        return self.filter_knn_matches(knn_matches)
