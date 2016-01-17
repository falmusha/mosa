import os
import sys
import yaml
import timeit
import numpy as np

from csv_writer import *
from mosa.feature_detector import *
from mosa.feature_descriptor import *
from mosa.feature_matcher import *
from mosa.algorithm_factory import *

class Benchmark:

    def __init__(self, detectors, descriptors, matchers, csv_file):
        self.detectors = detectors
        self.descriptors = descriptors
        self.matchers = matchers
        self.csv_writer = CSVWriter(matchers, csv_file)

    def find_homography(self, query_descriptors, train_descriptors):
        """
        Find the homogrpahy matrix between the two sets of image points and
        return a tuple of the matrix and the outlier founds
        """
        H, mask = cv2.findHomography(
                query_descriptors,
                train_descriptors,
                cv2.RANSAC,
                5.0
            )

        outlier_indices = []
        for i, m in enumerate(mask):
            if m[0] == 0:
                outlier_indices.append(i)

        return (H, outlier_indices)

    def instrument(self, image1_path, image2_path):
        self.csv_writer.write_header()

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        for detector in self.detectors:
            for descriptor in self.descriptors:
                self.experiment(
                        image1,
                        image2,
                        detector,
                        descriptor,
                        self.matchers
                    )

    def detect_and_compute(self, image, detector, descriptor):
        start_time = timeit.default_timer()
        features = detector.detect(image)
        elapsed = timeit.default_timer() - start_time

        self.csv_writer.write_column(len(features))
        self.csv_writer.write_column(elapsed)

        start_time = timeit.default_timer()
        descriptions = descriptor.compute(image, features)
        elapsed = timeit.default_timer() - start_time

        self.csv_writer.write_column(len(descriptions))
        self.csv_writer.write_column(elapsed)

        return descriptions

    def outliers(self, query_features, train_features, matches):

        matched_query_features = \
                np.float32([ query_features[m.queryIdx].pt for m in matches ])

        matched_train_features = \
                np.float32([ train_features[m.trainIdx].pt for m in matches ])

        (homography, outliers) = self.find_homography(
                matched_query_features,
                matched_train_features
            )

        return outliers

    def match(self, query_descriptions, train_descriptions, matcher, min_match=4):
        start_time = timeit.default_timer()
        matches = matcher.match(query_descriptions, train_descriptions)
        elapsed = timeit.default_timer() - start_time

        self.csv_writer.write_column(len(matches))
        self.csv_writer.write_column(elapsed)

        if len(matches) < min_match:
            print("Not enough matches. found %d out of minimum %d" % \
                    (len(matches), min_match))
            return []
        else:
            return matches


    def knn_match(self, query_descriptions, train_descriptions, matcher, min_match=4):
        start_time = timeit.default_timer()
        knn_matches = matcher.knn_match(query_descriptions, train_descriptions)
        elapsed = timeit.default_timer() - start_time

        self.csv_writer.write_column(len(knn_matches))
        self.csv_writer.write_column(elapsed)

        if len(knn_matches) < min_match:
            print("Not enough matches. found %d out of minimum %d" % \
                    (len(knn_matches), min_match))
            return []
        else:
            return knn_matches

    def experiment(self, image1, image2, detector, descriptor, matchers):

        self.csv_writer.write_column("%s/%s" % (detector.name, descriptor.name))

        (query_features, query_descriptions) = \
                self.detect_and_compute(image1, detector, descriptor)

        (train_features, train_descriptions) = \
                self.detect_and_compute(image2, detector, descriptor)

        for m in matchers:
            matches = self.match(query_descriptions, train_descriptions, m)
            if matches:
                outliers = self.outliers(query_features, train_features, matches)
            else:
                outliers = []

            self.csv_writer.write_column(len(outliers))

            knn_matches = self.knn_match(
                    query_descriptions,
                    train_descriptions,
                    m
                )
            if knn_matches:
                knn_outliers = self.outliers(
                        query_features,
                        train_features,
                        knn_matches
                    )
            else:
                outliers = []

            self.csv_writer.write_column(len(outliers))

        self.csv_writer.write_row()

if __name__ == "__main__":

    if sys.argv < 3:
        exit(0)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file    = os.path.join(current_dir, 'out.csv')

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]


    with open(os.path.join(current_dir, 'benchmark.yml'), 'r') as f:
        config = yaml.load(f)

    detectors = list()
    for algorithm_name in config['detectors']:
        detector = FeatureDetector(algorithm_name)
        if detector != None:
            detectors.append(detector)

    descriptors = list()
    for algorithm_name in config['descriptors']:
        descriptor = FeatureDescriptor(algorithm_name)
        if descriptor != None:
            descriptors.append(descriptor)

    matchers = list()
    for algorithm_name in config['matchers']:
        matcher = FeatureMatcher(algorithm_name)
        if matcher != None:
            matchers.append(matcher)

    benchmark = Benchmark(detectors, descriptors, matchers, csv_file)
    benchmark.instrument(image1_path, image2_path)

