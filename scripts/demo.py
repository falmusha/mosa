import os
import cv2

from mosa.image_stitcher import *
from mosa.feature_detector import *
from mosa.feature_descriptor import *
from mosa.feature_matcher import *

def test_on_dir(path, detector, descriptor, matcher):
    stitcher = ImageStitcher(detector, descriptor, matcher, True)

    files = list()
    for f in os.listdir(path):
        if f.endswith('.jpg'):
            files.append(os.path.join(path, f))

    files.sort()
    out = cv2.imread(files.pop(0))

    for f in files:
        n = cv2.imread(f)
        out = stitcher.stitch(out, n)
        if out == None:
            out = n
            break

    cv2.imwrite('out.jpg', out)

algs = {
    'hummingbird': {
        'detector': {
            'name': 'fast',
            'opts': {}
        },
        'descriptor': {
            'name': 'surf',
            'opts': {}
        },
        'matcher': {
            'name': 'flann',
            'opts': {}
        }
    },
    'paper_texture': {
        'detector': {
            'name': 'fast',
            'opts': {}
        },
        'descriptor': {
            'name': 'surf',
            'opts': {}
        },
        'matcher': {
            'name': 'flann',
            'opts': {}
        }
    }
}

if __name__ == "__main__":

    # The test_images directory should contain subdirectories of sequences of
    # images to mosaic
    current_dir   = os.path.dirname(os.path.realpath(__file__))
    sample_images = os.path.join(current_dir, '..', 'test_images')

    for root, dirs, files in os.walk(sample_images, topdown=False):
        for d in dirs:
            if algs.has_key(d):
                detector_name = algs[d]['detector']['name']
                descriptor_name = algs[d]['descriptor']['name']
                matcher_name = algs[d]['matcher']['name']

                detector = FeatureDetector(detector_name)
                descriptor = FeatureDescriptor(descriptor_name)
                matcher = FeatureMatcher(matcher_name)

                test_on_dir(
                        (os.path.join(root, d)),
                        detector,
                        descriptor,
                        matcher
                    )
