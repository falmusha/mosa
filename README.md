# mosa

[WIP] Python Image Processing module to stitch a series of similar images to
create a mosaic.

# Dependencies

* OpenCV 2.4
* Python 2.7

# Demo

![demo](demo.gif)

To run an interactive demo. Make sure you have OpenCV 2.4 with python bindings
loaded in `PYTHONPATH`. There is a script that'll walk you through the
stitching process in a GUI.

```
cd to/mosa/path
python scripts/demo.py
```
If you have the dependencies installed, this should pop up the hummingbird image
with the features found and linked. To go to the next frame, type `q`.

# Benchmarking

You can benchmark different OpenCV feature detection and matching algorithms to
see how they perform against each other given two images.

The benchmarking script will output a CSV file and mix and match the algorithms
specified in the `benchmark.yml` file in
[here](https://github.com/iFahad7/mosa/blob/master/scripts/benchmark/benchmark.yml).
The result are the of benchmarking on two images are:

* Number of keyponints
* Time to find keypoints
* Number of features
* Time to find features
* Number of matches
* Time to find matches
* Number of outliers
* Number of KNN matches
* Time to find KNN matches
* Number of KNN outliers

Right now, only the following algorithms are supported.
* Detectors/Descriptors
  * [SIFT](http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html?highlight=sift#sift)
  * [SURF](http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html?highlight=surf#surf)
  * [FAST](http://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html#fast)
* Matchers
  * [FLANN](http://docs.opencv.org/2.4/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html)

To run the benchmark, do the following:

```
cd to/mosa/path
python scripts/benchmark/benchmark.py path/to/image1 path/to/image2
```
The above script will generate a `out.csv` file in `most/scripts/benchmark`.
