import cv2
import numpy as np
import math
import pdb

class ImageStitcher:

    def __init__(self, detector, descriptor, matcher, debug_mode=False):
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.debug_mode = debug_mode
        self.matcher = matcher

    def show(self, name, img):
        """
        Show a window to view the image
        """
        # Only show if is in debug mode
        if not self.debug_mode:
            return
        cv2.imshow(name, img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(name)


    def crop_black_edges(self, img):
        """
        Remove extra black space around an image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        crop = img[y:y+h, x:x+w]

        return crop

    def draw_matches(self, img1, img2, pts1, pts2):
        """
        Return an image that puts img1 and img2 side-by-side and draw lines
        between the matches indicated by pts1, pts2
        """
        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]

        # Size of new image of the img1 and img2 side-by-side
        h = max(h1, h2)
        w = w1 + w2

        if len(img1.shape) > 2 and len(img1.shape) > 2:
            # Create empty colored image
            out = np.zeros((h, w, 3), np.uint8)
        else:
            # Create empty grayscale image
            out = np.zeros((h, w), np.uint8)

        # create new image
        out[:h1, :w1] = img1
        out[:h2, w1:w1+w2] = img2

        for i in range(len(pts1)):
            x1, y1 = (int(pts1[i][0]), int(pts1[i][1]))
            x2, y2 = (int(pts2[i][0]), int(pts2[i][1]))

            # Draw line between the two points
            cv2.line(
                    out,
                    (x1, y1),
                    (w1+x2, y2),
                    (0, 0, 255), # red colored line
                )

        return out

    def point_by_homography(self, (x, y), homography):
        """
        Transform the points (x, y) to image plane specified by the homography
        matrix
        """
        # pixel vector is [x, y, 1]
        pt_vector = np.ones(3, np.float32)
        pt_vector[0] = x
        pt_vector[1] = y

        # matrix shape should be (3,1) = [[x], [y], [1]]
        pt_matrix = np.matrix(homography, np.float32) \
                * np.matrix(pt_vector, np.float32).T

        # TODO: divide by pt_matrix[2, 0]. Theoratically, it should be 1. But
        # you never know -_-
        # return tuple (x, y)
        return (pt_matrix[0, 0], pt_matrix[1, 0])

    def dimensions_by_homography(self, img_size, homography):
        """
        Transform the dimesnsions of an image to the plane specified by the
        homography matrix
        """
        (y, x) = img_size[:2]

        corners = [(0, 0), (x, 0), (0, y), (x, y)]
        corners = [self.point_by_homography(c, homography) for c in corners]

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]

        min_x = int(math.ceil(min(xs)))
        min_y = int(math.ceil(min(ys)))
        max_x = int(math.ceil(max(xs)))
        max_y = int(math.ceil(max(ys)))

        return (min_x, min_y, max_x, max_y)

    def calculate_output_image_size(self, img1_size, img2_size, homography):
        # find dimenstions of image2 in the image plane of image1
        (min_x, min_y, max_x, max_y) = \
                self.dimensions_by_homography(img2_size, homography)

        # The output image should not be smaller then img1
        max_x = max(max_x, img1_size[1])
        max_y = max(max_y, img1_size[0])

        # The above values can be negative, so should account for that as offset
        # in the result stitched image
        x_offset = 0
        y_offset = 0

        if min_x < 0:
            x_offset += -(min_x)
            max_x += -(min_x)
        if min_y < 0:
            y_offset += -(min_y)
            max_y += -(min_y)

        offset = (x_offset, y_offset)
        size   = (max_y, max_x)

        return (size, offset)

    def detect_and_compute(self, image):
        """
        Find the keypoints in an image and return a feature vector for the found
        keypoints
            kp_a  is keypoint algorithim,
            des_a is decription algorithim
        """
        kp = self.detector.detect(image)
        return self.descriptor.compute(image, kp)

    def find_homography(self, src_pts, dst_pts):
        """
        Find the homogrpahy matrix between the two sets of image points and
        return a tuple of the matrix and the outlier founds
        """
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        outlier_indices = []
        for i, m in enumerate(mask):
            if m[0] == 0:
                outlier_indices.append(i)
        return (H, outlier_indices)

    def stitch(self, img1, img2, min_match=4):
        """
        Stitch two images together given a keypoint detection and description
        algorithims. Must match al least 4 points to be able to stitch
            kp_a  is keypoint algorithim,
            des_a is decription algorithim
        """
        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)

        matches = self.matcher.knn_match(des1, des2)

        if len(matches) < min_match:
            print("Not enough matches. found %d out of minimum %d" % \
                    (len(matches), min_match))
            return None

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

        (H, outliers) = self.find_homography(dst_pts, src_pts)

        valid_matches = len(matches)-len(outliers)
        if  valid_matches < min_match:
            print("Not enough matches. found %d out of minimum %d" % \
                    (len(valid_matches), min_match))
            return None

        src_pts = np.delete(src_pts, outliers, 0)
        dst_pts = np.delete(dst_pts, outliers, 0)

        drawn_matches = self.draw_matches(img1, img2, src_pts, dst_pts)
        self.show('new matches', drawn_matches)

        # Calculate the output image size given the homography transformation
        (output_size, homography_offset) = self.calculate_output_image_size(
                img1.shape,
                img2.shape,
                H
            )

        output_h = output_size[0] # y
        output_w = output_size[1] # x

        homography_offset_matrix = np.matrix(np.identity(3), np.float32)
        homography_offset_matrix[0,2] = homography_offset[0]
        homography_offset_matrix[1,2] = homography_offset[1]

        # Account for the offset in the homography matrix
        H[0,2] += homography_offset[0]
        H[1,2] += homography_offset[1]

        # Warp img1 into a new image of the desired output image size to
        # account for the offset created from overlaying img2 on img1
        warped_img1 = cv2.warpPerspective(
                    img1,
                    homography_offset_matrix,
                    (output_w, output_h)
                )

        # Warp img2 into a trasnformed new image to properly overlay img2 on
        # img1
        warped_img2 = cv2.warpPerspective(
                    img2,
                    H,
                    (output_w, output_h)
                )

        # Create a mask of img2
        warped_gray_img2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
        (_, warped_img2_mask) = cv2.threshold(
                    warped_gray_img2,
                    0,
                    255,
                    cv2.THRESH_BINARY
                )

        # Output colored image, initially empty
        out = np.zeros((output_h, output_w, 3), np.uint8)

        # Add img1 to output image, and only add pixels that DONT overlay with
        # img2
        out = cv2.add(
                out,
                warped_img1,
                mask=np.bitwise_not(warped_img2_mask),
                dtype=cv2.CV_8U)

        # Add img1 to output image
        out = cv2.add(out, warped_img2, dtype=cv2.CV_8U)
        out = self.crop_black_edges(out)

        self.show('out', out)

        return out
