import os

import cv2
import numpy as np

# loading im
file_pattern = ''
file_pattern = os.path.join('data_lb2', 'im', 'photo')
im1_path = file_pattern + 'main.jpg'
im2_path = file_pattern + '1.jpg'
img1 = cv2.imread('main.jpg', cv2.IMREAD_COLOR)
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread(im2_path, cv2.IMREAD_COLOR)
# img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

scale_percent = 40  # percent of original size

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)

# akaze

akaze = cv2.AKAZE_create(threshold=0.01)

k1, d1 = akaze.detectAndCompute(img1, None)
k2, d2 = akaze.detectAndCompute(img2, None)

# matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(d1, d2, k=2)
# matches = sorted(matches, key=lambda x: x.distance)
#
#
# matching_result = cv2.drawMatches(img1, k1, img2, k2, matches[:20], None)
#
# matching_result = cv2.resize(matching_result, (1280, 720//2))
# cv2.imshow("AKAZE matching", matching_result)
# cv2.waitKey(0)
goodMatches = []

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        goodMatches.append(m)

MIN_MATCH_COUNT = 10

if len(goodMatches) > MIN_MATCH_COUNT:
    # Get the good key points positions
    sourcePoints = np.float32([k1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    destinationPoints = np.float32([k2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    # Obtain the homography matrix
    M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    matchesMask = mask.ravel().tolist()

    # Apply the perspective transformation to the source image corners
    h, w = img1.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(corners, M)

    # Draw a polygon on the second image joining the transformed corners
    img2 = cv2.polylines(img2, [np.int32(transformedCorners)], True, (255, 255, 255), 2, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
    matchesMask = None

# Draw the matches
drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
result = cv2.drawMatches(img1, k1, img2, k2, goodMatches, None, **drawParameters)
result = cv2.resize(result, (1280, 720 // 2))
# Display the results
cv2.imshow('Homography', result)

cv2.waitKey(0)
