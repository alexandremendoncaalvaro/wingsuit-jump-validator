import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    result_size = 600

    dst = np.array([
        [0, 0],
        [result_size - 1, 0],
        [result_size - 1, result_size - 1],
        [0, result_size - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (result_size, result_size))

    return warped


image = cv2.imread('examples/005.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = True
params.minArea = 300
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray_image)

color_red = (0, 0, 255)
color_green = (0, 255, 0)
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.5

im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array(
    []), color_red, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

wingsuits = [[k.pt[0] for k in keypoints], [k.pt[1] for k in keypoints]]

max_diameter = max([k.size for k in keypoints])

left_wingsuit_id = wingsuits[0].index(min(wingsuits[0]))
right_wingsuit_id = wingsuits[0].index(max(wingsuits[0]))
top_wingsuit_id = wingsuits[1].index(min(wingsuits[1]))
bottom_wingsuit_id = wingsuits[1].index(max(wingsuits[1]))

left_wingsuit = (int(wingsuits[0][left_wingsuit_id]),
                 int(wingsuits[1][left_wingsuit_id]))
right_wingsuit = (int(wingsuits[0][right_wingsuit_id]),
                  int(wingsuits[1][right_wingsuit_id]))
top_wingsuit = (int(wingsuits[0][top_wingsuit_id]),
                int(wingsuits[1][top_wingsuit_id]))
bottom_wingsuit = (int(wingsuits[0][bottom_wingsuit_id]),
                   int(wingsuits[1][bottom_wingsuit_id]))


corner_wingsuits = [[left_wingsuit[0], left_wingsuit[1]], [top_wingsuit[0], top_wingsuit[1]],
                    [right_wingsuit[0], right_wingsuit[1]], [bottom_wingsuit[0], bottom_wingsuit[1]]]

cv2.line(im_with_keypoints, top_wingsuit, left_wingsuit, color_red, 3)
cv2.line(im_with_keypoints, top_wingsuit, right_wingsuit, color_red, 3)
cv2.line(im_with_keypoints, bottom_wingsuit, left_wingsuit, color_red, 3)
cv2.line(im_with_keypoints, bottom_wingsuit, right_wingsuit, color_red, 3)

pattern_size = distance.euclidean(top_wingsuit, left_wingsuit) / 5

radius = int(max_diameter / 2)

for keypoint in keypoints:
    x, y, diameter = int(keypoint.pt[0]), int(keypoint.pt[1]), keypoint.size
    text = f'x: {x}, y: {y}'
    cv2.circle(im_with_keypoints, (x, y), radius, color_green, 2)
    cv2.putText(im_with_keypoints, text, (x + radius,
                                          y - radius), font, font_size, color_red, 1)

pts = np.array(corner_wingsuits, dtype="float32")
warped = four_point_transform(im_with_keypoints, pts)

cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("Wrapped", warped)
cv2.waitKey(0)
