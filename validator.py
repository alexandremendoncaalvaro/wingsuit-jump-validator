import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def get_image(image_path):
    bgr_image = cv2.imread(image_path)
    height, width, channels = bgr_image.shape
    return bgr_image, height, width, channels


def get_wingsuits_keypoints(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
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
    return keypoints


def get_diamond_wingsuits_positions(keypoints):
    wingsuits = keypoints_to_numpy_xy(keypoints)

    left_wingsuit_id = wingsuits[0].index(min(wingsuits[0]))
    right_wingsuit_id = wingsuits[0].index(max(wingsuits[0]))
    top_wingsuit_id = wingsuits[1].index(min(wingsuits[1]))
    bottom_wingsuit_id = wingsuits[1].index(max(wingsuits[1]))

    left_wingsuit = [int(wingsuits[0][left_wingsuit_id]),
                     int(wingsuits[1][left_wingsuit_id])]
    right_wingsuit = [int(wingsuits[0][right_wingsuit_id]),
                      int(wingsuits[1][right_wingsuit_id])]
    top_wingsuit = [int(wingsuits[0][top_wingsuit_id]),
                    int(wingsuits[1][top_wingsuit_id])]
    bottom_wingsuit = [int(wingsuits[0][bottom_wingsuit_id]),
                       int(wingsuits[1][bottom_wingsuit_id])]

    return left_wingsuit, right_wingsuit, top_wingsuit, bottom_wingsuit


def keypoints_to_numpy_xy(keypoints):
    wingsuits = [[k.pt[0] for k in keypoints], [k.pt[1] for k in keypoints]]
    return wingsuits


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def adjust_perspective(image, pts_origin, margin):
    height, width, channels = image.shape

    pts_destiny = np.float32(
        [[0 + margin, 500 + margin], [0 + margin, 0 + margin], [500 + margin, 0 + margin]])
    M = cv2.getAffineTransform(pts_origin, pts_destiny)

    image_destiny = cv2.warpAffine(image, M, (width, height))

    return image_destiny


def calculate_intersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1:  # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:  # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0:  # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1:  # Intersects left
        intersection = b1 - a0
    else:  # No intersection (either side)
        intersection = 0

    return intersection


def check_intersection(rectangle, wingsuit, tolerance):
    intersection = False
    X0, Y0, X1, Y1, = rectangle
    x0, y0, x1, y1 = wingsuit

    AREA = float((x1 - x0) * (y1 - y0))

    width = calculate_intersection(x0, x1, X0, X1)
    height = calculate_intersection(y0, y1, Y0, Y1)
    area = width * height
    percent = area / AREA

    if percent >= tolerance:
        intersection = True

    return intersection


def paint_wingsuits_keypoints(image, keypoints, radius):
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_yellow = (0, 255, 255)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_size = 0.5
    tolerance_border = .95

    height, width, channels = image.shape
    estimated_positions, perpendicular_distance = get_estimated_positions(
        keypoints, height, width)

    # image = cv2.drawKeypoints(image, keypoints, np.array(
    #     []), color_red, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    square_radius = int((perpendicular_distance / 2)*tolerance_border)


    grid_painted = False
    for keypoint in keypoints:
        x, y, diameter = int(keypoint.pt[0]), int(
            keypoint.pt[1]), keypoint.size

        wingsuit_rectangle = [x - radius, y - radius, x + radius, y + radius]
        x0, y0, x1, y1 = wingsuit_rectangle

        wingsuit_fit = False

        for estimated_position in estimated_positions:
            X, Y = estimated_position
            X = int(X)
            Y = int(Y)
            rectange_conference = [X - square_radius, Y -
                                square_radius, X + square_radius, Y + square_radius]
            X0, Y0, X1, Y1 = rectange_conference
            if not grid_painted:
                cv2.rectangle(image, (X0, Y0), (X1, Y1), color_yellow, 1)

            intersected = check_intersection(
                rectange_conference, wingsuit_rectangle, 1)

            if intersected:
                wingsuit_fit = True

        if wingsuit_fit:
            color = color_green
        else:
            color = color_red

        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        
        grid_painted = True
            
    return image


def get_points_closest_from(reference, points, K):
    points.sort(key=lambda K: distance.euclidean(reference, K))
    return points[:K]


def get_estimated_positions(keypoints, height, width):

    wingsuits_pos = [k.pt for k in keypoints]
    total_wingsuits = len(wingsuits_pos)
    square_side_wingsuits = total_wingsuits ** (1/2)

    tl_wingsuit = get_points_closest_from([0, 0], wingsuits_pos, 1)
    tr_wingsuit = get_points_closest_from([width, 0], wingsuits_pos, 1)
    bl_wingsuit = get_points_closest_from([0, height], wingsuits_pos, 1)
    br_wingsuit = get_points_closest_from([height, width], wingsuits_pos, 1)

    top_size = distance.euclidean(tl_wingsuit, tr_wingsuit)
    bottom_size = distance.euclidean(bl_wingsuit, br_wingsuit)
    left_size = distance.euclidean(tl_wingsuit, bl_wingsuit)
    right_size = distance.euclidean(tr_wingsuit, br_wingsuit)

    sides_avg_size = np.average([top_size, bottom_size, left_size, right_size])
    perpendicular_distance = sides_avg_size / (square_side_wingsuits - 1)

    estimated_positions = []

    row = 0
    column = 0
    for item in range(total_wingsuits):
        isColumnFirstRow = int(item % square_side_wingsuits) == 0
        if isColumnFirstRow:
            row = 0

        # print(f'{row}, {column}')
        x, y = tl_wingsuit[0]
        new_x = x + perpendicular_distance * row
        new_y = y + perpendicular_distance * column
        estimated_positions.append([new_x, new_y])

        isColumnLastRow = row == square_side_wingsuits - 1
        if isColumnLastRow:
            column += 1
        row += 1

    return estimated_positions, perpendicular_distance


def main():
    image_path = sys.argv[1]
    
    bgr_image, height, width, channels = get_image(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    keypoints = get_wingsuits_keypoints(bgr_image)
    diamond_left_wingsuit, diamond_right_wingsuit, diamond_top_wingsuit, diamond_bottom_wingsuit = get_diamond_wingsuits_positions(
        keypoints)

    pts_origin_wingsuits = np.float32(
        [diamond_left_wingsuit, diamond_top_wingsuit, diamond_right_wingsuit])
    adjusted_image = adjust_perspective(bgr_image, pts_origin_wingsuits, 100)

    adjusted_keypoints = get_wingsuits_keypoints(adjusted_image)
    max_diameter = max([k.size for k in adjusted_keypoints])
    radius = int(max_diameter / 2)

    image_painted = paint_wingsuits_keypoints(
        adjusted_image, adjusted_keypoints, radius)

    final_image = cv2.cvtColor(image_painted, cv2.COLOR_BGR2RGB)

    # plt.subplot(121), plt.imshow(rgb_image), plt.title('Input')
    plt.subplot(111), plt.imshow(final_image), plt.title('Output')

    plt.show()


if __name__ == '__main__':
    main()
