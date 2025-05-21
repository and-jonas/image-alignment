
import glob
import os
from natsort import natsorted

import cv2
import copy
import numpy as np

from scipy.spatial import distance as dist

def get_series(path_images):
    """
    Creates two lists of file paths: to key point coordinate files and to images
    for each of the samples monitored over time, stored in date-wise folders.
    :return:
    """
    id_series = []

    images = glob.glob(f'{path_images}/*.JPG')
    image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
    uniques = natsorted(np.unique(image_image_id))

    # compile the lists
    for unique_sample in uniques:
        image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
        sample_image_names = [images[i] for i in image_idx]
        # sort to ensure sequential processing of subsequent images
        sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
        id_series.append(sample_image_names)

    return id_series







def make_bbox_overlay(img, pts, box):
    """
    Creates an overlay on the original image that shows the detected marks and the fitted bounding box
    :param img: original image
    :param pts: list of coordinate [x,y] pairs denoting the detected mark positions
    :param box: the box coordinates in cv2 format
    :return: image with overlay
    """
    overlay = copy.copy(img)
    if type(pts) is tuple:
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
        for i in range(len(pts)):
            for point in pts[i]:
                cv2.circle(overlay, (point[0], point[1]), radius=15, color=colors[i], thickness=9)
    else:
        for point in pts:
            cv2.circle(overlay, (point[0], point[1]), radius=15, color=(0, 0, 255), thickness=9)
    if box is not None:
        box_ = np.intp(box)
        cv2.drawContours(overlay, [box_], 0, (255, 0, 0), 9)
    overlay = cv2.resize(overlay, (0, 0), fx=0.25, fy=0.25)
    return overlay

def remove_double_detections(x, y, tol):
    """
    Removes one of two coordinate pairs if their distance is below 50
    :param x: x-coordinates of points
    :param y: y-coordinates of points
    :param tol: minimum distance required for both points to be retained
    :return: the filtered list of points and their x and y coordinates
    """
    point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
    dist_mat = dist.cdist(point_list, point_list, "euclidean")
    np.fill_diagonal(dist_mat, np.nan)
    dbl_idx = np.where(dist_mat < tol)[0].tolist()[::2]
    point_list = np.delete(point_list, dbl_idx, axis=0)
    x = np.delete(x, dbl_idx, axis=0)
    y = np.delete(y, dbl_idx, axis=0)
    return point_list, x, y