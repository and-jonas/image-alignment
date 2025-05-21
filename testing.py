
import pickle
import json
from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
import os
import re
import skimage

from PIL import Image
import cv2

import utils

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

base_dir = "Z:/Public/Jonas/Data/ESWW009/SingleLeaf"

# # get series of images for individual leaves
# series = utils.get_series(path_images = f'{base_dir}/*/JPEG_cam')
# series = series[0]

# # get paths
# leaf_uid = re.search(r'(ESWW00\d+_\d+)', series[0]).group(1)
# base_dir = 'Z:/Public/Jonas/Data/ESWW009/SingleLeaf'
# roi_dir = os.path.join(base_dir, 'Output', leaf_uid, 'roi')

# # get image ids
# image_uid = [os.path.basename(x) for x in series]

# # load images
# images = [Image.open(f) for f in series]

# # load tforms
# tforms = [None]  # First image has no transform
# for img_path in series[1:]:
#     print(img_path)
#     img_name = os.path.splitext(os.path.basename(img_path))[0]
#     tform_path = os.path.join(roi_dir, f"{img_name}_tform_piecewise.pkl")
#     if os.path.exists(tform_path):
#         with open(tform_path, 'rb') as file:
#             tform = pickle.load(file)
#     else:
#         print(f"Warning: Transformation not found for {img_name}")
#         tform = None
#     tforms.append(tform)

# # load rois
# rois = []
# for img_path in series:
#     print(img_path)
#     img_name = os.path.splitext(os.path.basename(img_path))[0]
#     roi_path = os.path.join(roi_dir, f"{img_name}.json")
#     if os.path.exists(roi_path):
#         with open(roi_path, 'r') as file:
#             roi = json.load(file)
#     else:
#         print(f"Warning: ROI not found for {img_name}")
#         roi = None
#     rois.append(roi)

# # load masks
# masks = []
# mask_dir = os.path.join(base_dir, "Output", leaf_uid, "mask_aligned/piecewise")
# for img_path in series:
#     print(img_path)
#     img_name = os.path.splitext(os.path.basename(img_path))[0]
#     mask_path = os.path.join(mask_dir, f"{img_name}.png")
#     if os.path.exists(mask_path):
#         mask = Image.open(mask_path)
#     else:
#         print(f"Warning: mask not found for {img_name}")
#         mask = None
#     masks.append(mask)

# # load targets
# targets = []
# target_dir = os.path.join(base_dir, "Output", leaf_uid, "result/piecewise")
# for img_path in series:
#     print(img_path)
#     img_name = os.path.splitext(os.path.basename(img_path))[0]
#     target_path = os.path.join(target_dir, f"{img_name}.JPG")
#     if os.path.exists(target_path):
#         target = Image.open(target_path)
#     else:
#         print(f"Warning: target not found for {img_name}")
#         target = None
#     targets.append(target)

# # Store in a structured format
# series_data = {
#     'leaf_uid': leaf_uid,
#     'image_uids': image_uid,
#     'images': images,
#     'rois': rois,
#     'tforms': tforms,  # align with images: None for reference frame
#     'masks': masks, 
#     'targets': targets,
# }

from importlib import reload
from LeafImageSeries import Series
leaf = Series(base_dir='Z:/Public/Jonas/Data/ESWW009/SingleLeaf', load=('images', 'rois', 'tforms', 'masks', 'targets'))
# leaf.show_frame(7)
# leaf.show_series(interval=1000, show=('targets'))
# leaf.warp_images()
leaf.show_series(interval=1000, show=('targets', 'warped_images'))







for i in range(len(series_data["images"])):
    print(i)
    projective_warped = None
    image_uid = series_data["image_uids"][i]
    img = np.asarray(series_data["images"][i])
    msk = np.asarray(series_data["masks"][i])
    
    # rotate 
    roi = series_data["rois"][i]
    M_img= np.asarray(roi["rotation_matrix"])
    rows, cols = img.shape[0], img.shape[1]
    img_rot = cv2.warpAffine(img, M_img, (cols, rows))
    
    # crop
    box = np.asarray(roi["bounding_box"])
    crop = img_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]
    
    # warp
    tform = series_data["tforms"][i]
    if tform is not None:
        projective_warped = skimage.transform.warp(crop, tform, output_shape=(938, 7097))
        projective_warped = skimage.util.img_as_ubyte(projective_warped)
    else:
        print(f"Warning: No transformation found for {image_uid}")
        projective_warped = crop




i = 2

leaf_uid = series_data["leaf_uid"]
image_uid = series_data["image_uids"][i]

img = series_data["images"][i]
img = np.asarray(img)

msk = series_data["masks"][i]
msk = np.asarray(msk)

# rotate 
roi = series_data["rois"][i]
M_img= np.asarray(roi["rotation_matrix"])
rows, cols = img.shape[0], img.shape[1]
img_rot = cv2.warpAffine(img, M_img, (cols, rows))
# crop
box = np.asarray(roi["bounding_box"])
crop = img_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]
# warp
tform_piecewise = series_data["tforms"][i]
projective_warped = skimage.transform.warp(crop, tform_piecewise,
                                            output_shape=(938, 7097))
projective_warped = skimage.util.img_as_ubyte(projective_warped)

# load result
target = series_data["targets"][i]
target = np.asarray(target)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(target)
axs[0].set_title('projective_warped')
axs[1].imshow(msk)
axs[1].set_title('warped')
plt.show(block=True)
# OK

# ====================================================================================

# get piecewise affine transformatin
matrices = [tform.params for tform in tform_piecewise.affines]

# get source points
src_pts = tform_piecewise._tesselation.points

# Perform Delaunay triangulation
tri = Delaunay(src_pts)

# Plot image and overlay tessellation
plt.imshow(crop)
plt.triplot(src_pts[:, 0], src_pts[:, 1], tri.simplices, color='red', linewidth=1)
plt.scatter(src_pts[:, 0], src_pts[:, 1], color='blue', s=10)
plt.title("Tessellation Overlaid on Image")
plt.axis('off')  # Optional: turn off axis for cleaner image
plt.show()

plt.imshow(crop)
plt.triplot(src_pts[:, 0], src_pts[:, 1], tri.simplices, color='red', linewidth=1)
plt.scatter(src_pts[:, 0], src_pts[:, 1], color='blue', s=10)

# Annotate triangles with their index at the centroid
for i, triangle in enumerate(tri.simplices):
    # Get the coordinates of the triangle's vertices
    pts = src_pts[triangle]
    # Compute centroid
    centroid = np.mean(pts, axis=0)
    plt.text(centroid[0], centroid[1], str(i), color='yellow', fontsize=8, ha='center', va='center')

plt.title("Tessellation Overlaid on Image with Segment Indices")
plt.axis('off')
plt.show()



diff_norms = []
for i in range(len(matrices) - 1):
    diff = matrices[i+1] - matrices[i]
    norm = np.linalg.norm(diff, ord='fro')  # Frobenius norm
    diff_norms.append(norm)

plt.plot(diff_norms, marker='x')
plt.title("Frobenius Norm of Differences Between Adjacent Transformations")
plt.xlabel("Segment Index")
plt.ylabel("Difference Norm")
plt.show()

# get source points
src_pts = tform_piecewise._tesselation.points

# Perform Delaunay triangulation
tri = Delaunay(src_pts)

# get image
img = Image.open("Z:/Public/Jonas/Data/ESWW009/SingleLeaf/20240604/JPEG_cam/20240604_091814_ESWW0090023_1.JPG")
img = np.asarray(img)
plt.imshow(img)
plt.show()

# get bounding box
with open("Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output/ESWW0090023_1/roi/20240604_091814_ESWW0090023_1.json") as file:
    roi = json.load(file)

# rotate image
M = np.asarray(roi["rotation_matrix"])
rows, cols = img.shape[0], img.shape[1]
img_rot = cv2.warpAffine(img, M, (cols, rows))

box = np.asarray(roi["bounding_box"])
crop = img_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

# Plot image and overlay tessellation
plt.imshow(crop)
plt.triplot(src_pts[:, 0], src_pts[:, 1], tri.simplices, color='red', linewidth=1)
plt.scatter(src_pts[:, 0], src_pts[:, 1], color='blue', s=10)
plt.title("Tessellation Overlaid on Image")
plt.axis('off')  # Optional: turn off axis for cleaner image
plt.show()

plt.imshow(crop)
plt.triplot(src_pts[:, 0], src_pts[:, 1], tri.simplices, color='red', linewidth=1)
plt.scatter(src_pts[:, 0], src_pts[:, 1], color='blue', s=10)

# Annotate triangles with their index at the centroid
for i, triangle in enumerate(tri.simplices):
    # Get the coordinates of the triangle's vertices
    pts = src_pts[triangle]
    # Compute centroid
    centroid = np.mean(pts, axis=0)
    plt.text(centroid[0], centroid[1], str(i), color='yellow', fontsize=8, ha='center', va='center')

plt.title("Tessellation Overlaid on Image with Segment Indices")
plt.axis('off')
plt.show()



# ======================================================================================================================== 

# get keypoint coords
coords = pd.read_table("Z:/Public/Jonas/Data/ESWW009/SingleLeaf/20240604/JPEG_cam/runs/pose/predict/labels/20240604_091814_ESWW0090023_1.txt", header=None, sep=" ")
x = coords.iloc[:, 5] * 8192
y = coords.iloc[:, 6] * 5464
kpts, x, y = utils.remove_double_detections(x=x, y=y, tol=50)

kpts = np.intp(cv2.transform(np.array([kpts]), M))[0]

box = np.asarray(roi["bounding_box"])
overlay = utils.make_bbox_overlay(img_rot, kpts, box)

crop = img_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

# Plot image and overlay tessellation
plt.imshow(crop)
plt.triplot(src_pts[:, 0], src_pts[:, 1], tri.simplices, color='red', linewidth=1)
plt.scatter(src_pts[:, 0], src_pts[:, 1], color='blue', s=10)
plt.title("Tessellation Overlaid on Image")
plt.axis('off')  # Optional: turn off axis for cleaner image
plt.show()


# get target dimensions
init_crop = Image.open("Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output/ESWW0090023_1/result/piecewise/20240528_094338_ESWW0090023_1.JPG")
init_crop = np.asarray(init_crop)
init_roi_height, init_roi_width = init_crop.shape[:2]

projective_warped = skimage.transform.warp(crop, tform_piecewise,
                                            output_shape=(init_roi_height, init_roi_width))
plt.imshow(projective_warped)
plt.show()

target = Image.open("Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output/ESWW0090023_1/result/piecewise/20240604_091814_ESWW0090023_1.JPG")
target = np.asarray(target)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(target)
axs[0].set_title('mask')
axs[1].imshow(projective_warped)
axs[1].set_title('density')
plt.show(block=True)

# Plot image and overlay tessellation
plt.imshow(crop)
plt.triplot(src_pts[:, 0], src_pts[:, 1], tri.simplices, color='red', linewidth=1)
plt.scatter(src_pts[:, 0], src_pts[:, 1], color='blue', s=10)
plt.title("Tessellation Overlaid on Image")
plt.axis('off')  # Optional: turn off axis for cleaner image
plt.show()