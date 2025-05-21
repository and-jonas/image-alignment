import os
import re
import json
import pickle
from PIL import Image
import utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import skimage
from tqdm import tqdm

class Series:

    def __init__(self, base_dir, load=('images', 'tforms', 'rois', 'masks', 'targets')):
        self.base_dir = base_dir
        self.series = utils.get_series(path_images=os.path.join(base_dir, "*", "JPEG_cam"))[0][:5]
        self.leaf_uid = self._extract_leaf_uid(self.series[0])
        self.image_uids = [os.path.basename(p) for p in self.series]
        self.output_base = os.path.join(base_dir, "Output", self.leaf_uid)

        # Initialize data containers
        self.images = None
        self.tforms = None
        self.rois = None
        self.masks = None
        self.targets = None
        self.warped_images = None

        self._load_requested(load)

    def _extract_leaf_uid(self, path):
        return re.search(r'(ESWW00\d+_\d+)', path).group(1)

    def _load_requested(self, load):
        if 'images' in load:
            self.images = [Image.open(p) for p in self.series]

        if 'tforms' in load:
            roi_dir = os.path.join(self.output_base, "roi")
            self.tforms = [None]
            for path in self.series[1:]:
                name = os.path.splitext(os.path.basename(path))[0]
                path_tform = os.path.join(roi_dir, f"{name}_tform_piecewise.pkl")
                if os.path.exists(path_tform):
                    with open(path_tform, 'rb') as f:
                        self.tforms.append(pickle.load(f))
                else:
                    print(f"Warning: tform not found for {name}")
                    self.tforms.append(None)

        if 'rois' in load:
            roi_dir = os.path.join(self.output_base, "roi")
            self.rois = []
            for path in self.series:
                name = os.path.splitext(os.path.basename(path))[0]
                path_roi = os.path.join(roi_dir, f"{name}.json")
                if os.path.exists(path_roi):
                    with open(path_roi, 'r') as f:
                        self.rois.append(json.load(f))
                else:
                    print(f"Warning: ROI not found for {name}")
                    self.rois.append(None)

        if 'masks' in load:
            mask_dir = os.path.join(self.output_base, "mask_aligned", "piecewise")
            self.masks = self._load_images_from_dir(mask_dir)

        if 'targets' in load:
            target_dir = os.path.join(self.output_base, "result", "piecewise")
            self.targets = self._load_images_from_dir(target_dir)

    def _load_images_from_dir(self, dir_path):
        result = []
        for path in self.series:
            name = os.path.splitext(os.path.basename(path))[0]
            img_path = os.path.join(dir_path, f"{name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(dir_path, f"{name}.JPG")
            if os.path.exists(img_path):
                result.append(Image.open(img_path))
            else:
                print(f"Warning: file not found for {name}")
                result.append(None)
        return result

    def show_frame(self, i):
        """Display image, mask, and/or target for a specific index."""
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        if self.masks:
            axs[0].imshow(self.masks[i])
            axs[0].set_title("Mask")
        if self.targets:
            axs[1].imshow(self.targets[i])
            axs[1].set_title("Target")
        plt.show()

    def show_series(self, interval=500, show=('images', 'masks', 'targets')):
        """Animate the image series (like a GIF).
        
        Parameters:
            interval (int): Delay between frames in milliseconds.
            show (tuple): Elements to show, e.g. ('images', 'masks')
        """

        if isinstance(show, str):
            show = (show,)  # auto-wrap single string in a tuple

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        if len(show) == 1:
            axs = [axs]  # Ensure it's iterable

        n_frames = len(self.image_uids)

        def get_frame(i, element):
            data = getattr(self, element)
            if data and data[i] is not None:
                return np.asarray(data[i])
            else:
                return np.zeros((100, 100))  # blank fallback

        def update(i):
            for j, element in enumerate(show):
                axs[j].clear()
                axs[j].imshow(get_frame(i, element))
                axs[j].set_title(f"{element} - frame {i}")
            fig.suptitle(f"Frame {i+1}/{n_frames}", fontsize=16)

        ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=True)
        plt.show()
    
    
    def warp_images(self):
        self.warped_images = []
        for i in tqdm(range(len(self.images)), desc="Processing series"):

            # get elements
            pw_warped = None  # re-initilize for each iteration
            image_uid = self.image_uids[i]
            img = np.asarray(self.images[i])
            msk = np.asarray(self.masks[i])
            
            # rotate 
            roi = self.rois[i]
            M_img= np.asarray(roi["rotation_matrix"])
            rows, cols = img.shape[0], img.shape[1]
            img_rot = cv2.warpAffine(img, M_img, (cols, rows))
            
            # crop
            box = np.asarray(roi["bounding_box"])
            crop = img_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

            # get target dimensions from initial frame
            if i == 0:
                h, w = crop.shape[:2]
            
            # warp
            tform = self.tforms[i]
            if tform is not None:
                pw_warped = skimage.transform.warp(crop, tform, output_shape=(h, w))
                pw_warped = skimage.util.img_as_ubyte(pw_warped)
            else:
                print(f"Warning: No transformation found for {image_uid}")
                pw_warped = crop
            self.warped_images.append(pw_warped)

    def process_tform(self):

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            # get piecewise affine transformatin
            tf = self.tforms[i]
            image_uid = self.image_uids[i]

            if tf is not None:
                matrices = [tform.params for tform in tf.affines]
            else:
                print(f"Warning: No transformation found for {image_uid}")
                continue

            # get differences between adjacent triangles in piecewise transform
            diff_norms = []
            for j in range(len(matrices) - 1):
                diff = matrices[j+1] - matrices[j]
                norm = np.linalg.norm(diff, ord='fro')  # Frobenius norm
                diff_norms.append(norm)

            plt.plot(diff_norms, marker='x')
            plt.title("Frobenius Norm of Differences Between Adjacent Transformations")
            plt.xlabel("Segment Index")
            plt.ylabel("Difference Norm")
            plt.show(block=True)