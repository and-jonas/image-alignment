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
from scipy.spatial import Delaunay
from skimage.draw import polygon2mask
from tqdm import tqdm

class Series:

    def __init__(self, base_dir, load=('images', 'tforms', 'rois', 'masks', 'targets')):
        self.base_dir = base_dir
        self.series = utils.get_series(path_images=os.path.join(base_dir, "data", "*", "*"))[0][:5]
        self.leaf_uid = self._extract_leaf_uid(self.series[0])
        self.image_uids = [os.path.basename(p) for p in self.series]
        self.output_base = os.path.join(base_dir, "Output", self.leaf_uid)
        self.output_ts = os.path.join(base_dir, "Output", "ts")

        # Initialize data containers
        self.images = None
        self.tforms = None
        self.tforms_filter_shape = None
        self.rois = None
        self.masks = None
        self.leaf_masks = None
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

        if 'leaf_masks' in load:
            mask_dir = os.path.join(self.output_ts, self.leaf_uid, "leaf_mask")
            self.leaf_masks = self._load_images_from_dir(mask_dir)

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

    def show_frame(self, i, show=('images', 'masks', 'targets')):
        """Display selected elements for a specific frame index.
        
        Parameters:
            i (int): Index of the frame to display.
            show (tuple or str): Elements to show, e.g., ('masks', 'targets')
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if isinstance(show, str):
            show = (show,)  # allow a single string

        available_data = {
            'images': self.images if hasattr(self, 'images') else None,
            'masks': self.masks if hasattr(self, 'masks') else None,
            'targets': self.targets if hasattr(self, 'targets') else None,
        }

        valid_show = [s for s in show if available_data.get(s) is not None]

        if not valid_show:
            raise ValueError("None of the requested elements are available.")

        fig, axs = plt.subplots(1, len(valid_show), sharex=True, sharey=True)
        if len(valid_show) == 1:
            axs = [axs]

        for ax, element in zip(axs, valid_show):
            data_list = available_data[element]
            if i >= len(data_list) or data_list[i] is None:
                img = np.zeros((100, 100))  # fallback if missing
            else:
                img = data_list[i]
            ax.imshow(img)
            ax.set_title(f"{element.capitalize()} - frame {i}")

        fig.suptitle(f"Frame {i+1}", fontsize=16)
        plt.show()


    def show_series(self, interval=500, show=('images', 'masks', 'targets')):
        """Animate the image series (like a GIF), with zoom preserved between frames.
        
        Parameters:
            interval (int): Delay between frames in milliseconds.
            show (tuple): Elements to show, e.g. ('images', 'masks')
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np

        if isinstance(show, str):
            show = (show,)

        fig, axs = plt.subplots(1, len(show), sharex=True, sharey=True)
        if len(show) == 1:
            axs = [axs]

        n_frames = len(self.image_uids)

        def find_first_valid_frame(element):
            data = getattr(self, element)
            for d in data:
                if d is not None:
                    return np.asarray(d)
            return np.zeros((100, 100))  # fallback if all are None

        # Track zoom state per axis
        zoom_state = [None for _ in show]
        images = []

        # Initialize axes with imshow and store the image objects
        for j, element in enumerate(show):
            img_data = find_first_valid_frame(element)
            im = axs[j].imshow(img_data)
            images.append(im)
            axs[j].set_title(f"{element} - frame 0")

        fig.suptitle(f"Frame 1/{n_frames}", fontsize=16)

        def on_xlim_changed(event_ax):
            for j, ax in enumerate(axs):
                if ax == event_ax:
                    zoom_state[j] = (ax.get_xlim(), ax.get_ylim())

        for ax in axs:
            ax.callbacks.connect('xlim_changed', on_xlim_changed)

        def get_frame(i, element):
            data = getattr(self, element)
            if data and data[i] is not None:
                return np.asarray(data[i])
            else:
                return None

        def update(i):
            for j, element in enumerate(show):
                img_data = get_frame(i, element)
                if img_data is not None:
                    images[j].set_data(img_data)
                else:
                    images[j].set_data(np.zeros_like(images[j].get_array()))
                axs[j].set_title(f"{element} - frame {i}")
                if zoom_state[j] is not None:
                    axs[j].set_xlim(zoom_state[j][0])
                    axs[j].set_ylim(zoom_state[j][1])
            fig.suptitle(f"Frame {i+1}/{n_frames}", fontsize=16)

        ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=True)
        plt.show()
        
    
    def warp_images(self, use="full"):
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
            if use == "full":
                tform = self.tforms[i]
            if use == "filter_shape":
                tform = self.tforms_filter_shape[i]
            if tform is not None:
                pw_warped = skimage.transform.warp(crop, tform, output_shape=(h, w))
                pw_warped = skimage.util.img_as_ubyte(pw_warped)
            else:
                print(f"Warning: No transformation found for {image_uid}")
                pw_warped = crop
            self.warped_images.append(pw_warped)

    def filter_tform_shape(self, AR_threshold, out_of_bounds_shift=10000):

        self.tforms_filter_shape = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            tf = self.tforms[i]
            image_uid = self.image_uids[i]

            if tf is None:
                print(f"Warning: No transformation found for {image_uid}")
                self.tforms_filter_shape.append(None)  # Append placeholder
                continue

            # source points and triangulation
            src_pts = tf._tesselation.points
            tri = Delaunay(src_pts)
            triangles = src_pts[tri.simplices]

            shift_affine = np.array([[1, 0, out_of_bounds_shift],
                                    [0, 1, out_of_bounds_shift],
                                    [0, 0, 1]])

            num_kept = 0
            for j, triangle in enumerate(triangles):
                ar = utils.triangle_aspect_ratio(triangle)
                if ar > AR_threshold:
                    tf.affines[j].params[:, :] = shift_affine
                else:
                    num_kept += 1

            kept_ratio = num_kept / len(triangles)
            print(f"{image_uid}: Kept {num_kept} of {len(triangles)} triangles ({kept_ratio:.1%})")

            self.tforms_filter_shape.append(tf)  # Append even if modified



    def filter_tform_shape(self, AR_threshold, out_of_bounds_shift=10000):

        self.tforms_filter_shape = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            
            
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