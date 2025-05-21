from importlib import reload
import LeafImageSeries
reload(LeafImageSeries)
from LeafImageSeries import Series

# load series
leaf = Series(base_dir='C:/Users/anjonas/PycharmProjects/sympathique-wheat', load=('images', 'rois', 'tforms', 'masks', 'leaf_masks', 'targets'))

# # show a frame
# leaf.show_frame(4, show=('masks', 'targets'))

# # show a sequence
# leaf.show_series(interval=1000, show=('masks', 'targets'))

# # repeat the warping of the original image with the provided output
# leaf.warp_images()

# # compare with original output
# leaf.show_series(interval=1000, show=('warped_images', 'targets'))

# filter triangles according to their aspect ratio
leaf.filter_tform_shape(AR_threshold=5)

# filter triangles according to distance from neighbours


# warp images using the filtered tforms
leaf.warp_images(use="filter_shape")

# show warped images with shape outlier triangles excluded
leaf.show_series(interval=1000, show=('warped_images', 'targets'))


