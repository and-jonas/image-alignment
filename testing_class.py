from importlib import reload
import LeafImageSeries
reload(LeafImageSeries)
from LeafImageSeries import Series

leaf = Series(base_dir='Z:/Public/Jonas/Data/ESWW009/SingleLeaf', load=('images', 'rois', 'tforms', 'masks', 'targets'))

# leaf.show_frame(7)
# leaf.show_series(interval=1000, show=('targets'))

leaf.warp_images()
leaf.show_series(interval=1000, show=('targets', 'warped_images'))

leaf.process_tform()
