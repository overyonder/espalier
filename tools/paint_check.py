"""Compare a painting with its input render: same pose, size and placement, judged by silhouette."""
import math

import numpy as np
from PIL import Image

MEASURE_RESOLUTION = 512
SIZE_TOLERANCE     = 1.15  # painted extent over input extent, on either axis
OFFSET_TOLERANCE   = 0.05  # centre shift as a fraction of the input's longest side
SHAPE_TOLERANCE    = 0.30  # 1 - overlap once size and placement are removed


def silhouette_mask(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGBA').resize((MEASURE_RESOLUTION,) * 2, Image.BILINEAR))
    return (pixels[..., 3] > 128) & (pixels[..., :3].max(2) > 12)


def bounding_box(mask):
    # Two pixels per row or column, so a stray speck on the black canvas doesn't move the box.
    columns = np.flatnonzero(mask.sum(0) >= 2)
    rows = np.flatnonzero(mask.sum(1) >= 2)
    if not len(columns) or not len(rows):
        return None
    return columns[0], rows[0], columns[-1] + 1, rows[-1] + 1


def measure_painting_against_render(rendered, painted):
    """Return size ratios, centre offset, shape overlap and a score that passes at 1 or below."""
    input_mask, painted_mask = silhouette_mask(rendered), silhouette_mask(painted)
    input_box, painted_box = bounding_box(input_mask), bounding_box(painted_mask)
    if input_box is None:
        raise ValueError(f'Input render is empty: {rendered}')
    if painted_box is None:
        return {'score': math.inf, 'width_ratio': 0.0, 'height_ratio': 0.0, 'offset': math.inf, 'shape': 0.0}
    input_width, input_height = input_box[2] - input_box[0], input_box[3] - input_box[1]
    painted_width, painted_height = painted_box[2] - painted_box[0], painted_box[3] - painted_box[1]
    # Four pixels of slack keep a one-pixel difference on a thin object from reading as a large ratio.
    width_ratio = (painted_width + 4) / (input_width + 4)
    height_ratio = (painted_height + 4) / (input_height + 4)
    offset = math.hypot((painted_box[0] + painted_box[2] - input_box[0] - input_box[2]) / 2,
                        (painted_box[1] + painted_box[3] - input_box[1] - input_box[3]) / 2) / max(input_width, input_height)
    painted_crop = Image.fromarray(painted_mask[painted_box[1]:painted_box[3], painted_box[0]:painted_box[2]].astype('u1') * 255)
    fitted = np.asarray(painted_crop.resize((int(input_width), int(input_height)), Image.BILINEAR)) > 127
    input_crop = input_mask[input_box[1]:input_box[3], input_box[0]:input_box[2]]
    shape = float((input_crop & fitted).sum() / max((input_crop | fitted).sum(), 1))
    size_error = max(abs(math.log(width_ratio)), abs(math.log(height_ratio))) / math.log(SIZE_TOLERANCE)
    score = max(size_error, offset / OFFSET_TOLERANCE, (1 - shape) / SHAPE_TOLERANCE)
    return {'score': score, 'width_ratio': width_ratio, 'height_ratio': height_ratio, 'offset': offset, 'shape': shape}
