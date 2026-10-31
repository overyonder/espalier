#!/usr/bin/env bash
# Puts a painting's object back where the render had it: image models sometimes paint the object larger or shifted,
# and a projection needs every pixel where the render put it. The object's box is measured in the render, from its
# transparent background, and in the painting, from its plain background; the painting's object is cropped, scaled to
# the render's box and placed on a canvas the render's size. One scale and offset per view, so nothing is warped.
#   tools/align_silhouette.sh <render.png> <painting.png> <aligned.png> [white|black]
set -eo pipefail
render=$1 painting=$2 out=$3 background=${4:-black}
MAGICK=${MAGICK:-$(command -v magick || command -v convert || true)}

if [ -n "$MAGICK" ]; then
    read -r rw rh rx ry < <("$MAGICK" "$render" -alpha extract -trim -format '%w %h %X %Y\n' info:)
    read -r pw ph px py < <("$MAGICK" "$painting" -fuzz 8% -trim -format '%w %h %X %Y\n' info:)
    size=$("$MAGICK" identify -format '%wx%h' "$render")
    "$MAGICK" "$painting" -crop "${pw}x${ph}${px}${py}" +repage -resize "${rw}x${rh}!" \
      \( -size "$size" "xc:$background" \) +swap -geometry "+${rx#+}+${ry#+}" -composite "$out"
else
    # Pure Python fallback using Pillow and NumPy
    python3 -c "
import sys, numpy as np
from PIL import Image

render = Image.open('$render').convert('RGBA')
painting = Image.open('$painting').convert('RGB')
alpha = render.getchannel('A')
render_bbox = alpha.getbbox()
if not render_bbox:
    painting.save('$out'); sys.exit(0)
rx0, ry0, rx1, ry1 = render_bbox
rw, rh = rx1 - rx0, ry1 - ry0

p_arr = np.array(painting, dtype=np.float32)
bg_val = 0.0 if '$background' == 'black' else 255.0
diff = np.abs(p_arr - bg_val)
mask = np.any(diff > 20.4, axis=-1)
if not np.any(mask):
    painting.save('$out'); sys.exit(0)
y_indices, x_indices = np.where(mask)
py0, py1 = int(y_indices.min()), int(y_indices.max()) + 1
px0, px1 = int(x_indices.min()), int(x_indices.max()) + 1

cropped = painting.crop((px0, py0, px1, py1))
resized = cropped.resize((rw, rh), Image.Resampling.LANCZOS)
bg_color = (0, 0, 0) if '$background' == 'black' else (255, 255, 255)
canvas = Image.new('RGB', render.size, color=bg_color)
canvas.paste(resized, (rx0, ry0))
canvas.save('$out')
"
fi
