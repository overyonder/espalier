#!/usr/bin/env bash
# Adds a caption band under an image: white text on black, sized to the image width.
# Usage: caption.sh IMAGE "CAPTION" [OUTPUT]   (OUTPUT defaults to IMAGE with -captioned before the extension)
#        caption.sh -b OUTDIR IMAGE "CAPTION" [IMAGE "CAPTION" ...]   (many images, written to OUTDIR, in parallel)
# Starting nix shell costs seconds; magick itself takes milliseconds. So the shell starts once per call, not per image.
set -euo pipefail
if [[ -z ${CAPTION_IN_SHELL:-} ]]; then
  CAPTION_IN_SHELL=1 exec nix shell nixpkgs#imagemagick nixpkgs#dejavu_fonts -c "$0" "$@"
fi
one() {
  local in=$1 text=$2 out=$3 w size
  w=$(magick identify -format %w "$in")
  size=$(( w / 55 > 14 ? w / 55 : 14 ))
  magick "$in" \
    \( -size "${w}x" -background black -fill white -font DejaVu-Sans -pointsize "$size" \
       -gravity west caption:"$text" -bordercolor black -border "$((size))x$((size / 2))" -resize "${w}x" \) \
    -gravity center -append -define png:compression-level=1 "$out"
  echo "$out"
}
if [[ $1 == -b ]]; then
  dir=$2; shift 2; mkdir -p "$dir"
  while (( $# >= 2 )); do one "$1" "$2" "$dir/$(basename "$1")" & shift 2; done
  wait
else
  in=$1; one "$in" "$2" "${3:-${in%.*}-captioned.${in##*.}}"
fi
