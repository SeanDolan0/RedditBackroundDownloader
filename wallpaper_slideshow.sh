#!/usr/bin/env bash
#
# wallpaper_slideshow.sh — KDE Plasma (Wayland or X11) per-screen, hourly-rotating wallpaper slideshow.
#
# Reads images from a folder and assigns a DISTINCT image to each screen, advancing
# one full screen-set per invocation. Schedule it hourly with the systemd --user
# timer in wallpaper-slideshow.timer (or cron `0 * * * *`). KDE Plasma 6 provides
# org.kde.PlasmaShell.evaluateScript over D-Bus; qdbus6 (or qdbus) is used.
#
# Usage: wallpaper_slideshow.sh [IMAGE_DIR]   (default ./output_images)
#   Env:  QDBUS=qdbus6  STATE_FILE=~/.local/state/wallpaper_slideshow.state
#         FILL_MODE=2    (KDE org.kde.image FillMode; 2=PreserveAspectCrop)
#
# NOTE: run only once IMAGE_DIR actually contains your wallpapers.
set -euo pipefail

IMG_DIR="${1:-./output_images}"
QDBUS="${QDBUS:-$(command -v qdbus6 || command -v qdbus || echo qdbus6)}"
SERVICE="org.kde.plasmashell"
OBJ="/PlasmaShell"
IFACE="org.kde.PlasmaShell"
FILL_MODE="${FILL_MODE:-2}"
STATE_FILE="${STATE_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/wallpaper_slideshow.state}"

# Collect images (null-delimited, sorted).
mapfile -t IMAGES < <(find "$IMG_DIR" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
  -print0 | sort -z | while IFS= read -r -d '' f; do printf '%s\n' "$f"; done)

N=${#IMAGES[@]}
if [ "$N" -eq 0 ]; then
  echo "wallpaper-slideshow: no images in $IMG_DIR; nothing to do" >&2
  exit 0
fi

# Number of connected/enabled screens (KDE reports one Desktop per screen).
# NOTE method is org.kde.PlasmaShell.evaluateScript; capture raw first, then grep
# (avoids `head` + `pipefail` SIGPIPE); don't use `head`.
raw_screens="$("$QDBUS" "$SERVICE" "$OBJ" "$IFACE.evaluateScript" \
  'print(desktops().length);' 2>/dev/null || true)"
SCREENS="$(printf '%s' "$raw_screens" | grep -oE '[0-9]+' | sed -n '1p' || true)"
SCREENS=${SCREENS:-1}
[ "$SCREENS" -ge 1 ] 2>/dev/null || SCREENS=1

# Rotation state: advance one full screen-set per invocation.
offset=0
if [ -f "$STATE_FILE" ]; then offset="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"; fi
offset=$(( (offset + SCREENS) % N ))
mkdir -p "$(dirname "$STATE_FILE")"

# JS-escape a string for embedding in single quotes.
js_escape() { printf '%s' "$1" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g"; }

for s in $(seq 0 $((SCREENS - 1))); do
  idx=$(( (offset + s) % N ))            # distinct image per screen when N >= SCREENS
  img="file://$(realpath "${IMAGES[$idx]}")"
  img_esc="$(js_escape "$img")"
  # Target the Desktop whose .screen == s (fall back to array index).
  "$QDBUS" "$SERVICE" "$OBJ" "$IFACE.evaluateScript" "
    var ds = desktops();
    var d = null;
    for (var i = 0; i < ds.length; i++) { if (ds[i].screen == $s) { d = ds[i]; break; } }
    if (d === null) { d = ds[$s]; }
    d.wallpaperPlugin = 'org.kde.image';
    d.currentConfigGroup = ['Wallpaper', 'org.kde.image', 'General'];
    d.writeConfig('Image', '$img_esc');
    d.writeConfig('FillMode', '$FILL_MODE');
  " >/dev/null 2>&1 || echo "wallpaper-slideshow: failed to set screen $s" >&2
done

echo "$offset" > "$STATE_FILE"
echo "wallpaper-slideshow: $SCREENS screen(s), $N image(s), offset=$offset"
