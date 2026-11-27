# Reddit Background Downloader

A Python command-line tool that downloads high-resolution images from Reddit, crops them to a wallpaper-friendly aspect ratio with content-aware placement, and saves them under sanitized, collision-free filenames. No Reddit API credentials are required.

Two output modes are supported: a native-resolution 16:9 crop at the source's full quality, or an exact `WxH` output size. Runs interactively with guided prompts or non-interactively for scripts. Includes a test suite, optional source-attribution sidecars, and KDE Plasma scripts for an hourly per-screen wallpaper slideshow.

## Features

- Interactive guided mode (`python scraper.py -i`) or plain CLI for automation.
- Content-aware cropping: CIE Lab color/luminance saliency, texture suppression, rule-of-thirds weighting, boundary-cut avoidance, sky/horizon balance, and optional head-and-shoulders face framing (OpenCV Haar cascade, `smart` mode only).
- Three crop modes: `smart`, `landscape` (saliency without face-pulling), `center`.
- Portrait guardrail: portrait sources are skipped by default (`--min-aspect-ratio 1.0`); opt in with `--allow-portrait`.
- Resolution control: source minimum (`--min-width`/`--min-height`, default 2560x1440), optional exact `--target-resolution WxH`, optional upscaling with `--allow-upscale`, optional sharpness filter (`--min-sharpness`, variance of Laplacian).
- Reliable downloading: one shared `httpx` client, bounded concurrency, retry with exponential backoff, `429` rate-limit handling with `Retry-After`, and a distinction between transient errors and end of listing.
- De-duplication: skip already-saved posts by id (`--skip-existing`) and drop perceptual near-duplicates (`--no-duplicates`, dHash).
- Optional attribution sidecar per image (`--write-metadata`, `<name>.jpg.json`).
- NSFW/spoiler posts skipped by default (`--nsfw` to allow); GIFs skipped; EXIF orientation applied.
- Windows-safe filenames; UTF-8 console output on Windows; works on Windows, macOS, and Linux.
- JPEG output quality control (`--jpeg-quality`, default 95).

## Requirements

- Python 3.10 or newer.
- Dependencies from `requirements.txt`: `httpx`, `numpy`, `opencv-python` (pinned `<5`), `rich`, `questionary`, `pytest`. OpenCV 4.x is required for face detection; on OpenCV 5.x face detection is disabled and cropping falls back to saliency-only. Losing face detection does not affect `landscape` or `center` modes.

## Installation

```bash
git clone https://github.com/SeanDolan0/Reddit-backround-downloader.git
cd Reddit-backround-downloader

# optional but recommended
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Update dependencies later with `pip install -r requirements.txt --upgrade`.

## Usage

### Interactive mode

```bash
python scraper.py --interactive    # or: python scraper.py -i
```

Prompts cover subreddit, image count, sort method (`hot`, `new`, `top`), time range for `top` (`hour`, `day`, `week`, `month`, `year`, `all`), crop mode (`smart`, `landscape`, `center`), and output folder.

### Command line

```bash
# basic
python scraper.py --subreddit EarthPorn --limit 20

# top posts of all time
python scraper.py --subreddit EarthPorn --limit 20 --sort top --time all

# scenery, no face-pulling
python scraper.py --subreddit EarthPorn --limit 20 --crop-mode landscape

# simple centered crop
python scraper.py --subreddit EarthPorn --limit 20 --crop-mode center

# exact output size for your screen; no upscaling unless --allow-upscale
python scraper.py --subreddit EarthPorn --limit 20 --target-resolution 2560x1440

# reject blurry crops, skip already-saved posts and perceptual duplicates
python scraper.py --subreddit EarthPorn --limit 20 --min-sharpness 50 --skip-existing --no-duplicates

# attribution sidecar + custom output folder
python scraper.py --subreddit EarthPorn --limit 20 --write-metadata --output ./my_wallpapers
```

A full 100-wallpaper run that skips portrait shots, enforces 1440p minimums, drops blurry and duplicate images, and uses 6 concurrent downloads:

```bash
python scraper.py --subreddit EarthPorn --limit 100 \
  --sort top --time all \
  --crop-mode landscape \
  --min-aspect-ratio 1.0 \
  --min-width 2560 --min-height 1440 \
  --min-sharpness 15 \
  --no-duplicates --skip-existing \
  --concurrency 6 --jpeg-quality 95 \
  --output ./output_images
```

### Options

Run `python scraper.py --help` for the always-current reference.

| Option | Default | Description |
|--------|---------|-------------|
| `--interactive`, `-i` | off | Launch guided interactive mode |
| `--subreddit` | required (CLI) | Subreddit name without `r/` |
| `--limit` | `50` | Number of images to save; the script pages through Reddit until it reaches this count or runs out of usable posts |
| `--sort` | `hot` | `hot`, `new`, or `top` |
| `--time` | `all` | Time range for `--sort top`: `hour`, `day`, `week`, `month`, `year`, `all` |
| `--output` | `./output_images` | Destination folder, created automatically |
| `--crop-mode` | `smart` | `smart`, `landscape`, or `center` |
| `--face-weight` | `1.0` | Scale face-aware framing in `smart`; `0` disables face-pulling |
| `--concurrency` | `4` | Maximum simultaneous image downloads on one shared client |
| `--target-resolution` | off | Output exactly `WxH` (e.g. `2560x1440`); crop aspect is matched to it |
| `--allow-upscale` | off | Allow upscaling to reach `--target-resolution` |
| `--min-sharpness` | `0` | Reject blurry crops below this Laplacian-variance sharpness; `0` disables |
| `--min-width` | `2560` | Source-crop minimum width in px |
| `--min-height` | `1440` | Source-crop minimum height in px |
| `--min-aspect-ratio` | `1.0` | Minimum source `width/height` ratio before cropping; `0` disables; skips portrait photos |
| `--allow-portrait` | off | Allow portrait images to be cropped; sets `--min-aspect-ratio` to `0` |
| `--skip-existing` | off | Skip posts already saved in the output folder, matched by post id; no re-download or overwrite |
| `--no-duplicates` | off | Skip perceptually near-duplicate images within the run (dHash) |
| `--write-metadata` | off | Write a `<name>.jpg.json` attribution sidecar per image |
| `--nsfw` | off | Allow NSFW and spoiler posts (default skips `over_18` + spoiler) |
| `--jpeg-quality` | `95` | JPEG output quality 1-100 |

## Cropping

Every image is cropped to a target aspect ratio: 16:9 by default, or `W/H` when `--target-resolution WxH` is set.

- `smart` scans candidate windows over a 512 px saliency proxy and picks the best-scoring placement. Scoring combines scale-invariant CIE Lab color/luminance uniqueness, multi-scale center-surround contrast, texture suppression for repetitive grass, gravel, or water, rule-of-thirds and center-band alignment, boundary-cut avoidance, sky/horizon balance, and head-and-shoulders face framing. Faces are detected with OpenCV's `haarcascade_frontalface_default.xml`, kept inside the frame, expanded to head-and-shoulders, and weighted by `--face-weight`.
- `landscape` runs the identical saliency scan with face-pulling disabled. Prefer it for pure scenery.
- `center` takes a simple centered crop.

Filters applied after cropping:

- Source minimum: cropped region must be at least `--min-width` x `--min-height` (default 2560x1440).
- Portrait guardrail: sources with `width/height` below `--min-aspect-ratio` (default `1.0`) are skipped before cropping so vertical shots are not sliced into strips.
- Sharpness: `--min-sharpness` rejects crops whose variance-of-Laplacian score is below the threshold.
- De-duplication: `--skip-existing` and `--no-duplicates`.

Output sizing: by default the output is a native-resolution 16:9 crop that meets the minimum, not resampled to a fixed size. With `--target-resolution WxH`, the crop aspect is matched and the image is resized exactly, using `INTER_AREA` downscaling and `INTER_LANCZOS4` upscaling; upscaling happens only with `--allow-upscale`. EXIF orientation is applied automatically. If `--min-width`/`--min-height` reject everything, lower them or pick a subreddit with larger images; see the summary counters for what was skipped.

## Output

Each run prints a configuration table, per-image progress, and a final summary: elapsed time, pages fetched, posts examined, image candidates, download failures, saved images, and counts skipped for size, portrait aspect, blur, already-existing, duplicate, no image, and errors.

Saved files are JPEG named `<sanitized_title>_<post_id>_<WxH>.jpg`, e.g. `Beautiful_Mountain_1a2b3c_2560x1440.jpg`, sanitized for Windows and collision-free. With `--write-metadata`, a `<name>.jpg.json` sidecar records source URL, post id, subreddit, author, title, and score.

## KDE Plasma wallpaper slideshow (Linux)

`wallpaper_slideshow.sh` assigns a distinct image to each screen from a folder, advancing one full screen-set per invocation. Requires KDE Plasma 6 (`org.kde.PlasmaShell.evaluateScript` over D-Bus) and `qdbus6` or `qdbus`. Reads `.jpg`, `.jpeg`, `.png`, `.webp` from `IMAGE_DIR` (default `./output_images`). Environment overrides: `QDBUS`, `STATE_FILE`, `FILL_MODE` (default `2`, `PreserveAspectCrop`).

```bash
./wallpaper_slideshow.sh ./output_images
```

Schedule hourly rotation with the bundled systemd user units:

```bash
cp wallpaper_slideshow.sh ~/.local/bin/
cp wallpaper-slideshow.service wallpaper-slideshow.timer ~/.config/systemd/user/
# edit wallpaper-slideshow.service: set ExecStart paths and image dir
systemctl --user daemon-reload
systemctl --user enable --now wallpaper-slideshow.timer
```

The bundled `.service` ships a placeholder `ExecStart` path; adjust it to your checkout. Cron alternative: `0 * * * * ~/.local/bin/wallpaper_slideshow.sh /path/to/output_images`. Run only once the image folder is populated.

## How it works

1. Fetch posts from `r/<subreddit>` using Reddit's public listing JSON (`hot`, `new`, `top`; `t=<time>` for `top`; pagination with `after` until the target count or end of listing).
2. Extract image URLs from common Reddit image hosts; skip GIFs, self/stickied posts, and (by default) NSFW/spoiler posts; skip posts already saved when `--skip-existing` is set.
3. Download images concurrently (`--concurrency`) on one shared `httpx` client; retry transient failures with exponential backoff; honor `429` + `Retry-After`.
4. Apply EXIF orientation; reject sources below `--min-width`/`--min-height`, below `--min-aspect-ratio`, or unreadable.
5. Crop to target aspect per `--crop-mode`; reject crops below `--min-sharpness`; filter duplicates when `--no-duplicates` is set.
6. Resize exactly when `--target-resolution` is set (`INTER_AREA` / `INTER_LANCZOS4`; no upscaling without `--allow-upscale`).
7. Save sanitized, collision-free JPEG at `--jpeg-quality`; write attribution sidecar when `--write-metadata` is set.
8. Print summary statistics.

## Reddit access and legal notice

- No user credentials, API key, or OAuth app are needed. Only publicly available content is downloaded.
- The tool reads Reddit's public listing JSON. Since Reddit blocks unauthenticated `www.reddit.com` JSON, it warms up once against `old.reddit.com`, takes the anonymous `token_v2` cookie, and sends it as a Bearer token to `oauth.reddit.com` API calls only; it is never forwarded to image hosts. Requests use a descriptive User-Agent.
- NSFW and spoiler content is skipped unless `--nsfw` is passed.
- Use responsibly and within Reddit's terms of service. For bulk downloads, consider asking Reddit for guidance. Respect image authors' rights; downloaded images remain subject to their original licenses. `--write-metadata` records source and author for attribution.

## Tests

No network access is required; tests use synthetic images and `httpx.MockTransport`.

```bash
python -m pytest -q
```

## Project layout

```
Reddit-backround-downloader/
├── scraper.py                   # Main CLI + cropping + download pipeline
├── test_scraper.py              # pytest suite
├── requirements.txt             # Python dependencies
├── wallpaper_slideshow.sh       # KDE per-screen hourly wallpaper slideshow
├── wallpaper-slideshow.service  # systemd user service
├── wallpaper-slideshow.timer    # systemd hourly timer
├── output_images/               # Default output folder (gitignored)
├── README.md
├── LICENSE
└── .gitignore
```

## Troubleshooting

- `Module not found`: `pip install -r requirements.txt`.
- Face detection inactive: expected on OpenCV 5.x (pinned off); `opencv-python<5` is pinned. Cropping falls back to saliency-only; `landscape`/`center` are unaffected.
- Fewer images than `--limit`: minimums (`--min-width`/`--min-height`, `--min-aspect-ratio`, `--min-sharpness`), skipped portrait shots, NSFW filtering, de-duplication, source posts without images, or Reddit listing caps. Check the summary counters; lower filters or choose a larger-image subreddit.
- Portrait images skipped: default `--min-aspect-ratio 1.0`. Opt in with `--allow-portrait` or `--min-aspect-ratio 0`.
- Output permission denied: ensure write access to `--output`, or use `./output_images`.
- Empty slideshow: populate the image folder first; check `qdbus6`/`qdbus` availability and KDE session.

## Contributing

Issues and pull requests are welcome. Run `python -m pytest -q` before submitting; keep changes consistent with the existing style.

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Sean Dolan.
