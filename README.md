# Reddit Background Downloader

A lightweight Python CLI tool that downloads high-resolution Reddit images, intelligently crops them for wallpaper use (16:9 aspect ratio), and saves them with Windows-safe filenames. Features an interactive mode for beginners and traditional CLI for automation.

## ✨ Features

- **Interactive Mode** - Answer simple prompts instead of remembering CLI flags (`python scraper.py -i`)
- **Beautiful CLI Output** - Colored tables, progress notifications, and professional formatting
- **No API Credentials Required** - Uses Reddit's public JSON endpoint
- **Smart Content-Aware Cropping** - Window-scored salient-region placement + optional face-aware framing
- **Landscape Mode** - Smart saliency scan WITHOUT face-pulling, best for pure scenery
- **Center Crop Mode** - Simple fallback for straightforward cropping
- **Reliable Downloading** - One shared HTTP client, bounded concurrency, retry with backoff, 429 rate-limit handling, and transient-vs-end-of-list distinction
- **Resolution Control** - Source minimum (`--min-width/--min-height`, default 2560×1440); optional exact `--target-resolution WxH`, `--allow-upscale`, `--min-sharpness`
- **De-duplication** - Skip already-saved posts (`--skip-existing`) and perceptually near-duplicate images (`--no-duplicates`)
- **Attribution** - Optional source/author sidecar JSON (`--write-metadata`)
- **Clean Processing** - Minimal, quiet output with detailed final summary
- **Windows-Safe Filenames** - Automatically sanitizes filenames and prevents collisions
- **Cross-Platform** - Works on Windows, macOS, and Linux

## 🚀 Quick Start

### For Beginners (Interactive Mode)
```bash
python scraper.py -i
```
Simply answer the prompts and let it run!

### For Scripts/Automation (CLI Mode)
```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort hot
```

### Get Help
```bash
python scraper.py --help
```

## 📋 Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements.txt` (`opencv-python` is pinned `<5`; face detection needs OpenCV 4.x)

## 📦 Installation

### First Time Setup
```bash
# Clone or download the repository
cd Reddit-downloader

# Install dependencies
pip install -r requirements.txt
```

### Updating Dependencies
If you already have the repo installed, update with:
```bash
pip install -r requirements.txt --upgrade
```

## 📖 Usage

### Interactive Mode (Recommended for beginners)

Launch the interactive mode:

```bash
python scraper.py --interactive
```

or shorthand:

```bash
python scraper.py -i
```

You'll be prompted for:
1. **Subreddit name** - Which subreddit to scrape (e.g., `EarthPorn`)
2. **Image count** - How many images to download
3. **Sort method** - `hot`, `new`, or `top`
4. **Time range** - If sorting by `top` (hour, day, week, month, year, all)
5. **Crop mode** - `smart`, `landscape` (saliency, no face-pulling), or `center` (simple center crop)
6. **Output folder** - Where to save the images

### Command Line Mode

#### Basic Usage
```bash
python scraper.py --subreddit EarthPorn --limit 20
```

#### Top Posts All Time
```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort top --time all
```

#### Center Crop Mode
```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort hot --crop-mode center
```

#### Landscape Mode (scenery, no face-pulling)
```bash
python scraper.py --subreddit EarthPorn --limit 20 --crop-mode landscape
```

#### Exact Output Resolution for Your Screen
```bash
# crop to match + resize exactly; no upscaling unless --allow-upscale
python scraper.py --subreddit EarthPorn --limit 20 --target-resolution 2560x1440
```

#### Skip Blurry, Existing, and Duplicate Images
```bash
python scraper.py --subreddit EarthPorn --limit 20 \
  --min-sharpness 50 --skip-existing --no-duplicates
```

#### Source Attribution Sidecar
```bash
python scraper.py --subreddit EarthPorn --limit 20 --write-metadata
```

#### Custom Output Folder
```bash
python scraper.py --subreddit EarthPorn --limit 20 --output ./my_wallpapers
```

#### Full Example
```bash
python scraper.py \
  --subreddit EarthPorn \
  --limit 50 \
  --sort top \
  --time all \
  --crop-mode smart \
  --output "C:\Users\[User]\Pictures\Wallpapers"
```

## 🎯 CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--interactive` | `-i` | N/A | Launch interactive mode with guided prompts |
| `--subreddit` | N/A | Required (CLI) | Subreddit name to scrape (without `r/`) |
| `--limit` | N/A | `50` | Number of images to save |
| `--sort` | N/A | `hot` | Sort method: `hot`, `new`, or `top` |
| `--time` | N/A | `all` | Time range for `--sort top`: `hour`, `day`, `week`, `month`, `year`, `all` |
| `--crop-mode` | N/A | `smart` | Cropping strategy: `smart`, `landscape`, or `center` |
| `--face-weight` | N/A | `1.0` | Scale face-aware framing in `smart`; `0` disables face-pulling |
| `--concurrency` | N/A | `4` | Max simultaneous image downloads (shared client) |
| `--target-resolution` | N/A | off | Output exactly `WxH` (e.g. `2560x1440`); crop aspect matched |
| `--allow-upscale` | N/A | off | Allow upscaling to reach `--target-resolution` |
| `--min-sharpness` | N/A | `0` | Reject blurry crops below this Laplacian-variance sharpness |
| `--min-width` | N/A | `2560` | Source-crop minimum width in px |
| `--min-height` | N/A | `1440` | Source-crop minimum height in px |
| `--skip-existing` | N/A | off | Skip posts already saved (by post id); don't re-download/overwrite |
| `--no-duplicates` | N/A | off | Skip perceptually near-duplicate images (dHash) within the run |
| `--write-metadata` | N/A | off | Write an attribution/source `<name>.jpg.json` sidecar |
| `--nsfw` | N/A | off | Allow NSFW/spoiler posts (default: skip) |
| `--jpeg-quality` | N/A | `95` | JPEG output quality 1-100 |
| `--output` | N/A | `./output_images` | Destination folder for saved images |
| `--help` | `-h` | N/A | Show help message |

## 📊 Output

The tool provides clean, organized output:

1. **Welcome Banner** - Styled title
2. **Configuration Table** - Your selected settings
3. **Processing Messages** - Status updates and save confirmations
4. **Summary Table** - Detailed statistics including:
   - Elapsed time
   - Pages fetched from Reddit
   - Posts examined
   - Image candidates found
   - Download failures
   - Successfully saved images
   - Images skipped (too small, blurry, existing, duplicate, no image, errors)

## 💡 Tips

- **`--limit` refers to saved images**, not posts fetched. The script will page through Reddit until it reaches your target or runs out of usable posts.
- **Face detection** - Only affects `smart` mode; `landscape` and `center` don't use it. If unavailable (or on OpenCV 5.x, where it's pinned off), it falls back to saliency-only cropping. Pure-landscape scenery isn't affected by face-pulling anyway.
- **Output sizing** - By default output is a **native-resolution 16:9 crop** meeting the minimum (`--min-width/--min-height`, default 2560×1440); it is **not** downscaled/upscaled to a fixed size. Use `--target-resolution WxH` for an exact output (`--allow-upscale` to permit upscaling).
- **Output folder** - Created automatically if it doesn't exist.
- **Filenames** - Include post ID to reduce collisions, e.g., `Beautiful_Mountain_1a2b3c_2560x1440.jpg`
- **Smart Crop** - Walks candidate windows, keeps detected faces inside the frame, and picks the most salient position
- **Landscape Crop** - Same saliency scan without face-pulling; good for scenery
- **Center Crop** - Simple centered crop, good for predictable compositions
- **Orientation** - EXIF orientation is applied automatically (OpenCV `IMREAD_COLOR`); GIFs are skipped; NSFW/spoiler posts are skipped unless `--nsfw`.

## 🔧 How It Works

1. **Fetch Posts** - Retrieves posts from the specified subreddit using Reddit's public JSON API (shared client, bounded concurrency, retry + 429 backoff, transient-vs-end-of-list)
2. **Find Images** - Extracts image URLs from common Reddit image hosts; skips GIFs, self/stickied, and (by default) NSFW/spoiler
3. **Download** - Asynchronously downloads high-resolution image files concurrently on one shared HTTP client; retries transient failures
4. **Crop** - Content-aware crop to the target aspect (16:9, or `W/H` from `--target-resolution`); `smart`/`landscape`/`center`; source minimum + optional sharpness filter
5. **Resize (optional)** - Exact `--target-resolution WxH` with `INTER_AREA` downscale / `INTER_LANCZOS4` upscale; no upscaling unless `--allow-upscale`
6. **De-dup** - Optional `--skip-existing` + `--no-duplicates`
7. **Save** - Stores images with sanitized, collision-free filenames; optional attribution sidecar
8. **Report** - Displays comprehensive summary statistics

## 📂 File Structure

```
Reddit-downloader/
├── scraper.py              # Main script
├── test_scraper.py         # pytest suite (synthetic + mocked network)
├── requirements.txt        # Python dependencies (opencv-python<5, pytest)
├── README.md              # This file
├── LICENSE                # License
└──.gitignore             # Git exclusions
```

## 🤝 Contributing

Found a bug or have a feature idea? Feel free to open an issue or submit a pull request!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Legal Notice

- This tool uses Reddit's **public** JSON endpoint - no credentials needed
- Only downloads publicly available content
- Respects Reddit's terms of service by using appropriate user-agents
- For bulk downloads, consider reaching out to Reddit for guidance

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Face detection not working
- This is normal! The script automatically falls back to saliency-based cropping
- Requires OpenCV 4.x (`opencv-python<5` is pinned); on OpenCV 5.x face detection is off
- Face detection only affects `--crop-mode smart`; `landscape`/`center` don't use it; no functionality is lost

### Getting fewer images than requested
- The subreddit might not have enough high-resolution images meeting the criteria
- Try a different subreddit or increase the `--limit` value

### Output folder permission denied
- Ensure you have write permissions to the specified output folder
- Try using `./output_images` instead of a full path

## 🧪 Tests

No network needed (synthetic images + `httpx.MockTransport`):

```bash
pip install -r requirements.txt
pytest -q
```

## 📚 Documentation

- Run `python scraper.py --help` for the full, always-current CLI reference.
- `--write-metadata` produces a `<name>.jpg.json` attribution sidecar per image.

## 🎉 Enjoy!

Download beautiful wallpapers with ease!
