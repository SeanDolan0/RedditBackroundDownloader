# Reddit Background Downloader

A small Python CLI tool that downloads high-resolution Reddit images, crops them for wallpaper use, and saves the results with Windows-safe filenames.

## Features

- Uses Reddit's public JSON endpoint, so no API credentials are required
- Resolves common Reddit image hosts like `i.redd.it`, `preview.redd.it`, and imgur
- Supports content-aware cropping with saliency and face detection
- Includes a `center` crop mode when you want a simple fallback
- Crops to 16:9 and enforces a minimum resolution for wallpaper quality
- Prints a concise summary at the end of each run

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Basic run:

```bash
python scraper.py --subreddit EarthPorn --limit 100 --sort hot --output "C:\Users\[User]\Desktop\Backround Immages"
```

Top posts across all time:

```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort top --time all --output "C:\Users\[User]\Desktop\Backround Immages"
```

Use a simple center crop instead of smart cropping:

```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort hot --crop-mode center --output "C:\Users\[User]\Desktop\Backround Immages"
```

## CLI Options

- `--subreddit` (required): Subreddit name to scrape
- `--limit` (default `50`): Number of final images to save
- `--sort` (default `hot`): One of `hot`, `new`, or `top`
- `--time` (default `all`): Used with `--sort top`; one of `hour`, `day`, `week`, `month`, `year`, or `all`
- `--crop-mode` (default `smart`): `smart` uses saliency and face detection, `center` uses a plain centered crop
- `--output` (default `./output_images`): Destination folder for saved images

## Notes

- `--limit` refers to the number of saved images, not the number of posts fetched.
- The script will keep paging Reddit until it reaches the target number of saved images or runs out of usable posts.
- Filenames are sanitized for Windows and include the post id to reduce collisions.
- If face detection is unavailable in the environment, the smart crop still falls back to saliency-based framing.
- The script creates the output folder automatically if it does not exist.
