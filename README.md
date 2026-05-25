# Reddit Backround Downloader

A small Python CLI tool that downloads and crops high-resolution Reddit images to a 16:9 wallpaper-friendly format.

## Features

- Fetches posts from a subreddit using Reddit's public JSON endpoint
- Resolves common Reddit-hosted image URLs
- Crops images to 16:9
- Skips images below the minimum resolution guardrail
- Saves files with Windows-safe filenames

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scraper.py --subreddit EarthPorn --limit 100 --sort hot --output "C:\\Users\\[User]]\\Desktop\\BackroundImmages"
```

For top listings, you can also choose a time window:

```bash
python scraper.py --subreddit EarthPorn --limit 100 --sort top --time all --output "C:\\Users\\[User]\\Desktop\\BackroundImmages"
```

## Arguments

- `--subreddit`: Subreddit name to scrape
- `--limit`: Number of final images to save
- `--sort`: One of `hot`, `new`, or `top`
- `--time`: Time window for `top` results: `hour`, `day`, `week`, `month`, `year`, or `all`
- `--output`: Destination folder for saved images

## Notes

- The script only saves images that survive the direct image lookup, download, and crop checks.
- If the output folder does not exist, it will be created automatically.
- A local `.env` file is not required for the current version.
