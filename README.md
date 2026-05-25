# Reddit Backround Downloader

A small Python CLI tool that downloads and crops high-resolution Reddit images to a 16:9 wallpaper-friendly format.

# Reddit Background Downloader

Tiny CLI to fetch high-resolution images from a subreddit, center-crop them to 16:9, and save ready-to-use wallpaper images.
- Resolves common Reddit-hosted image URLs
**Key features**
- Skips images below the minimum resolution guardrail
- Uses Reddit's public JSON endpoint (no API credentials required)
- Resolves common Reddit image hosts (i.redd.it, preview.redd.it, imgur)
- Center-crops images to 16:9 and enforces a minimum resolution
- Produces Windows-safe filenames and avoids overwrites
- Prints a concise summary at the end of each run
- Dependencies from `requirements.txt`
**Requirements**
## Install
- Python 3.10+
- Install dependencies:
pip install -r requirements.txt
**Quick start**

```bash
pip install -r requirements.txt
```
**Quick start**

```bash
python scraper.py --subreddit EarthPorn --limit 100 --sort hot --output "C:\\Users\\[User]\\Desktop\\Backround Immages"
```



**CLI options**
- `--limit`: Number of final images to save
- `--subreddit` (required): Subreddit name to scrape
- `--limit` (default 50): Number of final images to save (the script will page Reddit until this many images are successfully saved)
- `--sort` (default `hot`): One of `hot`, `new`, or `top`
- `--time` (used with `--sort top`, default `all`): One of `hour`, `day`, `week`, `month`, `year`, `all`
- `--output` (default `./output_images`): Destination folder for saved images

**Behaviour notes**
- `--limit` refers to the number of saved images, not the number of posts fetched. The tool will fetch pages of posts and skip items that do not resolve to valid images or that fail the crop/size constraints.
- The script enforces a minimum cropped resolution (configurable in `scraper.py`) to ensure wallpaper quality.
- Filenames are sanitized for Windows and include the Reddit post id to avoid collisions.

**Output & summary**
- The script only saves images that survive the direct image lookup, download, and crop checks.
At the end of a run the tool prints a short summary with pages fetched, posts scanned, candidates found, failures, and number saved.
- If the output folder does not exist, it will be created automatically.
**Contributing & License**
- A local `.env` file is not required for the current version.
If you want to publish this repository, add a `LICENSE` (MIT recommended) and initialize a Git repository. Contributions and enhancements (for example, adding `rich` for prettier console output) are welcome.

---

See `scraper.py` for runtime options and implementation details.
