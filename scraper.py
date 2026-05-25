import asyncio
import io
import argparse
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image
from tqdm.asyncio import tqdm

# --- Configuration Constants ---
# Target aspect ratio: 16:9
TARGET_ASPECT_RATIO = 16.0 / 9.0
# Resolution Guardrail: Minimum dimensions to consider an image valid for cropping.
MIN_WIDTH = 2560
MIN_HEIGHT = 1440
# -------------------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REDDIT_IMAGE_HOSTS = {
    "i.redd.it",
    "preview.redd.it",
    "i.imgur.com",
    "imgur.com",
}


# Removed PRAW/AsyncPRAW dependencies and .env loading.
# The script now relies on direct HTTP requests to the public JSON endpoint.
# The credential checks and PRAW initialization are replaced by basic logging.
# The new fetch_posts_from_json function handles the public JSON structure.


async def download_image(url: str) -> bytes | None:
    """Downloads an image from a URL using httpx."""
    try:
        # Use httpx with a timeout for robustness
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=20.0)
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as e:
        logger.warning(f"Network error downloading {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading {url}: {e}")
        return None


async def fetch_posts_from_json(
    subreddit_name: str,
    limit: int,
    sort: str,
    after: str | None = None,
    top_time: str = "all",
):
    """Fetches a page of subreddit posts from Reddit's public JSON endpoint."""
    url = f"https://www.reddit.com/r/{subreddit_name}/{sort}.json?limit={limit}"
    if sort == "top":
        url = f"{url}&t={top_time}"
    if after:
        url = f"{url}&after={after}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RedditDownloader/1.0)"}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=20.0), headers=headers
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", {})
            return data.get("children", []), data.get("after")
    except httpx.HTTPError as e:
        logger.warning(f"Network error fetching posts from r/{subreddit_name}: {e}")
        return [], None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while fetching posts from r/{subreddit_name}: {e}"
        )
        return [], None


def resolve_image_url(post: dict) -> str | None:
    """Find the best direct image URL for a Reddit post."""
    candidates = [
        post.get("url_overridden_by_dest"),
        post.get("url"),
    ]

    preview = post.get("preview", {})
    preview_images = preview.get("images", [])
    if preview_images:
        source = preview_images[0].get("source", {})
        candidates.append(source.get("url"))

    for raw_url in candidates:
        if not raw_url:
            continue

        image_url = raw_url.replace("&amp;", "&")
        parsed_url = urlparse(image_url)
        host = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            return image_url

        if host in REDDIT_IMAGE_HOSTS and parsed_url.scheme in {"http", "https"}:
            return image_url

    if post.get("post_hint") == "image":
        for raw_url in candidates:
            if raw_url:
                return raw_url.replace("&amp;", "&")

    return None


def sanitize_filename(text: str, max_length: int = 80) -> str:
    """Convert arbitrary text into a Windows-safe filename stem."""
    cleaned_text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    cleaned_text = re.sub(r"\s+", "_", cleaned_text)
    cleaned_text = re.sub(r"_+", "_", cleaned_text).strip("._ ")
    return cleaned_text[:max_length] or "untitled"


def crop_and_save_image(
    image_bytes: bytes, output_path: Path, post_title: str, post_id: str | None = None
) -> bool:
    """
    Processes the downloaded image: centers and crops it to 16:9 while enforcing
    a minimum resolution guardrail.
    """
    try:
        # 1. Open image and get dimensions
        image = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = image.size

        # 2. Determine cropping strategy
        target_ratio = TARGET_ASPECT_RATIO

        if original_width / original_height > target_ratio:
            # Image is wider than 16:9 (landscape). Height is limiting.
            new_height = original_height
            new_width = int(original_height * target_ratio)

            # Calculate crop box (left, top, right, bottom)
            left = (original_width - new_width) / 2
            top = 0
            right = left + new_width
            bottom = original_height

        else:
            # Image is taller or perfect 16:9. Width is limiting.
            new_width = original_width
            new_height = int(original_width / target_ratio)

            # Calculate crop box
            left = 0
            top = (original_height - new_height) / 2
            right = original_width
            bottom = top + new_height

        # 3. Apply Resolution Guardrail
        if new_width < MIN_WIDTH or new_height < MIN_HEIGHT:
            logger.info(
                f"SKIP: Cropped dimensions ({int(new_width)}x{int(new_height)}) are below the minimum resolution guardrail ({MIN_WIDTH}x{MIN_HEIGHT})."
            )
            return False

        # 4. Perform the crop
        cropped_image = image.crop((left, top, right, bottom))

        # 5. Save the image
        safe_title = sanitize_filename(post_title)
        id_suffix = f"_{post_id}" if post_id else ""
        filename = (
            f"{safe_title[:50]}{id_suffix}_{int(new_width)}x{int(new_height)}.jpg"
        )
        save_path = output_path / filename
        cropped_image.save(save_path)
        logger.info(f"SUCCESS: Saved image to {save_path}")
        return True

    except ValueError as e:
        logger.error(f"Processing error (likely bad image format): {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during image processing: {e}")
        return False


async def process_subreddit(
    subreddit_name: str,
    limit: int,
    sort: str,
    output_dir: Path,
    top_time: str = "all",
):
    """Fetches posts, filters for images, processes, and saves them."""

    logger.info(f"Starting scraper for r/{subreddit_name}...")

    saved_images = 0
    after = None
    seen_post_ids: set[str] = set()

    while saved_images < limit:
        # Fetch another page of posts until we have enough saved images.
        post_children, after = await fetch_posts_from_json(
            subreddit_name, 100, sort, after, top_time
        )

        if not post_children:
            logger.info("No more posts available from Reddit.")
            break

        for post_data in tqdm(post_children, desc="Processing posts"):
            post = post_data.get("data", {})  # 'data' holds the actual post object
            post_id = post.get("id")
            if post_id and post_id in seen_post_ids:
                continue
            if post_id:
                seen_post_ids.add(post_id)

            # Filter: Resolve a direct image URL from common Reddit image fields.
            image_url = resolve_image_url(post)
            if not image_url:
                continue

            post_title = post.get("title", "UntitledPost")
            post_id = post.get("id")

            # Download the image bytes
            image_bytes = await download_image(image_url)
            if image_bytes is None:
                continue

            # Process and save the image
            if crop_and_save_image(image_bytes, output_dir, post_title, post_id):
                saved_images += 1
                logger.info(f"Saved {saved_images}/{limit} images.")

            if saved_images >= limit:
                break

        if not after:
            logger.info("Reached the end of the subreddit listing.")
            break


async def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Scrape, center-crop, and save high-resolution images from a Reddit subreddit using the public JSON endpoint."
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        required=True,
        help="The name of the subreddit to scrape (e.g., 'reactjs').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="The number of posts to fetch (default: 50).",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="hot",
        choices=["hot", "new", "top"],
        help="The sorting method for posts (default: hot).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="all",
        choices=["hour", "day", "week", "month", "year", "all"],
        help="The time range used when --sort top is selected (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_images",
        help="The destination folder for saved images (default: ./output_images).",
    )
    args = parser.parse_args()

    # 1. Setup output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_path.resolve()}")

    try:
        # 2. Process the subreddit
        await process_subreddit(
            args.subreddit, args.limit, args.sort, output_path, args.time
        )
        logger.info("--- Scraping and processing complete! ---")

    except Exception as e:
        logger.critical(f"A critical error occurred during the scraping process: {e}")


if __name__ == "__main__":
    asyncio.run(main())
