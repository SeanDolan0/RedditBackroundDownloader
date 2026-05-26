import asyncio
import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import cv2
import numpy as np
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

FACE_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


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


def get_face_cascade_path() -> str | None:
    """Return the bundled OpenCV face cascade path if it exists."""
    bundled_path = Path(cv2.__file__).resolve().parent / "data" / FACE_CASCADE_FILENAME
    return str(bundled_path) if bundled_path.exists() else None


def load_face_cascade() -> cv2.CascadeClassifier | None:
    """Load the bundled Haar face cascade once for content-aware cropping."""
    cascade_path = get_face_cascade_path()
    if not cascade_path:
        return None

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return None

    return face_cascade


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode downloaded image bytes into an OpenCV image."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data.")
    return image


def calculate_saliency_map(img: np.ndarray) -> np.ndarray:
    """Compute a simple saliency map from gradients and contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return cv2.convertScaleAbs(gradient_magnitude)


def detect_faces(
    img: np.ndarray, face_cascade: cv2.CascadeClassifier
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Detect faces and return a binary mask plus face centers."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    face_mask = np.zeros_like(gray, dtype=np.uint8)
    face_centers = []

    for x, y, w, h in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
        face_centers.append((x + w / 2, y + h / 2))

    return face_mask, face_centers


def calculate_weighted_center_of_mass(
    saliency_map: np.ndarray, face_mask: np.ndarray
) -> tuple[int, int]:
    """Calculate the weighted center of mass from saliency and face masks."""
    combined_weight = cv2.addWeighted(saliency_map, 0.5, face_mask, 1.5, 0.0)

    height, width = combined_weight.shape[:2]
    weight_float = combined_weight.astype(np.float32)
    weighted_sum_x = np.sum(np.arange(width)[None, :] * weight_float)
    weighted_sum_y = np.sum(np.arange(height)[:, None] * weight_float)
    total_weight = np.sum(weight_float)

    if total_weight == 0:
        return width // 2, height // 2

    center_x = int(weighted_sum_x / total_weight)
    center_y = int(weighted_sum_y / total_weight)
    return center_x, center_y


def calculate_best_crop_box(
    img: np.ndarray,
    target_aspect_ratio: float,
    face_cascade: cv2.CascadeClassifier | None = None,
) -> tuple[int, int, int, int, int]:
    """Find a content-aware crop box that preserves the target aspect ratio."""
    original_height, original_width = img.shape[:2]

    saliency_map = calculate_saliency_map(img)

    if face_cascade is None:
        face_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        face_centers: list[tuple[float, float]] = []
    else:
        face_mask, face_centers = detect_faces(img, face_cascade)

    center_x, center_y = calculate_weighted_center_of_mass(saliency_map, face_mask)

    if original_width / original_height > target_aspect_ratio:
        crop_h = original_height
        crop_w = max(1, int(original_height * target_aspect_ratio))
    else:
        crop_w = original_width
        crop_h = max(1, int(original_width / target_aspect_ratio))

    start_x = center_x - crop_w // 2
    start_y = center_y - crop_h // 2

    start_x = max(0, min(start_x, original_width - crop_w))
    start_y = max(0, min(start_y, original_height - crop_h))

    return start_x, start_y, crop_w, crop_h, len(face_centers)


def calculate_center_crop_box(
    img: np.ndarray, target_aspect_ratio: float
) -> tuple[int, int, int, int]:
    """Calculate a simple center crop box for comparison or fallback."""
    original_height, original_width = img.shape[:2]

    if original_width / original_height > target_aspect_ratio:
        crop_h = original_height
        crop_w = max(1, int(original_height * target_aspect_ratio))
    else:
        crop_w = original_width
        crop_h = max(1, int(original_width / target_aspect_ratio))

    start_x = max(0, (original_width - crop_w) // 2)
    start_y = max(0, (original_height - crop_h) // 2)
    return start_x, start_y, crop_w, crop_h


def crop_and_save_image(
    image_bytes: bytes,
    output_path: Path,
    post_title: str,
    post_id: str | None = None,
    face_cascade: cv2.CascadeClassifier | None = None,
    crop_mode: str = "smart",
) -> tuple[bool, str]:
    """
    Processes the downloaded image with content-aware cropping for wallpaper use.
    """
    try:
        image = decode_image_bytes(image_bytes)
        face_count = 0

        if crop_mode == "center":
            start_x, start_y, crop_w, crop_h = calculate_center_crop_box(
                image, TARGET_ASPECT_RATIO
            )
        else:
            start_x, start_y, crop_w, crop_h, face_count = calculate_best_crop_box(
                image, TARGET_ASPECT_RATIO, face_cascade
            )

        # 3. Apply Resolution Guardrail
        if crop_w < MIN_WIDTH or crop_h < MIN_HEIGHT:
            logger.debug(
                f"SKIP: Cropped dimensions ({int(crop_w)}x{int(crop_h)}) are below the minimum resolution guardrail ({MIN_WIDTH}x{MIN_HEIGHT})."
            )
            return False, "too_small"

        # 4. Perform the crop
        cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]

        if cropped_image.size == 0:
            raise ValueError("Cropping resulted in an empty image.")

        # 5. Save the image
        safe_title = sanitize_filename(post_title)
        id_suffix = f"_{post_id}" if post_id else ""
        filename = f"{safe_title[:50]}{id_suffix}_{int(crop_w)}x{int(crop_h)}.jpg"
        save_path = output_path / filename
        cv2.imwrite(str(save_path), cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        logger.debug(
            f"SUCCESS: Saved image to {save_path} (faces_detected={face_count})"
        )
        return True, "saved"

    except ValueError as e:
        logger.error(f"Processing error (likely bad image format): {e}")
        return False, "format_error"
    except Exception as e:
        logger.error(f"An unexpected error occurred during image processing: {e}")
        return False, "error"


async def process_subreddit(
    subreddit_name: str,
    limit: int,
    sort: str,
    output_dir: Path,
    top_time: str = "all",
    crop_mode: str = "smart",
):
    """Fetches posts, filters for images, processes, and saves them."""

    logger.info(f"Starting scraper for r/{subreddit_name}...")

    saved_images = 0
    after = None
    seen_post_ids: set[str] = set()
    face_cascade = load_face_cascade()

    if face_cascade is None:
        logger.info("Face detection unavailable; using saliency-only crop guidance.")
    else:
        logger.info("Face detection enabled for content-aware cropping.")

    # Counters for summary
    counters = {
        "pages_fetched": 0,
        "posts_seen": 0,
        "image_candidates": 0,
        "download_failures": 0,
        "saved": 0,
        "skipped_too_small": 0,
        "skipped_no_image": 0,
        "processing_errors": 0,
    }

    # Header
    header = (
        f"=== Reddit Downloader: r/{subreddit_name}  sort={sort}"
        + (f" time={top_time}" if sort == "top" else "")
        + f"  target={limit} images ==="
    )
    print("\n" + header)
    print("Starting... this may take a few minutes.\n")
    start_time = time.time()

    while saved_images < limit:
        # Fetch another page of posts until we have enough saved images.
        post_children, after = await fetch_posts_from_json(
            subreddit_name, 100, sort, after, top_time
        )
        counters["pages_fetched"] += 1

        if not post_children:
            logger.info("No more posts available from Reddit.")
            break

        for post_data in tqdm(post_children, desc="Processing posts"):
            post = post_data.get("data", {})  # 'data' holds the actual post object
            counters["posts_seen"] += 1
            post_id = post.get("id")
            if post_id and post_id in seen_post_ids:
                continue
            if post_id:
                seen_post_ids.add(post_id)

            # Filter: Resolve a direct image URL from common Reddit image fields.
            image_url = resolve_image_url(post)
            if not image_url:
                counters["skipped_no_image"] += 1
                continue
            counters["image_candidates"] += 1

            post_title = post.get("title", "UntitledPost")
            post_id = post.get("id")

            # Download the image bytes
            image_bytes = await download_image(image_url)
            if image_bytes is None:
                counters["download_failures"] += 1
                continue

            # Process and save the image
            ok, reason = crop_and_save_image(
                image_bytes,
                output_dir,
                post_title,
                post_id,
                face_cascade,
                crop_mode,
            )
            if ok and reason == "saved":
                saved_images += 1
                counters["saved"] += 1
                logger.info(f"Saved {saved_images}/{limit} images.")
            else:
                if reason == "too_small":
                    counters["skipped_too_small"] += 1
                else:
                    counters["processing_errors"] += 1

            if saved_images >= limit:
                break

            if not after:
                logger.info("Reached the end of the subreddit listing.")
                break

    # Summary (printed once after processing finishes)
    elapsed = time.time() - start_time
    print("\n=== Summary ===")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Pages fetched: {counters['pages_fetched']}")
    print(f"Posts seen: {counters['posts_seen']}")
    print(f"Image candidates: {counters['image_candidates']}")
    print(f"Downloaded failures: {counters['download_failures']}")
    print(f"Saved: {counters['saved']}")
    print(f"Skipped (too small): {counters['skipped_too_small']}")
    print(f"Skipped (no image): {counters['skipped_no_image']}")
    print(f"Processing errors: {counters['processing_errors']}")
    print("============\n")


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
    parser.add_argument(
        "--crop-mode",
        type=str,
        default="smart",
        choices=["smart", "center"],
        help="Crop strategy to use: smart saliency-based framing or center crop.",
    )
    args = parser.parse_args()

    # 1. Setup output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_path.resolve()}")

    try:
        # 2. Process the subreddit
        await process_subreddit(
            args.subreddit,
            args.limit,
            args.sort,
            output_path,
            args.time,
            args.crop_mode,
        )
        logger.info("--- Scraping and processing complete! ---")

    except Exception as e:
        logger.critical(f"A critical error occurred during the scraping process: {e}")


if __name__ == "__main__":
    asyncio.run(main())
