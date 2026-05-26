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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# --- Configuration Constants ---
# Target aspect ratio: 16:9
TARGET_ASPECT_RATIO = 16.0 / 9.0
# Resolution Guardrail: Minimum dimensions to consider an image valid for cropping.
MIN_WIDTH = 2560
MIN_HEIGHT = 1440
# -------------------------------

# Setup logging and rich console
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
console = Console()

REDDIT_IMAGE_HOSTS = {
    "i.redd.it",
    "preview.redd.it",
    "i.imgur.com",
    "imgur.com",
}

FACE_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


# --- Rich Output Helpers ---
def print_welcome():
    """Print welcome message with project info."""
    title = Text("Reddit Background Downloader", style="bold cyan")
    console.print(Panel(title, expand=False, border_style="cyan"))


def print_config(subreddit: str, limit: int, sort: str, time_filter: str, crop_mode: str, output_dir: str):
    """Print configuration table."""
    table = Table(title="⚙️  Configuration", show_header=False, box=None)
    table.add_column(style="cyan", width=15)
    table.add_column(style="white")
    
    table.add_row("Subreddit:", f"r/{subreddit}")
    table.add_row("Target images:", str(limit))
    table.add_row("Sort by:", sort)
    if sort == "top":
        table.add_row("Time range:", time_filter)
    table.add_row("Crop mode:", crop_mode)
    table.add_row("Output:", output_dir)
    
    console.print(table)


def print_summary(counters: dict, elapsed: float):
    """Print formatted summary table."""
    table = Table(title="📊 Summary", show_header=False, box=None)
    table.add_column(style="cyan", width=25)
    table.add_column(style="yellow")
    
    table.add_row("Elapsed time:", f"{elapsed:.1f}s")
    table.add_row("Pages fetched:", str(counters['pages_fetched']))
    table.add_row("Posts seen:", str(counters['posts_seen']))
    table.add_row("Image candidates found:", str(counters['image_candidates']))
    table.add_row("Download failures:", str(counters['download_failures']))
    table.add_row("✅ Successfully saved:", f"[green]{counters['saved']}[/green]")
    table.add_row("⏭️  Skipped (too small):", str(counters['skipped_too_small']))
    table.add_row("⏭️  Skipped (no image):", str(counters['skipped_no_image']))
    table.add_row("❌ Processing errors:", str(counters['processing_errors']))
    
    console.print(table)


def print_status(message: str, status_type: str = "info"):
    """Print status message with icon."""
    if status_type == "success":
        console.print(f"[green]✓[/green] {message}")
    elif status_type == "error":
        console.print(f"[red]✗[/red] {message}")
    elif status_type == "warning":
        console.print(f"[yellow]⚠[/yellow] {message}")
    elif status_type == "info":
        console.print(f"[cyan]ℹ[/cyan] {message}")


def print_error(title: str, message: str):
    """Print error message in a panel."""
    error_panel = Panel(message, title=title, border_style="red", style="red")
    console.print(error_panel)


def print_processing_status(saved_count: int, target_count: int, current_page: int):
    """Print a clean processing status line."""
    percent = (saved_count / target_count) * 100
    bar_length = 30
    filled = int(bar_length * saved_count / target_count)
    bar = "█" * filled + "░" * (bar_length - filled)
    status = f"[cyan]Downloading images...[/cyan] {bar} {saved_count}/{target_count} (Page {current_page})"
    console.print(status)


# Removed PRAW/AsyncPRAW dependencies and .env loading.
# The script now relies on direct HTTP requests to the public JSON endpoint.
# The credential checks and PRAW initialization are replaced by basic logging.
# The new fetch_posts_from_json function handles the public JSON structure.


async def download_image(url: str) -> bytes | None:
    """Downloads an image from a URL using httpx."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=20.0)
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        return None
    except httpx.HTTPError:
        return None
    except Exception:
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
    except Exception:
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

        # Apply Resolution Guardrail
        if crop_w < MIN_WIDTH or crop_h < MIN_HEIGHT:
            return False, "too_small"

        # Perform the crop
        cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]

        if cropped_image.size == 0:
            raise ValueError("Cropping resulted in an empty image.")

        # Save the image
        safe_title = sanitize_filename(post_title)
        id_suffix = f"_{post_id}" if post_id else ""
        filename = f"{safe_title[:50]}{id_suffix}_{int(crop_w)}x{int(crop_h)}.jpg"
        save_path = output_path / filename
        cv2.imwrite(str(save_path), cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return True, "saved"

    except ValueError:
        return False, "format_error"
    except Exception:
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

    saved_images = 0
    after = None
    seen_post_ids: set[str] = set()
    face_cascade = load_face_cascade()

    if face_cascade is None:
        print_status("Face detection unavailable; using saliency-only crop guidance.", "warning")
    else:
        print_status("Face detection enabled for content-aware cropping.", "success")

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

    # Print configuration
    print_config(subreddit_name, limit, sort, top_time, crop_mode, str(output_dir))
    
    console.print("\n[cyan]Processing... this may take a few minutes.[/cyan]\n")
    start_time = time.time()

    while saved_images < limit:
        # Fetch another page of posts until we have enough saved images.
        post_children, after = await fetch_posts_from_json(
            subreddit_name, 100, sort, after, top_time
        )
        counters["pages_fetched"] += 1

        if not post_children:
            print_status("No more posts available from Reddit.", "warning")
            break

        # Process posts without verbose progress bar
        for post_data in post_children:
            post = post_data.get("data", {})
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
                # Print clean save message
                print_status(f"Saved image {saved_images}/{limit}", "success")
            else:
                if reason == "too_small":
                    counters["skipped_too_small"] += 1
                else:
                    counters["processing_errors"] += 1

            if saved_images >= limit:
                break

            if not after:
                print_status("Reached the end of the subreddit listing.", "warning")
                break

    # Summary
    elapsed = time.time() - start_time
    print("\n")
    print_summary(counters, elapsed)


def run_interactive_mode():
    """Run interactive mode to collect user input."""
    try:
        import questionary
    except ImportError:
        print_error("Missing Dependency", "questionary not installed. Run: pip install -r requirements.txt")
        return None

    console.print("[cyan]Answer a few questions to get started:[/cyan]\n")
    
    subreddit = questionary.text(
        "Subreddit name (without r/)?",
        default="EarthPorn"
    ).ask()
    
    if not subreddit:
        print_error("Error", "Subreddit name is required.")
        return None

    limit = questionary.text(
        "How many images to download?",
        default="50"
    ).ask()
    
    try:
        limit = int(limit)
    except ValueError:
        print_error("Error", "Limit must be a number.")
        return None

    sort = questionary.select(
        "Sort posts by?",
        choices=["hot", "new", "top"],
        default="hot"
    ).ask()

    time_filter = "all"
    if sort == "top":
        time_filter = questionary.select(
            "Time range?",
            choices=["hour", "day", "week", "month", "year", "all"],
            default="all"
        ).ask()

    crop_mode = questionary.select(
        "Crop mode?",
        choices=["smart (saliency + face detection)", "center (simple center crop)"],
        default="smart (saliency + face detection)"
    ).ask()
    crop_mode = "smart" if "smart" in crop_mode else "center"

    output = questionary.text(
        "Output folder?",
        default="./output_images"
    ).ask()

    console.print()
    
    return {
        "subreddit": subreddit,
        "limit": limit,
        "sort": sort,
        "time": time_filter,
        "crop_mode": crop_mode,
        "output": output,
    }


async def main(
    subreddit: str,
    limit: int,
    sort: str,
    time_filter: str,
    crop_mode: str,
    output: str,
):
    """Main async function that processes the subreddit."""
    # Setup output directory
    output_path = Path(output)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_error("Error", f"Could not create output directory: {e}")
        return

    try:
        # Process the subreddit
        await process_subreddit(
            subreddit,
            limit,
            sort,
            output_path,
            time_filter,
            crop_mode,
        )
        print_status("Scraping and processing complete!", "success")

    except Exception as e:
        print_error("Critical Error", f"An unexpected error occurred: {e}")
        logger.critical(f"A critical error occurred during the scraping process: {e}")


def cli_main():
    """Synchronous entry point that handles CLI parsing and interactive mode."""
    parser = argparse.ArgumentParser(
        description="Scrape, crop, and save high-resolution images from Reddit using the public JSON endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py --subreddit EarthPorn --limit 20 --sort hot
  python scraper.py --subreddit EarthPorn --limit 20 --sort top --time all
  python scraper.py -i  (interactive mode)
        """
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive mode for guided setup.",
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        help="The name of the subreddit to scrape (e.g., 'EarthPorn').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="The number of images to save (default: 50).",
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
        help="Crop strategy: 'smart' uses saliency/face detection, 'center' uses simple center crop (default: smart).",
    )
    args = parser.parse_args()

    print_welcome()

    # Handle interactive mode
    if args.interactive:
        config = run_interactive_mode()
        if config is None:
            return
        subreddit = config["subreddit"]
        limit = config["limit"]
        sort = config["sort"]
        time_filter = config["time"]
        crop_mode = config["crop_mode"]
        output = config["output"]
    else:
        # Validate required argument
        if not args.subreddit:
            parser.print_help()
            print_error("Missing Argument", "The --subreddit argument is required.")
            return

        subreddit = args.subreddit
        limit = args.limit
        sort = args.sort
        time_filter = args.time
        crop_mode = args.crop_mode
        output = args.output

    # Run the async main function
    asyncio.run(main(subreddit, limit, sort, time_filter, crop_mode, output))


if __name__ == "__main__":
    cli_main()
