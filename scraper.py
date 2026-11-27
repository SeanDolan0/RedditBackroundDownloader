import asyncio
import argparse
import json
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
    table.add_row("⏭️  Skipped (blurry):", str(counters.get('skipped_blurry', 0)))
    table.add_row("⏭️  Skipped (existing):", str(counters.get('skipped_existing', 0)))
    table.add_row("⏭️  Skipped (duplicate):", str(counters.get('duplicates_skipped', 0)))
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



# Removed PRAW/AsyncPRAW dependencies and .env loading.
# The script now relies on direct HTTP requests to the public JSON endpoint.
# The credential checks and PRAW initialization are replaced by basic logging.
# The new fetch_posts_from_json function handles the public JSON structure.


# Shared HTTP settings for Reddit + image downloads.
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RedditDownloader/1.0)"}
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=20.0)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds; exponential


def _retry_after_seconds(response: httpx.Response, fallback: float) -> float:
    """Parse Retry-After header (seconds or HTTP-date) if present, else fallback."""
    raw = response.headers.get("retry-after")
    if not raw:
        return fallback
    try:
        return max(0.0, float(raw))
    except ValueError:
        # HTTP-date form; we can't cheaply diff it, so just back off.
        return fallback


async def download_image(
    url: str,
    client: httpx.AsyncClient | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes | None:
    """Download an image, retrying transient failures with backoff.

    Returns image bytes, or None on permanent failure. Uses a shared `client`
    when provided; otherwise opens a short-lived one.
    """
    async def _attempt(c: httpx.AsyncClient) -> bytes | None:
        for attempt in range(max_retries + 1):
            try:
                response = await c.get(url)
            except httpx.HTTPError:
                if attempt < max_retries:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                return None
            except Exception:
                return None

            if response.status_code in RETRYABLE_STATUS:
                if attempt < max_retries:
                    delay = _retry_after_seconds(response, RETRY_BACKOFF_BASE * (2 ** attempt))
                    await asyncio.sleep(delay)
                    continue
                return None

            try:
                response.raise_for_status()
                return response.content
            except httpx.HTTPError:
                return None
        return None

    if client is not None:
        return await _attempt(client)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS) as owned:
        return await _attempt(owned)


async def fetch_posts_from_json(
    subreddit_name: str,
    limit: int,
    sort: str,
    after: str | None = None,
    top_time: str = "all",
    client: httpx.AsyncClient | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[list, str | None, bool]:
    """Fetch a page of posts from Reddit's public JSON endpoint.

    Returns (children, after, ok). `ok` is False when the request ultimately
    failed after retries, so callers can tell a real error apart from a
    genuine "no more posts" (empty `children` with ok=True).
    """
    url = f"https://www.reddit.com/r/{subreddit_name}/{sort}.json?limit={limit}"
    if sort == "top":
        url = f"{url}&t={top_time}"
    if after:
        url = f"{url}&after={after}"

    async def _attempt(c: httpx.AsyncClient) -> tuple[list, str | None, bool]:
        for attempt in range(max_retries + 1):
            try:
                response = await c.get(url)
            except httpx.HTTPError:
                if attempt < max_retries:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                return [], None, False
            except Exception:
                return [], None, False

            if response.status_code in RETRYABLE_STATUS:
                if attempt < max_retries:
                    delay = _retry_after_seconds(response, RETRY_BACKOFF_BASE * (2 ** attempt))
                    await asyncio.sleep(delay)
                    continue
                return [], None, False

            try:
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", {})
                return data.get("children", []), data.get("after"), True
            except httpx.HTTPError:
                return [], None, False
            except Exception:
                return [], None, False
        return [], None, False

    if client is not None:
        return await _attempt(client)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS) as owned:
        return await _attempt(owned)


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

        # Skip animated/non-photo GIFs: wasted download + can't decode reliably.
        if path.endswith(".gif"):
            continue

        if path.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return image_url

        if host in REDDIT_IMAGE_HOSTS and parsed_url.scheme in {"http", "https"}:
            return image_url

    if post.get("post_hint") == "image":
        for raw_url in candidates:
            if raw_url:
                image_url = raw_url.replace("&amp;", "&")
                if urlparse(image_url).path.lower().endswith(".gif"):
                    continue
                return image_url

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
    """Compute a smoothed edge saliency map as float32 in [0, 1].

    Blurs first so meaningful edges win over noise, then normalizes
    gradient magnitude to [0, 1].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    if magnitude.max() == 0.0:
        return np.zeros(magnitude.shape, dtype=np.float32)
    return magnitude / magnitude.max()


def integral_image(values: np.ndarray) -> np.ndarray:
    """2D integral image for O(1) rectangle-sum queries."""
    height, width = values.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    # Accumulate in float64: np.cumsum preserves float32 and the table reaches ~W*H.
    integral[1:, 1:] = np.cumsum(np.cumsum(values, axis=0, dtype=np.float64), axis=1, dtype=np.float64)
    return integral


def box_mean_saliency(
    integral: np.ndarray, start_x: int, start_y: int, width: int, height: int
) -> float:
    """Mean saliency inside a box, via the integral image."""
    total = (
        integral[start_y + height, start_x + width]
        - integral[start_y, start_x + width]
        - integral[start_y + height, start_x]
        + integral[start_y, start_x]
    )
    return total / (width * height)


def detect_faces(
    img: np.ndarray, face_cascade: cv2.CascadeClassifier
) -> list[tuple[int, int, int, int]]:
    """Detect faces and return (x, y, w, h) boxes in original image coordinates.

    Detection runs on a downscaled copy (long side <= 1024px) for speed;
    boxes are scaled back to the original image.
    """
    original_height, original_width = img.shape[:2]
    long_side = max(original_width, original_height)
    scale = min(1.0, 1024.0 / long_side)
    # Clamp to >=1px: extreme-aspect slivers could otherwise round a side to 0.
    resized = img if scale == 1.0 else cv2.resize(
        img,
        (
            max(1, round(original_width * scale)),
            max(1, round(original_height * scale)),
        ),
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    min_size = max(int(30 * scale), 10)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    boxes = []
    for x, y, w, h in faces:
        boxes.append(
            (round(x / scale), round(y / scale), round(w / scale), round(h / scale))
        )
    return boxes


def build_crop_box(
    img: np.ndarray, target_aspect_ratio: float
) -> tuple[int, int, str]:
    """Compute the crop box of `target_aspect_ratio` filling the short side.

    Returns (crop_w, crop_h, free_axis); free_axis is "x" when the image is
    wider than the target (full height fills the box), otherwise "y".

    Crop size uses integer floor (matches original behavior); in a wider-than-
    target image we clamp to original_width-1 so there is always at least 1px
    of horizontal slack and the output never becomes the uncropped original.
    """
    original_height, original_width = img.shape[:2]
    if original_width / original_height > target_aspect_ratio:
        crop_h = original_height
        crop_w = max(1, min(int(original_height * target_aspect_ratio), original_width - 1))
        free_axis = "x"
    else:
        crop_w = original_width
        crop_h = max(1, min(int(original_width / target_aspect_ratio), original_height))
        free_axis = "y"
    return crop_w, crop_h, free_axis


def calculate_center_crop_box(
    img: np.ndarray, target_aspect_ratio: float
) -> tuple[int, int, int, int]:
    """Calculate a simple center crop box."""
    original_height, original_width = img.shape[:2]
    crop_w, crop_h, _ = build_crop_box(img, target_aspect_ratio)
    start_x = (original_width - crop_w) // 2
    start_y = (original_height - crop_h) // 2
    return start_x, start_y, crop_w, crop_h


def calculate_best_crop_box(
    img: np.ndarray,
    target_aspect_ratio: float,
    face_cascade: cv2.CascadeClassifier | None = None,
    face_weight: float = 1.0,
) -> tuple[int, int, int, int]:
    """Find a content-aware crop box that preserves the target aspect ratio.

    The window slides along its free axis; each candidate position is scored
    (mean saliency inside the window, plus face rewards) and the best wins.
    The center candidate is seeded first, so ties favor the center.

    `face_weight` scales all face rewards/penalties; 0.0 disables face-aware
    framing (use for pure-landscape scenery), and `face_cascade=None` too.
    """
    original_height, original_width = img.shape[:2]
    crop_w, crop_h, free_axis = build_crop_box(img, target_aspect_ratio)

    saliency = calculate_saliency_map(img)
    integral = integral_image(saliency)
    use_faces = face_cascade is not None and face_weight != 0.0
    face_boxes = detect_faces(img, face_cascade) if use_faces else []
    margin = 0.05 * min(crop_w, crop_h)

    def score_box(start_x: int, start_y: int) -> float:
        score = box_mean_saliency(integral, start_x, start_y, crop_w, crop_h)
        for face_x, face_y, face_w, face_h in face_boxes:
            face_right = face_x + face_w
            face_bottom = face_y + face_h
            in_frame = (
                face_x >= start_x
                and face_right <= start_x + crop_w
                and face_y >= start_y
                and face_bottom <= start_y + crop_h
            )
            if in_frame:
                within_margin = (
                    face_x >= start_x + margin
                    and face_y >= start_y + margin
                    and face_right <= start_x + crop_w - margin
                    and face_bottom <= start_y + crop_h - margin
                )
                score += (1.5 if within_margin else 0.75) * face_weight
            else:
                partially_clipped = (
                    face_x < start_x + crop_w
                    and face_right > start_x
                    and face_y < start_y + crop_h
                    and face_bottom > start_y
                )
                score -= (10.0 if partially_clipped else 2.0) * face_weight
        return score

    if free_axis == "x":
        candidates = [(start_x, 0) for start_x in range(0, original_width - crop_w + 1)]
        ideal = ((original_width - crop_w) // 2, 0)
    else:
        candidates = [(0, start_y) for start_y in range(0, original_height - crop_h + 1)]
        ideal = (0, (original_height - crop_h) // 2)

    best_start = ideal
    best_score = score_box(*ideal)
    for start_x, start_y in candidates:
        if (start_x, start_y) == ideal:
            continue
        score = score_box(start_x, start_y)
        if score > best_score:
            best_score = score
            best_start = (start_x, start_y)

    start_x, start_y = best_start
    return start_x, start_y, crop_w, crop_h


def crop_and_save_image(
    image_bytes: bytes,
    output_path: Path,
    post_title: str,
    post_id: str | None = None,
    face_cascade: cv2.CascadeClassifier | None = None,
    crop_mode: str = "smart",
    *,
    face_weight: float = 1.0,
    target_resolution: tuple[int, int] | None = None,
    allow_upscale: bool = False,
    min_sharpness: float = 0.0,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
    skip_existing: bool = False,
    seen_hashes: set[int] | None = None,
    metadata: dict | None = None,
    write_metadata: bool = False,
    jpeg_quality: int = 95,
) -> tuple[bool, str]:
    """Process downloaded image with content-aware cropping for wallpaper use.

    crop_mode: "smart" (saliency + optional face-aware framing),
        "center" (centered crop), or "landscape" (smart saliency, NO face
        pulling - best for pure scenery).

    Returns (ok, reason); reason is one of:
      saved | too_small | too_blurry | exists | duplicate | format_error | error.
    """
    try:
        image = decode_image_bytes(image_bytes)

        # Choose crop aspect. With --target-resolution, crop to match it.
        target_aspect_ratio = TARGET_ASPECT_RATIO
        out_size: tuple[int, int] | None = None
        if target_resolution is not None:
            out_w_req, out_h_req = target_resolution
            if out_w_req <= 0 or out_h_req <= 0:
                return False, "error"
            target_aspect_ratio = out_w_req / out_h_req
            out_size = (out_w_req, out_h_req)

        use_face = face_cascade if crop_mode == "smart" and face_weight != 0.0 else None
        effective_face_weight = face_weight if crop_mode == "smart" else 0.0

        if crop_mode == "center":
            start_x, start_y, crop_w, crop_h = calculate_center_crop_box(
                image, target_aspect_ratio
            )
        else:
            # smart + landscape both use the saliency slider; landscape disables faces.
            start_x, start_y, crop_w, crop_h = calculate_best_crop_box(
                image, target_aspect_ratio, use_face, effective_face_weight
            )

        # Resolution guardrail on the SOURCE crop (image quality).
        if crop_w < min_width or crop_h < min_height:
            return False, "too_small"

        # Perform the crop.
        cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
        if cropped_image.size == 0:
            raise ValueError("Cropping resulted in an empty image.")

        # Sharpness filter on the actual crop (reject blurry wallpapers).
        if min_sharpness > 0.0:
            sharpness = _sharpness(cropped_image)
            if sharpness < min_sharpness:
                return False, "too_blurry"

        # Output sizing.
        final_image = cropped_image
        if out_size is not None:
            out_w, out_h = out_size
            # No-upscale policy: refuse to invent pixels unless allowed.
            if not allow_upscale and (crop_w < out_w or crop_h < out_h):
                return False, "too_small"
            downscaling = out_w <= crop_w and out_h <= crop_h
            interp = cv2.INTER_AREA if downscaling else cv2.INTER_LANCZOS4
            final_image = cv2.resize(final_image, (out_w, out_h), interpolation=interp)

        out_h_final, out_w_final = final_image.shape[:2]

        # Output name.
        safe_title = sanitize_filename(post_title)
        id_suffix = f"_{post_id}" if post_id else ""
        filename = f"{safe_title[:50]}{id_suffix}_{out_w_final}x{out_h_final}.jpg"
        save_path = output_path / filename

        # Skip existing outputs (--skip-existing); also re-check just before write.
        if skip_existing and save_path.exists():
            return False, "exists"

        # Perceptual near-duplicate filter (mutually checked before writing).
        img_hash = image_dhash(final_image)
        if seen_hashes is not None and img_hash in seen_hashes:
            return False, "duplicate"

        if not cv2.imwrite(
            str(save_path), final_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ):
            return False, "error"

        if seen_hashes is not None:
            seen_hashes.add(img_hash)

        # Optional attribution / source sidecar.
        if write_metadata:
            sidecar = {
                "filename": filename,
                "title": post_title,
                "post_id": post_id,
                "crop_mode": crop_mode,
                "width": out_w_final,
                "height": out_h_final,
                "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": metadata or {},
            }
            sidecar_path = save_path.with_name(save_path.name + ".json")
            try:
                sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
            except Exception:
                logger.warning("Could not write metadata sidecar for %s", filename)

        return True, "saved"

    except ValueError:
        return False, "format_error"
    except Exception:
        return False, "error"


def _sharpness(img: np.ndarray) -> float:
    """Variance of Laplacian on grayscale; higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def image_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """Perceptual difference-hash for near-duplicate detection.

    Downscale to (hash_size+1)x(hash_size) grayscale; each bit records whether
    a pixel is brighter than its right neighbor. Returns a hash_size**2-bit int.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


async def process_subreddit(
    subreddit_name: str,
    limit: int,
    sort: str,
    output_dir: Path,
    top_time: str = "all",
    crop_mode: str = "smart",
    *,
    concurrency: int = 4,
    face_weight: float = 1.0,
    target_resolution: tuple[int, int] | None = None,
    allow_upscale: bool = False,
    min_sharpness: float = 0.0,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
    skip_existing: bool = False,
    no_duplicates: bool = False,
    write_metadata: bool = False,
    allow_nsfw: bool = False,
    jpeg_quality: int = 95,
):
    """Fetch posts, filter images, download concurrently, process, and save.

    Downloads run concurrently with a bounded semaphore on ONE shared HTTP
    client; image processing + saving stays sequential (original order) so
    `limit`, paging, `--skip-existing`, and dedup remain correct.
    """
    saved_images = 0
    after = None
    seen_post_ids: set[str] = set()
    seen_hashes: set[int] | None = set() if no_duplicates else None
    face_cascade = load_face_cascade()

    # Face-aware framing only matters in "smart" mode; "landscape" disables it.
    if crop_mode == "landscape":
        print_status("Landscape mode: saliency-only cropping, face-pulling disabled.", "info")
    elif face_cascade is None:
        print_status("Face detection unavailable; using saliency-only crop guidance.", "warning")
    else:
        print_status("Face detection enabled for content-aware cropping.", "success")

    counters = {
        "pages_fetched": 0,
        "posts_seen": 0,
        "image_candidates": 0,
        "download_failures": 0,
        "saved": 0,
        "skipped_too_small": 0,
        "skipped_blurry": 0,
        "skipped_existing": 0,
        "duplicates_skipped": 0,
        "skipped_no_image": 0,
        "processing_errors": 0,
    }

    print_config(subreddit_name, limit, sort, top_time, crop_mode, str(output_dir))
    console.print("\n[cyan]Processing... this may take a few minutes.[/cyan]\n")
    start_time = time.time()

    concurrency = max(1, concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(
        max_connections=max(8, concurrency * 2),
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, limits=limits
    ) as client:

        async def download_bounded(image_url: str) -> bytes | None:
            async with semaphore:
                try:
                    return await download_image(image_url, client=client)
                except Exception:
                    return None

        while saved_images < limit:
            post_children, after, ok = await fetch_posts_from_json(
                subreddit_name, 100, sort, after, top_time, client=client
            )
            counters["pages_fetched"] += 1

            if not ok:
                # Request failed after retries: NOT silently "end of list".
                print_status(
                    "Reddit request failed after retries; stopping (not treating as end-of-list).",
                    "error",
                )
                break

            if not post_children:
                print_status("No more posts available from Reddit.", "warning")
                break

            # Build this page's candidates sequentially (keeps counters + id dedup simple).
            page_candidates: list[tuple] = []
            for post_data in post_children:
                post = post_data.get("data", {})
                counters["posts_seen"] += 1
                post_id = post.get("id")
                if post_id and post_id in seen_post_ids:
                    continue
                if post_id:
                    seen_post_ids.add(post_id)

                # Safety + noise filters.
                if post.get("is_self") or post.get("stickied"):
                    continue
                if not allow_nsfw and (post.get("over_18") or post.get("spoiler")):
                    continue

                image_url = resolve_image_url(post)
                if not image_url:
                    counters["skipped_no_image"] += 1
                    continue
                counters["image_candidates"] += 1

                # --skip-existing: don't even download if this post id already saved.
                if skip_existing and post_id:
                    safe_title = sanitize_filename(post.get("title", "UntitledPost"))[:50]
                    id_suffix = f"_{post_id}"
                    already = any(output_dir.glob(f"*{id_suffix}_*.jpg")) or any(
                        output_dir.glob(f"{safe_title}{id_suffix}_*.jpg")
                    )
                    if already:
                        counters["skipped_existing"] += 1
                        continue

                post_title = post.get("title", "UntitledPost")
                metadata = {
                    "source_url": image_url,
                    "source_post_url": post.get("url"),
                    "permalink": post.get("permalink"),
                    "author": post.get("author"),
                    "subreddit": post.get("subreddit", subreddit_name),
                    "post_id": post_id,
                }
                page_candidates.append((post, post_title, post_id, image_url, metadata))

            # Concurrent bounded downloads on the shared client.
            download_results = await asyncio.gather(
                *(download_bounded(cand[3]) for cand in page_candidates),
                return_exceptions=True,
            )

            # Sequential processing in original order for correctness.
            for (post, post_title, post_id, image_url, metadata), image_bytes in zip(
                page_candidates, download_results
            ):
                if isinstance(image_bytes, BaseException) or image_bytes is None:
                    counters["download_failures"] += 1
                    continue

                ok_img, reason = crop_and_save_image(
                    image_bytes,
                    output_dir,
                    post_title,
                    post_id,
                    face_cascade,
                    crop_mode,
                    face_weight=face_weight,
                    target_resolution=target_resolution,
                    allow_upscale=allow_upscale,
                    min_sharpness=min_sharpness,
                    min_width=min_width,
                    min_height=min_height,
                    skip_existing=skip_existing,
                    seen_hashes=seen_hashes,
                    metadata=metadata,
                    write_metadata=write_metadata,
                    jpeg_quality=jpeg_quality,
                )

                if ok_img and reason == "saved":
                    saved_images += 1
                    counters["saved"] += 1
                    print_status(f"Saved image {saved_images}/{limit}", "success")
                elif reason == "too_small":
                    counters["skipped_too_small"] += 1
                elif reason == "too_blurry":
                    counters["skipped_blurry"] += 1
                elif reason == "exists":
                    counters["skipped_existing"] += 1
                elif reason == "duplicate":
                    counters["duplicates_skipped"] += 1
                else:
                    counters["processing_errors"] += 1

                if saved_images >= limit:
                    break

            if saved_images >= limit:
                break
            if not after:
                print_status("Reached the end of the subreddit listing.", "warning")
                break

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
        choices=[
            "smart (salient regions + faces inside frame)",
            "landscape (salient regions, no face-pulling)",
            "center (simple center crop)",
        ],
        default="smart (salient regions + faces inside frame)"
    ).ask()
    if "landscape" in crop_mode:
        crop_mode = "landscape"
    elif "center" in crop_mode:
        crop_mode = "center"
    else:
        crop_mode = "smart"

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


def parse_resolution(text: str) -> tuple[int, int]:
    """Parse 'WxH' (e.g. '2560x1440') into (width, height)."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid resolution {text!r}; expected WxH e.g. 2560x1440")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive.")
    return width, height


async def main(
    subreddit: str,
    limit: int,
    sort: str,
    time_filter: str,
    crop_mode: str,
    output: str,
    *,
    concurrency: int = 4,
    face_weight: float = 1.0,
    target_resolution: tuple[int, int] | None = None,
    allow_upscale: bool = False,
    min_sharpness: float = 0.0,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
    skip_existing: bool = False,
    no_duplicates: bool = False,
    write_metadata: bool = False,
    allow_nsfw: bool = False,
    jpeg_quality: int = 95,
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
            concurrency=concurrency,
            face_weight=face_weight,
            target_resolution=target_resolution,
            allow_upscale=allow_upscale,
            min_sharpness=min_sharpness,
            min_width=min_width,
            min_height=min_height,
            skip_existing=skip_existing,
            no_duplicates=no_duplicates,
            write_metadata=write_metadata,
            allow_nsfw=allow_nsfw,
            jpeg_quality=jpeg_quality,
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
        choices=["smart", "center", "landscape"],
        help=(
            "Crop strategy: 'smart' scans candidate windows for the most salient placement, "
            "keeping detected faces inside the frame; 'landscape' does the same saliency scan "
            "WITHOUT face-pulling (best for pure scenery); 'center' uses a simple center crop "
            "(default: smart)."
        ),
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=1.0,
        help="Scale face-aware framing rewards/penalties in 'smart' mode; 0 disables face-pulling (default: 1.0).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max simultaneous image downloads on one shared HTTP client (default: 4).",
    )
    parser.add_argument(
        "--target-resolution",
        type=parse_resolution,
        default=None,
        metavar="WxH",
        help="Output exactly WxH (e.g. 2560x1440); crop aspect is matched to it. Default: native-res 16:9 crop.",
    )
    parser.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Allow upscaling source crops to reach --target-resolution (default: no upscaling).",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Reject blurry crops below this Laplacian-variance sharpness; 0 disables (default: 0).",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=MIN_WIDTH,
        help=f"Source-crop minimum width in px (default: {MIN_WIDTH}).",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=MIN_HEIGHT,
        help=f"Source-crop minimum height in px (default: {MIN_HEIGHT}).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip posts already saved in the output folder (match by post id); don't re-download or overwrite.",
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip perceptually near-duplicate images within this run (dHash).",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Write an attribution/source sidecar <name>.jpg.json next to each saved image.",
    )
    parser.add_argument(
        "--nsfw",
        action="store_true",
        help="Allow NSFW/spoiler posts (default: skip over_18 + spoiler).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG output quality 1-100 (default: 95).",
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
    asyncio.run(
        main(
            subreddit,
            limit,
            sort,
            time_filter,
            crop_mode,
            output,
            concurrency=args.concurrency,
            face_weight=args.face_weight,
            target_resolution=args.target_resolution,
            allow_upscale=args.allow_upscale,
            min_sharpness=args.min_sharpness,
            min_width=args.min_width,
            min_height=args.min_height,
            skip_existing=args.skip_existing,
            no_duplicates=args.no_duplicates,
            write_metadata=args.write_metadata,
            allow_nsfw=args.nsfw,
            jpeg_quality=args.jpeg_quality,
        )
    )


if __name__ == "__main__":
    cli_main()
