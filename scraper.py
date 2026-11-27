import asyncio
import argparse
import json
import logging
import re
import sys
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
# Aspect Guardrail: Minimum source width/height ratio (1.0 skips portrait images).
MIN_ASPECT_RATIO = 1.0
# -------------------------------

# Windows consoles default to cp1252; make output UTF-8 so status glyphs
# (checkmarks, emoji) can't crash printing. No-op where already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

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
    table.add_row("⏭️  Skipped (portrait):", str(counters.get('skipped_aspect_ratio', 0)))
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

# Reddit now returns 403 for unauthenticated JSON on www.reddit.com. Anonymous
# visitors are instead issued a `token_v2` cookie that works as a Bearer token
# against oauth.reddit.com. We warm up once to grab it, then send it ONLY on
# Reddit API calls (never forwarded to image hosts).
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
_ANON_TOKEN: str | None = None
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


async def _anonymous_oauth_token(client: httpx.AsyncClient) -> str | None:
    """Return a cached anonymous Bearer token, warming up cookies if needed."""
    global _ANON_TOKEN
    if _ANON_TOKEN:
        return _ANON_TOKEN
    try:
        await client.get("https://old.reddit.com/", follow_redirects=True)
    except Exception:
        return None
    token = client.cookies.get("token_v2")
    if token:
        _ANON_TOKEN = token
        return token
    return None


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

    Uses Reddit's anonymous-OAuth route (`oauth.reddit.com` + `token_v2`
    Bearer), since Reddit now blocks unauthenticated www.reddit.com JSON;
    the request is still attempted if no token could be obtained.
    """
    async def _attempt(c: httpx.AsyncClient) -> tuple[list, str | None, bool]:
        token = await _anonymous_oauth_token(c)
        url = f"{REDDIT_OAUTH_BASE}/r/{subreddit_name}/{sort}?limit={limit}&raw_json=1"
        headers = {"Authorization": f"Bearer {token}"} if token else None
        if sort == "top":
            url = f"{url}&t={top_time}"
        if after:
            url = f"{url}&after={after}"

        for attempt in range(max_retries + 1):
            try:
                response = await c.get(url, headers=headers)
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


SALIENCY_PROXY_MAX_SIDE = 512


def _resize_to_proxy(
    img: np.ndarray, max_side: int = SALIENCY_PROXY_MAX_SIDE
) -> tuple[np.ndarray, float]:
    """Downscale image so its longest side is <= max_side; return (resized, scale)."""
    h, w = img.shape[:2]
    long_side = max(w, h)
    scale = min(1.0, float(max_side) / float(long_side))
    if scale == 1.0:
        return img, 1.0
    pw = max(1, round(w * scale))
    ph = max(1, round(h * scale))
    return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA), scale


def _normalize_robust(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Normalize non-negative array to float32 [0, 1] using a high percentile."""
    max_val = float(arr.max())
    if max_val <= 1e-8:
        return np.zeros(arr.shape, dtype=np.float32)
    ref = float(np.percentile(arr, percentile))
    denom = ref if ref > 1e-8 else max_val
    out = np.clip(arr / denom, 0.0, 1.0).astype(np.float32)
    cur_max = float(out.max())
    if 0.0 < cur_max < 1.0:
        out = np.clip(out / cur_max, 0.0, 1.0).astype(np.float32)
    return out


def _compute_proxy_saliency_and_edges(
    proxy_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (saliency, macro_edges, fine_edges) float32 maps in [0, 1] on a proxy image.

    Combines CIE Lab global color/luminance uniqueness, multi-scale Lab
    center-surround contrast, and texture-suppressed macro structure so
    subjects stand out over repetitive grass, gravel, or water ripples.
    """
    h, w = proxy_bgr.shape[:2]
    short_side = max(1, min(h, w))

    if int(proxy_bgr.max()) == int(proxy_bgr.min()):
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, zeros, zeros

    lab = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sigma_fine_lab = max(1.0, 0.008 * short_side)
    sigma_macro_lab = max(4.0, 0.06 * short_side)
    lab_smooth = cv2.GaussianBlur(
        lab, (0, 0), sigmaX=sigma_fine_lab, sigmaY=sigma_fine_lab
    )
    lab_macro = cv2.GaussianBlur(
        lab, (0, 0), sigmaX=sigma_macro_lab, sigmaY=sigma_macro_lab
    )

    chroma_weights = np.array([1.0, 1.35, 1.35], dtype=np.float32)
    med_lab = np.median(lab_smooth.reshape(-1, 3), axis=0)
    sal_global = np.sqrt(np.sum(((lab_smooth - med_lab) ** 2) * chroma_weights, axis=2))
    sal_surround = np.sqrt(
        np.sum(((lab_smooth - lab_macro) ** 2) * chroma_weights, axis=2)
    )

    gray = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Use BORDER_REPLICATE so 1px-tall sliver inputs never fail Sobel border checks.
    gx_f = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy_f = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    fine_mag = np.sqrt(gx_f ** 2 + gy_f ** 2)
    sigma_density = max(1.5, 0.02 * short_side)
    fine_density = cv2.GaussianBlur(
        fine_mag, (0, 0), sigmaX=sigma_density, sigmaY=sigma_density
    )

    sigma_struct = max(1.5, 0.012 * short_side)
    gray_macro = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=sigma_struct, sigmaY=sigma_struct
    )
    gx_m = cv2.Sobel(gray_macro, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy_m = cv2.Sobel(gray_macro, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    macro_mag = np.sqrt(gx_m ** 2 + gy_m ** 2)

    sal_global_n = _normalize_robust(sal_global)
    sal_surround_n = _normalize_robust(sal_surround)
    macro_n = _normalize_robust(macro_mag)
    fine_n = _normalize_robust(fine_density)

    # Suppress repetitive high-frequency texture without macro contrast (grass, ripples).
    clutter_penalty = np.clip(
        fine_n - 0.85 * macro_n - 0.5 * sal_surround_n, 0.0, 1.0
    )
    structure_n = np.clip(macro_n * (1.0 - 0.65 * clutter_penalty), 0.0, 1.0)

    combined = (0.45 * sal_global_n + 0.35 * sal_surround_n + 0.20 * structure_n) * (
        1.0 - 0.40 * clutter_penalty
    )
    saliency = _normalize_robust(combined)
    return saliency, macro_n, fine_n


def _estimate_sky_mask(proxy_bgr: np.ndarray, fine_n: np.ndarray) -> np.ndarray:
    """Return a soft [0, 1] mask of likely sky regions in the upper scene."""
    h, w = proxy_bgr.shape[:2]
    if h < 16 or int(proxy_bgr.max()) == int(proxy_bgr.min()):
        return np.zeros((h, w), dtype=np.float32)

    hsv = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    val = hsv[:, :, 2] / 255.0
    sat = hsv[:, :, 1] / 255.0
    hue = hsv[:, :, 0]

    lower_val = float(np.median(val[h // 2 :, :]))
    upper_val = float(np.median(val[: max(1, h // 3), :]))
    has_sky_brightness = upper_val > lower_val + 0.06
    is_blue = (hue >= 85.0) & (hue <= 135.0) & (sat >= 0.12) & (val >= 0.25)
    if not has_sky_brightness and float(is_blue[: max(1, h // 3), :].mean()) < 0.15:
        return np.zeros((h, w), dtype=np.float32)

    y_norm = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    upper_prior = np.clip((y_norm - 0.35) / 0.65, 0.0, 1.0)
    smoothness = np.clip(1.0 - 2.0 * fine_n, 0.0, 1.0)
    bright_or_blue = np.maximum(
        np.clip((val - lower_val) / 0.35, 0.0, 1.0),
        is_blue.astype(np.float32),
    )
    return (upper_prior * smoothness * bright_or_blue).astype(np.float32)


def calculate_saliency_map(img: np.ndarray) -> np.ndarray:
    """Compute a scale-invariant multi-cue saliency map as float32 in [0, 1].

    Computes CIE Lab color/luminance uniqueness, center-surround contrast, and
    texture-suppressed macro structure on a <=512px proxy, then upsamples back
    to the input image dimensions.
    """
    orig_h, orig_w = img.shape[:2]
    proxy, scale = _resize_to_proxy(img)
    saliency_small, _, _ = _compute_proxy_saliency_and_edges(proxy)
    if scale == 1.0:
        return saliency_small
    saliency_full = cv2.resize(
        saliency_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    max_val = float(saliency_full.max())
    if max_val <= 1e-8:
        return np.zeros((orig_h, orig_w), dtype=np.float32)
    if max_val < 1.0:
        saliency_full = saliency_full / max_val
    return np.clip(saliency_full, 0.0, 1.0).astype(np.float32)


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


def _weighted_window_saliency(
    sal_int: np.ndarray,
    px: int,
    py: int,
    pw: int,
    ph: int,
    free_axis: str,
) -> float:
    """Mean saliency weighted toward rule-of-thirds and center bands along free_axis."""
    span = ph if free_axis == "y" else pw
    if span < 5:
        return box_mean_saliency(sal_int, px, py, pw, ph)

    b0 = round(0.12 * span)
    b1 = round(0.24 * span)
    b3 = round(0.24 * span)
    b4 = round(0.12 * span)
    b2 = span - (b0 + b1 + b3 + b4)
    bands = ((b0, 0.40), (b1, 1.15), (b2, 1.15), (b3, 1.15), (b4, 0.40))

    weighted_sum = 0.0
    weight_total = 0.0
    offset = 0
    for size, weight in bands:
        if size <= 0:
            continue
        if free_axis == "y":
            band_mean = box_mean_saliency(sal_int, px, py + offset, pw, size)
        else:
            band_mean = box_mean_saliency(sal_int, px + offset, py, size, ph)
        weighted_sum += band_mean * size * weight
        weight_total += size * weight
        offset += size
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def _expand_face_box(
    face_box: tuple[int, int, int, int], img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Expand a tight face box into a head-and-shoulders framing box."""
    fx, fy, fw, fh = face_box
    pad_x = round(0.35 * fw)
    pad_top = round(0.45 * fh)
    pad_bottom = round(0.75 * fh)
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_top)
    x2 = min(img_w, fx + fw + pad_x)
    y2 = min(img_h, fy + fh + pad_bottom)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def calculate_best_crop_box(
    img: np.ndarray,
    target_aspect_ratio: float,
    face_cascade: cv2.CascadeClassifier | None = None,
    face_weight: float = 1.0,
) -> tuple[int, int, int, int]:
    """Find a composition-aware crop box preserving target_aspect_ratio.

    Scores candidate positions along `free_axis` on a <=512px proxy using:
    - Multi-cue Lab color/luminance + texture-suppressed saliency weighted toward
      rule-of-thirds and center bands.
    - Boundary-cut penalty to avoid slicing across salient structures/peaks.
    - Sky/horizon balance when cropping vertically on landscapes.
    - Soft center prior so noisy textures near edges do not hijack the frame.
    - Head-and-shoulders face framing when face detection is enabled.
    """
    original_height, original_width = img.shape[:2]
    crop_w, crop_h, free_axis = build_crop_box(img, target_aspect_ratio)
    ideal = (
        ((original_width - crop_w) // 2, 0)
        if free_axis == "x"
        else (0, (original_height - crop_h) // 2)
    )

    proxy_bgr, scale = _resize_to_proxy(img)
    proxy_h, proxy_w = proxy_bgr.shape[:2]
    sal_small, macro_n, fine_n = _compute_proxy_saliency_and_edges(proxy_bgr)

    use_faces = face_cascade is not None and face_weight != 0.0
    raw_faces = detect_faces(img, face_cascade) if use_faces else []
    face_boxes = [_expand_face_box(b, original_width, original_height) for b in raw_faces]

    if float(sal_small.max()) == 0.0 and not face_boxes:
        return ideal[0], ideal[1], crop_w, crop_h

    if free_axis == "x":
        pw = max(1, min(round(crop_w * scale), proxy_w - 1 if proxy_w > 1 else 1))
        ph = proxy_h
        max_full = original_width - crop_w
        max_proxy = proxy_w - pw
    else:
        pw = proxy_w
        ph = max(1, min(round(crop_h * scale), proxy_h))
        max_full = original_height - crop_h
        max_proxy = proxy_h - ph

    sal_int = integral_image(sal_small)
    edge_map = np.maximum(macro_n, 0.6 * sal_small).astype(np.float32)
    edge_int = integral_image(edge_map)

    sky_int: np.ndarray | None = None
    if free_axis == "y":
        sky_mask = _estimate_sky_mask(proxy_bgr, fine_n)
        if float(sky_mask.mean()) > 0.03:
            sky_int = integral_image(sky_mask)

    span = ph if free_axis == "y" else pw
    strip = max(2, round(0.03 * span)) if span >= 6 else 0
    margin = 0.05 * min(crop_w, crop_h)

    def score_candidate(start_x: int, start_y: int) -> float:
        if max_proxy > 0 and max_full > 0:
            pos_full = start_x if free_axis == "x" else start_y
            pos_proxy = min(max_proxy, max(0, round(pos_full * max_proxy / max_full)))
        else:
            pos_proxy = 0
        px, py = (pos_proxy, 0) if free_axis == "x" else (0, pos_proxy)

        score = _weighted_window_saliency(sal_int, px, py, pw, ph, free_axis)

        if strip > 0:
            if free_axis == "y":
                edge_start = box_mean_saliency(edge_int, px, py, pw, strip)
                edge_end = box_mean_saliency(edge_int, px, py + ph - strip, pw, strip)
            else:
                edge_start = box_mean_saliency(edge_int, px, py, strip, ph)
                edge_end = box_mean_saliency(edge_int, px + pw - strip, py, strip, ph)
            score -= 0.30 * (edge_start + edge_end)

        if sky_int is not None:
            sky_frac = box_mean_saliency(sky_int, px, py, pw, ph)
            score += 0.15 * min(sky_frac / 0.22, 1.0)
            if sky_frac < 0.05:
                score -= 0.12

        if max_full > 0:
            ideal_pos = ideal[0] if free_axis == "x" else ideal[1]
            cur_pos = start_x if free_axis == "x" else start_y
            half_range = max(1.0, max_full / 2.0)
            norm_dist = abs(cur_pos - ideal_pos) / half_range
            score -= 0.015 * norm_dist + 0.08 * (norm_dist ** 2)

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

    # Build candidate positions in full-res from each proxy step, plus exact center.
    seen_pos: set[tuple[int, int]] = {ideal}
    candidates: list[tuple[int, int]] = []
    if max_proxy > 0:
        for p in range(max_proxy + 1):
            full_p = min(max_full, round(p * max_full / max_proxy)) if max_full > 0 else 0
            cand = (full_p, 0) if free_axis == "x" else (0, full_p)
            if cand not in seen_pos:
                seen_pos.add(cand)
                candidates.append(cand)

    best_start = ideal
    best_score = score_candidate(*ideal)
    for start_x, start_y in candidates:
        score = score_candidate(start_x, start_y)
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
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
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
      saved | too_small | bad_aspect | too_blurry | exists | duplicate | format_error | error.
    """
    try:
        image = decode_image_bytes(image_bytes)
        orig_h, orig_w = image.shape[:2]
        if min_aspect_ratio > 0.0 and (orig_w / orig_h) < min_aspect_ratio:
            return False, "bad_aspect"

        # Choose crop aspect. With --target-resolution, crop to match it.
        target_aspect_ratio = TARGET_ASPECT_RATIO
        out_size: tuple[int, int] | None = None
        if target_resolution is not None:
            out_w_req, out_h_req = target_resolution
            if out_w_req <= 0 or out_h_req <= 0:
                return False, "error"
            target_aspect_ratio = out_w_req / out_h_req
            out_size = (out_w_req, out_h_req)

        # Resolution & no-upscale guardrails BEFORE expensive saliency/face scoring.
        crop_w, crop_h, _ = build_crop_box(image, target_aspect_ratio)
        if crop_w < min_width or crop_h < min_height:
            return False, "too_small"
        if out_size is not None and not allow_upscale and (
            crop_w < out_size[0] or crop_h < out_size[1]
        ):
            return False, "too_small"

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
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
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
        "skipped_aspect_ratio": 0,
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

                # Pre-download orientation/aspect guardrail using Reddit preview metadata.
                if min_aspect_ratio > 0.0:
                    preview_images = post.get("preview", {}).get("images", [])
                    if preview_images:
                        src = preview_images[0].get("source", {})
                        src_w, src_h = src.get("width"), src.get("height")
                        if (
                            isinstance(src_w, (int, float))
                            and isinstance(src_h, (int, float))
                            and src_w > 0
                            and src_h > 0
                            and (src_w / src_h) < min_aspect_ratio
                        ):
                            counters["skipped_aspect_ratio"] += 1
                            continue

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
                    min_aspect_ratio=min_aspect_ratio,
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
                elif reason == "bad_aspect":
                    counters["skipped_aspect_ratio"] += 1
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

    allow_portrait = questionary.confirm(
        "Allow portrait images (width < height) to be cropped to 16:9?",
        default=False,
    ).ask()

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
        "allow_portrait": bool(allow_portrait),
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
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
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
            min_aspect_ratio=min_aspect_ratio,
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
        "--min-aspect-ratio",
        type=float,
        default=MIN_ASPECT_RATIO,
        help=f"Minimum source width/height ratio before cropping; 0 disables (default: {MIN_ASPECT_RATIO}).",
    )
    parser.add_argument(
        "--allow-portrait",
        action="store_true",
        help="Allow portrait images (width < height) to be cropped; sets --min-aspect-ratio to 0.",
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
        min_aspect_ratio = 0.0 if config.get("allow_portrait") else args.min_aspect_ratio
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
        min_aspect_ratio = 0.0 if args.allow_portrait else args.min_aspect_ratio

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
            min_aspect_ratio=min_aspect_ratio,
            skip_existing=args.skip_existing,
            no_duplicates=args.no_duplicates,
            write_metadata=args.write_metadata,
            allow_nsfw=args.nsfw,
            jpeg_quality=args.jpeg_quality,
        )
    )


if __name__ == "__main__":
    cli_main()
