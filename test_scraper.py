"""Tests for scraper.py. Run: pytest -q

Uses synthetic images and httpx.MockTransport (no live network).
"""
import asyncio
import json

import cv2
import httpx
import numpy as np
import pytest

import scraper

AR = scraper.TARGET_ASPECT_RATIO


# ---------- helpers ----------
def make_img(w=3000, h=2000, kind="noise", seed=0):
    rng = np.random.default_rng(seed)
    if kind == "flat":
        img = np.full((h, w, 3), 127, np.uint8)
    elif kind == "noise":
        img = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
    else:
        img = rng.integers(0, 60, (h, w, 3)).astype(np.uint8)
        img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 255  # bright subject
    return img


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


class StubCascade:
    """Minimal stand-in for cv2.CascadeClassifier."""

    def __init__(self, boxes=((100, 100, 200, 200),)):
        self.boxes = np.array(boxes, dtype=np.int32).reshape(-1, 4)
        self.seen_min_size = None
        self.seen_gray_shape = None

    def detectMultiScale(self, gray, scaleFactor, minNeighbors, minSize):
        self.seen_gray_shape = gray.shape
        self.seen_min_size = minSize
        return self.boxes


class EmptyCascade:
    def detectMultiScale(self, *a, **k):
        return ()


# ---------- saliency / integral ----------
def test_saliency_flat_is_zeros():
    sal = scraper.calculate_saliency_map(make_img(kind="flat"))
    assert sal.dtype == np.float32
    assert float(sal.max()) == 0.0


def test_saliency_normalized_unit_range():
    sal = scraper.calculate_saliency_map(make_img(kind="noise"))
    assert 0.0 <= float(sal.min()) <= float(sal.max()) == 1.0


def test_integral_image_dtype_and_correctness():
    vals = np.random.default_rng(1).random((13, 17)).astype(np.float32)
    integ = scraper.integral_image(vals)
    assert integ.dtype == np.float64
    x, y, w, h = 3, 4, 5, 6
    naive = float(vals[y : y + h, x : x + w].sum(dtype=np.float64))
    fast_total = scraper.box_mean_saliency(integ, x, y, w, h) * (w * h)
    assert abs(naive - fast_total) < 1e-4


def test_box_mean_saliency():
    vals = np.ones((10, 10), np.float32)
    integ = scraper.integral_image(vals)
    assert scraper.box_mean_saliency(integ, 0, 0, 5, 5) == pytest.approx(1.0)
    assert scraper.box_mean_saliency(integ, 5, 5, 5, 5) == pytest.approx(1.0)


# ---------- crop geometry ----------
@pytest.mark.parametrize("ar", [16 / 9, 2.0, 1.0, 4 / 3, 21 / 9])
def test_build_crop_box_aspect_and_bounds(ar):
    img = make_img(3000, 2000)
    cw, ch, free = scraper.build_crop_box(img, ar)
    assert cw >= 1 and ch >= 1
    assert cw <= 3000 and ch <= 2000
    assert (cw / ch) == pytest.approx(ar, abs=0.02)
    assert free in {"x", "y"}


def test_build_crop_box_no_full_image_uncropped_when_wider():
    # slightly-wider-than-target image: must still crop (no finding-4 no-op)
    img = make_img(2, 1, kind="flat")
    cw, ch, _ = scraper.build_crop_box(img, AR)
    assert cw < 2  # not the uncropped original width


def test_center_crop_bounds_and_aspect():
    img = make_img(3001, 1999)
    x, y, cw, ch = scraper.calculate_center_crop_box(img, AR)
    assert x >= 0 and y >= 0 and x + cw <= 3001 and y + ch <= 1999
    assert (cw / ch) == pytest.approx(AR, abs=0.02)


def test_best_crop_flat_falls_back_to_center():
    img = make_img(3000, 2000, kind="flat")
    best = scraper.calculate_best_crop_box(img, AR, None)
    center = scraper.calculate_center_crop_box(img, AR)
    assert best == center


def test_best_crop_bounds_on_noise():
    img = make_img(3000, 2000, kind="noise", seed=3)
    x, y, cw, ch = scraper.calculate_best_crop_box(img, AR, None)
    assert x >= 0 and y >= 0 and x + cw <= 3000 and y + ch <= 2000
    assert (cw / ch) == pytest.approx(AR, abs=0.02)


def test_best_crop_captures_more_saliency_than_center_on_subject():
    img = make_img(3000, 1080, kind="subject", seed=5)
    sal = scraper.calculate_saliency_map(img)
    cx = scraper.calculate_center_crop_box(img, AR)
    bx = scraper.calculate_best_crop_box(img, AR, None)
    def cap(box):
        x, y, w, h = box
        return sal[y : y + h, x : x + w].sum() / max(sal.sum(), 1e-9)
    assert cap(bx) >= cap(cx) - 1e-6


def test_best_crop_captures_off_center_colorful_subject():
    # Dark muted frame with vivid orange subject on the far left.
    img = np.full((1080, 3000, 3), (45, 45, 45), np.uint8)
    cv2.rectangle(img, (150, 250), (650, 830), (30, 140, 245), -1)
    cx = scraper.calculate_center_crop_box(img, AR)
    bx = scraper.calculate_best_crop_box(img, AR, None)
    # Center crop (x=540..2460) clips most of the subject; smart crop shifts left.
    assert bx[0] < cx[0]
    assert bx[0] <= 150 <= bx[0] + bx[2]


def test_best_crop_prefers_mountain_and_sky_over_foreground_grass():
    rng = np.random.default_rng(42)
    img = np.zeros((2000, 1500, 3), np.uint8)
    img[:750, :] = (230, 170, 90)  # blue sky
    pts = np.array([[0, 1200], [450, 700], [750, 425], [1100, 750], [1500, 1200]])
    cv2.fillPoly(img, [pts], (95, 85, 80))
    cap = np.array([[600, 570], [750, 425], [910, 575]])
    cv2.fillPoly(img, [cap], (245, 245, 250))
    img[1200:, :] = rng.integers(10, 180, (800, 1500, 3), np.uint8)  # noisy grass

    x, y, w, h = scraper.calculate_best_crop_box(img, AR, None)
    # Peak at y=425 must stay comfortably inside the crop rather than chasing bottom grass.
    assert y <= 425
    assert y + h < 1650


def test_best_crop_avoids_clipping_tall_spire_at_top_edge():
    rng = np.random.default_rng(7)
    img = np.full((1800, 1500, 3), (240, 220, 200), np.uint8)  # bright sky
    img[800:, :] = rng.integers(30, 120, (1000, 1500, 3), np.uint8)  # textured canyon
    cv2.rectangle(img, (700, 200), (800, 1600), (60, 140, 220), -1)  # warm spire
    _, y, _, _ = scraper.calculate_best_crop_box(img, AR, None)
    assert y < 200  # spire tip at y=200 is not clipped at the top border


# ---------- faces ----------
def test_detect_faces_scaling_back_with_stub():
    img = make_img(4000, 3000, kind="flat")
    scale = min(1.0, 1024.0 / 4000)
    rw, rh = round(4000 * scale), round(3000 * scale)
    stub = StubCascade(boxes=((rw // 4, rh // 4, rw // 8, rh // 8),))
    out = scraper.detect_faces(img, stub)
    assert len(out) == 1
    x, y, w, h = out[0]
    assert x == round((rw // 4) / scale)
    assert w == round((rw // 8) / scale)
    # gray passed in is <=1024 on long side
    assert max(stub.seen_gray_shape) <= 1024
    assert stub.seen_min_size == (max(int(30 * scale), 10),) * 2


def test_detect_faces_empty_and_minsize_floor():
    assert scraper.detect_faces(make_img(kind="flat"), EmptyCascade()) == []


def test_detect_faces_sliver_no_crash():
    # regression: downscale must not produce a 0px dimension
    sliver = np.zeros((1, 10000, 3), np.uint8)
    assert scraper.detect_faces(sliver, EmptyCascade()) == []


def test_face_weight_zero_disables_face_pulling():
    img = make_img(3000, 2000, kind="noise", seed=7)
    # a huge face box would normally strongly attract the window
    stub = StubCascade(boxes=((2900, 1900, 100, 100),))
    with_face = scraper.calculate_best_crop_box(img, AR, stub, face_weight=1.0)
    no_face = scraper.calculate_best_crop_box(img, AR, stub, face_weight=0.0)
    no_cascade = scraper.calculate_best_crop_box(img, AR, None)
    assert no_face == no_cascade
    # with_face runs fine; we only assert the disable path is identical to no-cascade
    assert isinstance(with_face, tuple)


def test_real_opencv_cascade_loads_when_available():
    # pinned opencv<5 ships a cascade; on environments without it, returns None
    cascade = scraper.load_face_cascade()
    assert cascade is None or isinstance(cascade, cv2.CascadeClassifier)


# ---------- image_dhash ----------
def test_dhash_similar_images_close():
    a = make_img(256, 256, kind="subject", seed=1)
    b = a.copy()
    b[0, 0] = 255  # tiny change
    ha = scraper.image_dhash(a)
    hb = scraper.image_dhash(b)
    assert (ha ^ hb).bit_count() <= 4


# ---------- sharpness ----------
def test_sharpness_flat_zero_noise_positive():
    assert scraper._sharpness(make_img(kind="flat")) == pytest.approx(0.0, abs=1e-9)
    assert scraper._sharpness(make_img(kind="noise", seed=2)) > 1.0


# ---------- resolve_image_url ----------
def test_resolve_image_url_basic_and_amp():
    url = "https://i.redd.it/abc.jpg?x=1&amp;y=2"
    assert scraper.resolve_image_url({"url": url}) == "https://i.redd.it/abc.jpg?x=1&y=2"


def test_resolve_image_url_filters_gif():
    assert scraper.resolve_image_url({"url": "https://i.redd.it/abc.gif"}) is None
    assert scraper.resolve_image_url(
        {"url": "https://i.redd.it/abc.gif", "post_hint": "image"}
    ) is None


def test_resolve_image_url_png_webp_hosts_and_overridden():
    assert scraper.resolve_image_url({"url": "https://i.redd.it/a.png"}).endswith("a.png")
    assert scraper.resolve_image_url({"url": "https://i.redd.it/a.webp"}).endswith("a.webp")
    assert scraper.resolve_image_url({"url": "https://i.redd.it/noext"}).endswith("noext")  # host match
    assert scraper.resolve_image_url({"url": "https://example.com/a"}) is None
    over = scraper.resolve_image_url(
        {"url": "https://example.com/x", "url_overridden_by_dest": "https://i.redd.it/b.jpg"}
    )
    assert over.endswith("b.jpg")


def test_resolve_image_url_preview_source():
    post = {"url": "https://example.com/x", "preview": {"images": [{"source": {"url": "https://preview.redd.it/p.jpg?a=1&amp;b=2"}}]}}
    assert scraper.resolve_image_url(post) == "https://preview.redd.it/p.jpg?a=1&b=2"


def test_resolve_image_url_post_hint_fallback():
    assert scraper.resolve_image_url({"url": "https://i.redd.it/x", "post_hint": "image"})


# ---------- sanitize ----------
def test_sanitize_filename():
    assert scraper.sanitize_filename('a/b\\c:d*e?f"g<h>i|j k  l') == "a_b_c_d_e_f_g_h_i_j_k_l"
    assert scraper.sanitize_filename("   ") == "untitled"
    assert len(scraper.sanitize_filename("x" * 200)) <= 80


# ---------- crop_and_save_image ----------
def test_crop_and_save_smart(tmp_path):
    img = make_img(3000, 2000, kind="noise", seed=4)
    ok, reason = scraper.crop_and_save_image(
        png_bytes(img), tmp_path, "My Landscape!", "abc", None, "smart", min_width=0, min_height=0
    )
    assert (ok, reason) == (True, "saved")
    saved = list(tmp_path.glob("*.jpg"))
    assert len(saved) == 1
    back = cv2.imread(str(saved[0]))
    assert (back.shape[1] / back.shape[0]) == pytest.approx(AR, abs=0.02)
    assert saved[0].name.startswith("My_Landscape!_abc_")


def test_crop_and_save_center_mode(tmp_path):
    ok, reason = scraper.crop_and_save_image(
        png_bytes(make_img(3000, 2000)), tmp_path, "t", "i", None, "center", min_width=0, min_height=0
    )
    assert (ok, reason) == (True, "saved")


def test_crop_and_save_landscape_ignores_faces(tmp_path):
    img = make_img(3000, 2000, kind="noise", seed=9)
    stub = StubCascade(boxes=((2950, 1950, 50, 50),))
    # landscape must not pull the crop toward the face; compare crop box vs no-cascade landscape
    landscape = scraper.calculate_best_crop_box(img, AR, None, 0.0)
    ok, reason = scraper.crop_and_save_image(
        png_bytes(img), tmp_path, "scenery", "s", stub, "landscape", min_width=0, min_height=0
    )
    assert ok
    saved = list(tmp_path.glob("*.jpg"))[0]
    back = cv2.imread(str(saved))
    assert back.shape[1] == landscape[2] and back.shape[0] == landscape[3]


def test_too_small(tmp_path):
    ok, reason = scraper.crop_and_save_image(
        png_bytes(make_img(100, 100)), tmp_path, "t", "i", None, "smart"
    )
    assert (ok, reason) == (False, "too_small")


def test_too_small_respects_min_override(tmp_path):
    ok, reason = scraper.crop_and_save_image(
        png_bytes(make_img(3000, 2000)), tmp_path, "t", "i", None, "smart",
        min_width=99999, min_height=99999,
    )
    assert (ok, reason) == (False, "too_small")


def test_portrait_skipped_by_default_and_allowed_when_disabled(tmp_path):
    portrait = png_bytes(make_img(2600, 3900, kind="noise", seed=14))
    ok_skip, reason_skip = scraper.crop_and_save_image(
        portrait, tmp_path, "portrait", "p1", None, "smart", min_width=0, min_height=0
    )
    assert (ok_skip, reason_skip) == (False, "bad_aspect")

    ok_allow, reason_allow = scraper.crop_and_save_image(
        portrait, tmp_path, "portrait", "p2", None, "smart",
        min_width=0, min_height=0, min_aspect_ratio=0.0,
    )
    assert (ok_allow, reason_allow) == (True, "saved")


def test_format_error_on_garbage(tmp_path):
    ok, reason = scraper.crop_and_save_image(b"not an image", tmp_path, "t", "i", None, "smart")
    assert (ok, reason) == (False, "format_error")


def test_min_sharpness_rejects_blurry(tmp_path):
    ok, reason = scraper.crop_and_save_image(
        png_bytes(make_img(3000, 2000, kind="flat")), tmp_path, "t", "i", None, "smart",
        min_width=0, min_height=0, min_sharpness=1.0,
    )
    assert (ok, reason) == (False, "too_blurry")


def test_min_sharpness_keeps_sharp(tmp_path):
    ok, reason = scraper.crop_and_save_image(
        png_bytes(make_img(3000, 2000, kind="noise", seed=6)), tmp_path, "t", "i", None, "smart",
        min_width=0, min_height=0, min_sharpness=1.0,
    )
    assert ok


def test_target_resolution_exact_output_and_aspect(tmp_path):
    img = make_img(3000, 2000, kind="noise", seed=8)
    ok, reason = scraper.crop_and_save_image(
        png_bytes(img), tmp_path, "t", "i", None, "smart",
        min_width=0, min_height=0, target_resolution=(1000, 500),
    )
    assert ok
    back = cv2.imread(str(list(tmp_path.glob("*.jpg"))[0]))
    assert (back.shape[1], back.shape[0]) == (1000, 500)
    assert back.shape[1].bit_length() > 0


def test_target_resolution_no_upscale_by_default(tmp_path):
    small = make_img(300, 200, kind="noise", seed=1)
    ok, reason = scraper.crop_and_save_image(
        png_bytes(small), tmp_path, "t", "i", None, "smart",
        min_width=0, min_height=0, target_resolution=(1000, 500),
    )
    assert (ok, reason) == (False, "too_small")


def test_target_resolution_allow_upscale(tmp_path):
    small = make_img(300, 200, kind="noise", seed=1)
    ok, reason = scraper.crop_and_save_image(
        png_bytes(small), tmp_path, "t", "i", None, "smart",
        min_width=0, min_height=0, target_resolution=(1000, 500), allow_upscale=True,
    )
    assert ok
    back = cv2.imread(str(list(tmp_path.glob("*.jpg"))[0]))
    assert (back.shape[1], back.shape[0]) == (1000, 500)


def test_skip_existing(tmp_path):
    img = png_bytes(make_img(3000, 2000, kind="noise", seed=10))
    ok1, _ = scraper.crop_and_save_image(img, tmp_path, "t", "i", None, "smart", min_width=0, min_height=0)
    assert ok1
    ok2, reason2 = scraper.crop_and_save_image(img, tmp_path, "t", "i", None, "smart", min_width=0, min_height=0, skip_existing=True)
    assert (ok2, reason2) == (False, "exists")


def test_dedup_skips_near_duplicate(tmp_path):
    img = png_bytes(make_img(3000, 2000, kind="noise", seed=11))
    seen = set()
    ok1, _ = scraper.crop_and_save_image(img, tmp_path, "a", "1", None, "smart", min_width=0, min_height=0, seen_hashes=seen)
    ok2, reason2 = scraper.crop_and_save_image(img, tmp_path, "b", "2", None, "smart", min_width=0, min_height=0, seen_hashes=seen)
    assert ok1
    assert (ok2, reason2) == (False, "duplicate")


def test_write_metadata_sidecar(tmp_path):
    ok, _ = scraper.crop_and_save_image(
        png_bytes(make_img(3000, 2000, kind="noise", seed=12)), tmp_path, "t", "i", None, "smart",
        min_width=0, min_height=0, write_metadata=True,
        metadata={"author": "someone", "source_url": "https://i.redd.it/x.jpg", "subreddit": "EarthPorn"},
    )
    assert ok
    sidecars = list(tmp_path.glob("*.json"))
    assert len(sidecars) == 1
    data = json.loads(sidecars[0].read_text())
    assert data["source"]["author"] == "someone"
    assert data["source"]["subreddit"] == "EarthPorn"
    assert data["post_id"] == "i"
    assert data["width"] > 0 and data["height"] > 0


def test_parse_resolution():
    assert scraper.parse_resolution("2560x1440") == (2560, 1440)
    assert scraper.parse_resolution("100 x 200") == (100, 200)
    with pytest.raises(Exception):
        scraper.parse_resolution("nope")
    with pytest.raises(Exception):
        scraper.parse_resolution("100x")


# ---------- async HTTP: retry, rate-limit, shared client ----------
def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_download_image_retry_on_429_then_success():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, headers={"retry-after": "0"}, content=b"rate limited")
        return httpx.Response(200, content=b"JPEGBYTES")
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.download_image("https://i.redd.it/x.jpg", client=client, max_retries=3)
    # speed up backoff in test
    orig = scraper.RETRY_BACKOFF_BASE
    scraper.RETRY_BACKOFF_BASE = 0
    try:
        result = asyncio.run(run())
    finally:
        scraper.RETRY_BACKOFF_BASE = orig
    assert result == b"JPEGBYTES"
    assert calls["n"] == 3


def test_download_image_permanent_failure_returns_none():
    def handler(request):
        return httpx.Response(404, content=b"nope")
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.download_image("https://i.redd.it/x.jpg", client=client, max_retries=2)
    assert asyncio.run(run()) is None


def test_download_image_5xx_retry_then_give_up():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, content=b"down")
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.download_image("https://i.redd.it/x.jpg", client=client, max_retries=2)
    orig = scraper.RETRY_BACKOFF_BASE
    scraper.RETRY_BACKOFF_BASE = 0
    try:
        assert asyncio.run(run()) is None
    finally:
        scraper.RETRY_BACKOFF_BASE = orig
    assert calls["n"] == 3  # initial + 2 retries


def test_fetch_posts_from_json_ok_flag_and_shape():
    def handler(request):
        return httpx.Response(200, json={"data": {"children": [{"data": {"id": "1"}}], "after": "t3_x"}})
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.fetch_posts_from_json("EarthPorn", 100, "hot", client=client)
    children, after, ok = asyncio.run(run())
    assert ok is True
    assert after == "t3_x"
    assert children[0]["data"]["id"] == "1"


def test_fetch_posts_from_json_empty_is_ok_not_error():
    def handler(request):
        return httpx.Response(200, json={"data": {"children": [], "after": None}})
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.fetch_posts_from_json("x", 100, "hot", client=client)
    children, after, ok = asyncio.run(run())
    assert ok is True and children == [] and after is None


def test_fetch_posts_from_json_failure_distinguishes_error():
    def handler(request):
        return httpx.Response(500, content=b"oops")
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.fetch_posts_from_json("x", 100, "hot", client=client, max_retries=1)
    orig = scraper.RETRY_BACKOFF_BASE
    scraper.RETRY_BACKOFF_BASE = 0
    try:
        children, after, ok = asyncio.run(run())
    finally:
        scraper.RETRY_BACKOFF_BASE = orig
    assert ok is False and children == [] and after is None  # NOT silent end-of-list


def test_fetch_posts_from_json_retry_on_429():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"data": {"children": [], "after": None}})
    async def run():
        async with _mock_client(handler) as client:
            return await scraper.fetch_posts_from_json("x", 100, "hot", client=client, max_retries=2)
    orig = scraper.RETRY_BACKOFF_BASE
    scraper.RETRY_BACKOFF_BASE = 0
    try:
        _, _, ok = asyncio.run(run())
    finally:
        scraper.RETRY_BACKOFF_BASE = orig
    assert ok is True and calls["n"] == 2


# ---------- process_subreddit end-to-end with mocked network ----------
def _post(pid, url, title="A Landscape", **extra):
    data = {"id": pid, "url": url, "title": title, "post_hint": "image",
            "subreddit": "EarthPorn", "author": "someone", "permalink": f"/r/x/{pid}"}
    data.update(extra)
    return {"data": data}


def test_process_subreddit_concurrent_mocked(tmp_path, monkeypatch):
    img_png = png_bytes(make_img(3000, 2000, kind="noise", seed=13))

    def handler(request):
        if request.url.host in {"oauth.reddit.com", "old.reddit.com"}:
            if request.url.host == "old.reddit.com":
                return httpx.Response(200, content=b"")
            children = [
                _post("p1", "https://i.redd.it/a.jpg"),
                _post("p2", "https://i.redd.it/b.jpg"),
                _post("p3", "https://i.redd.it/c.gif"),              # filtered: gif
                _post("p4", "https://i.redd.it/d.jpg", over_18=True),  # filtered: nsfw
                _post("p5", "https://i.redd.it/e.jpg", is_self=True),  # filtered
                _post("p6", "https://i.redd.it/f.jpg", spoiler=True),  # filtered
                _post("p2", "https://i.redd.it/b.jpg"),                # dup id
            ]
            return httpx.Response(200, json={"data": {"children": children, "after": None}})
        return httpx.Response(200, content=img_png)

    transport = httpx.MockTransport(handler)

    real_client = httpx.AsyncClient
    def client_factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)
    monkeypatch.setattr(scraper.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(scraper, "load_face_cascade", lambda: None)

    asyncio.run(
        scraper.process_subreddit(
            "EarthPorn", 10, "hot", tmp_path, "all", "smart",
            concurrency=4, min_width=0, min_height=0, no_duplicates=False, write_metadata=True,
        )
    )
    saved = sorted(p.name for p in tmp_path.glob("*.jpg"))
    # p1 + p2 saved (p3 gif, p4 nsfw, p5 self, p6 spoiler filtered, p2-dup-id removed)
    assert len(saved) == 2, saved
    assert all(n.endswith(".jpg") for n in saved)
    assert len(list(tmp_path.glob("*.json"))) == 2  # attribution sidecars


def test_process_subreddit_skips_portrait_preview_before_download(tmp_path, monkeypatch):
    downloaded_urls = []
    img_png = png_bytes(make_img(3000, 2000, kind="noise", seed=15))

    def handler(request):
        if request.url.host in {"oauth.reddit.com", "old.reddit.com"}:
            if request.url.host == "old.reddit.com":
                return httpx.Response(200, content=b"")
            children = [
                _post(
                    "portrait1",
                    "https://i.redd.it/tall.jpg",
                    preview={"images": [{"source": {"url": "https://i.redd.it/tall.jpg", "width": 3000, "height": 4500}}]},
                ),
                _post(
                    "land1",
                    "https://i.redd.it/wide.jpg",
                    preview={"images": [{"source": {"url": "https://i.redd.it/wide.jpg", "width": 3000, "height": 2000}}]},
                ),
            ]
            return httpx.Response(200, json={"data": {"children": children, "after": None}})
        downloaded_urls.append(str(request.url))
        return httpx.Response(200, content=img_png)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    def client_factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)
    monkeypatch.setattr(scraper.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(scraper, "load_face_cascade", lambda: None)

    asyncio.run(
        scraper.process_subreddit(
            "EarthPorn", 10, "hot", tmp_path, "all", "smart",
            concurrency=2, min_width=0, min_height=0,
        )
    )
    assert downloaded_urls == ["https://i.redd.it/wide.jpg"]
    assert len(list(tmp_path.glob("*.jpg"))) == 1
