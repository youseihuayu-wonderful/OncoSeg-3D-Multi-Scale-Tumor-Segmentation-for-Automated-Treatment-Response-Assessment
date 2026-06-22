#!/usr/bin/env python3
"""Render the README preview assets from the live 3D architecture page.

Drives ``docs/architecture_3d.html`` headlessly (system Google Chrome via
Playwright) and produces two committed artefacts:

* ``docs/architecture_3d_preview.gif`` — the README banner. Captured in
  *poster mode* (``?poster=1&az=<deg>``): a left info column (title · result
  stats · legend) with the 3D model offset into the right ~60% of a wide 2:1
  frame, so text and model never overlap. The motion is a gentle ±38°
  sinusoidal *wobble* around the front (``--spin`` forces a full 360°): a full
  turn would put the model edge-on, where the encoder and decoder labels
  collide — the wobble keeps both columns separated and legible. ``--no-poster``
  falls back to just the centred ``&ui=0`` model.
* ``docs/architecture_3d_hero.png`` — a clean static still of the model
  (``&ui=0`` hides every overlay) used as the architecture image on the
  GitHub Pages landing page.

Why a GIF at all: GitHub strips <script>/<iframe> from rendered READMEs, so an
animated image is the only way to surface the interactive view on the repo page.

Usage (from the repo root, with the project venv)::

    .venv/bin/python docs/tools/render_preview.py            # both assets
    .venv/bin/python docs/tools/render_preview.py --gif      # GIF only
    .venv/bin/python docs/tools/render_preview.py --hero     # still only

Tunables: --frames (default 30), --delay-ms (default 150 → ~4.5 s/loop),
--sweep-deg (default 38), --spin, --gif-width, --colors, --width/--height,
--hero-az.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import math
import socketserver
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

DOCS_DIR = Path(__file__).resolve().parent.parent
PAGE = "architecture_3d.html"
POSTER_CAP = (1000, 500)  # capture viewport for the wide poster banner (2:1)


def _serve(directory: Path) -> tuple[socketserver.TCPServer, int]:
    """Serve *directory* on an ephemeral localhost port in a daemon thread."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    httpd.RequestHandlerClass.log_message = lambda *a, **k: None  # quiet
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _grab(page, url: str, width: int, height: int) -> bytes:
    """Load *url*, wait for the scene to settle, return a PNG screenshot."""
    page.set_viewport_size({"width": width, "height": height})
    page.goto(url, wait_until="networkidle")
    # The page removes #loading once three.js has built the scene.
    page.wait_for_selector("#loading", state="detached", timeout=30_000)
    page.wait_for_timeout(500)  # let bloom / tone-mapping stabilise
    return page.screenshot(type="png")


def render(*, gif: bool, hero: bool, frames: int, delay_ms: int, colors: int,
           width: int, height: int, hero_az: int, gif_width: int,
           spin: bool, sweep_deg: float, poster: bool) -> None:
    httpd, port = _serve(DOCS_DIR)
    base = f"http://127.0.0.1:{port}/{PAGE}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True,
                                        args=["--force-color-profile=srgb"])
            page = browser.new_page(device_scale_factor=2)
            try:
                if hero:
                    out = DOCS_DIR / "architecture_3d_hero.png"
                    png = _grab(page, f"{base}?az={hero_az}&ui=0", width, height)
                    out.write_bytes(png)
                    print(f"wrote {out.relative_to(DOCS_DIR.parent)} "
                          f"({len(png)//1024} KB, {width}x{height}@2x)")

                if gif:
                    # poster (default): the README banner — left info column
                    # (title · result stats · legend) + model offset right, wide
                    # 2:1 frame. plain (--no-poster): just the centred ui=0 model.
                    cap_w, cap_h = POSTER_CAP if poster else (width, height)
                    gw = gif_width
                    gh = round(cap_h * gw / cap_w)
                    # Azimuth schedule (seamless loop):
                    #  * sweep (default): a gentle sinusoidal wobble in ±sweep_deg
                    #    around the front. Both encoder/decoder columns stay
                    #    horizontally separated, so the 3D labels never collide —
                    #    the model reads cleanly and nothing overlaps.
                    #  * spin: a full 360° turn (labels overlap edge-on; opt-in).
                    if spin:
                        azimuths = [i * 360 / frames for i in range(frames)]
                        motion = f"360° spin, {360/frames:.1f}°/frame"
                    else:
                        azimuths = [sweep_deg * math.sin(2 * math.pi * i / frames)
                                    for i in range(frames)]
                        motion = f"±{sweep_deg}° sweep"
                    imgs: list[Image.Image] = []
                    for i, az in enumerate(azimuths):
                        az = round(az, 3)
                        url = (f"{base}?poster=1&az={az}" if poster
                               else f"{base}?az={az}&ui=0")
                        png = _grab(page, url, cap_w, cap_h)
                        import io
                        im = Image.open(io.BytesIO(png)).convert("RGB")
                        # frames captured @2x for AA, downscaled for a small GIF
                        imgs.append(im.resize((gw, gh), Image.LANCZOS))
                        print(f"  frame {i + 1}/{frames}  az={az:.1f}°", end="\r")
                    print()
                    # Quantise to a shared adaptive palette for a small file.
                    pal = imgs[0].quantize(colors=colors, method=Image.MEDIANCUT)
                    qimgs = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG)
                             for im in imgs]
                    out = DOCS_DIR / "architecture_3d_preview.gif"
                    qimgs[0].save(out, save_all=True, append_images=qimgs[1:],
                                  duration=delay_ms, loop=0, optimize=True,
                                  disposal=2)
                    loop_s = frames * delay_ms / 1000
                    print(f"wrote {out.relative_to(DOCS_DIR.parent)} "
                          f"({out.stat().st_size // 1024} KB, {frames} frames, "
                          f"{loop_s:.1f}s/loop, {motion})")
            finally:
                browser.close()
    finally:
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", action="store_true", help="render the GIF only")
    ap.add_argument("--hero", action="store_true", help="render the still only")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--delay-ms", type=int, default=150)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=506)
    ap.add_argument("--hero-az", type=int, default=28)
    ap.add_argument("--gif-width", type=int, default=760,
                    help="GIF is downscaled to this width (frames captured @2x)")
    ap.add_argument("--colors", type=int, default=64, help="GIF palette size")
    ap.add_argument("--spin", action="store_true",
                    help="full 360° turn instead of the default ±sweep wobble "
                         "(labels overlap when the model is edge-on)")
    ap.add_argument("--sweep-deg", type=float, default=38.0,
                    help="amplitude of the wobble; kept <~40° so the encoder/"
                         "decoder labels never collide")
    ap.add_argument("--no-poster", dest="poster", action="store_false",
                    help="GIF shows just the centred model (no info column)")
    ap.set_defaults(poster=True)
    args = ap.parse_args()
    # Default (neither flag) renders both.
    do_gif = args.gif or not (args.gif or args.hero)
    do_hero = args.hero or not (args.gif or args.hero)
    render(gif=do_gif, hero=do_hero, frames=args.frames, delay_ms=args.delay_ms,
           width=args.width, height=args.height, hero_az=args.hero_az,
           gif_width=args.gif_width, colors=args.colors,
           spin=args.spin, sweep_deg=args.sweep_deg, poster=args.poster)


if __name__ == "__main__":
    main()
