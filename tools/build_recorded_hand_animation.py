"""Segment the recorded hand frames and build the two-second cue animation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, features
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_DIRECTORY = PROJECT_ROOT / "tmp" / "recorded-hand-frames"
OUTPUT = PROJECT_ROOT / "assets" / "stimuli" / "hand-grasp-recorded-v1-20fps.webp"
QA_OUTPUT = PROJECT_ROOT / "video" / "recorded-hand-cue-qa.png"

FRAME_COUNT = 40
FRAME_DURATION_MS = 50
OUTPUT_SIZE = (512, 512)


def _hue_saturation_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = rgb.astype(np.float32) / 255.0
    red, green, blue = np.moveaxis(normalized, -1, 0)
    channel_max = normalized.max(axis=2)
    channel_min = normalized.min(axis=2)
    delta = channel_max - channel_min
    hue = np.zeros(channel_max.shape, dtype=np.float32)
    nonzero = delta > 1e-5

    red_max = nonzero & (channel_max == red)
    green_max = nonzero & (channel_max == green)
    blue_max = nonzero & (channel_max == blue)
    hue[red_max] = 60.0 * np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = 60.0 * (((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0)
    hue[blue_max] = 60.0 * (((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0)
    saturation = np.divide(
        delta,
        channel_max,
        out=np.zeros_like(delta),
        where=channel_max > 1e-5,
    )
    return hue, saturation, channel_max


def _keep_main_hand(mask: np.ndarray) -> np.ndarray:
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    axis = np.arange(-4, 5)
    disk = axis[:, None] ** 2 + axis[None, :] ** 2 <= 4**2
    mask = ndimage.binary_opening(mask, structure=disk)
    labels, component_count = ndimage.label(mask)
    if not component_count:
        raise RuntimeError("No foreground component found")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    hand_label = int(sizes.argmax())
    hand = labels == hand_label

    # Repair only tiny internal gaps caused by highlights; preserve the large
    # spaces between fingers and the opening inside a relaxed fist.
    holes, hole_count = ndimage.label(~hand)
    if hole_count:
        hole_sizes = np.bincount(holes.ravel())
        border_labels = np.unique(
            np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))
        )
        fill = np.zeros_like(hand)
        for label in range(1, hole_count + 1):
            if label not in border_labels and hole_sizes[label] <= 180:
                fill |= holes == label
        hand |= fill
    return hand


def _segment_frame(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS))
    hue, saturation, value = _hue_saturation_value(rgb)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)

    skin_core = (
        (hue <= 28.0)
        & (saturation >= 0.10)
        & (value >= 0.20)
        & (value <= 0.96)
        & ((red - green) >= 9)
        & ((red - blue) >= 17)
        & ((green - blue) >= -8)
        & ((green - blue) <= 34)
    )
    foreground_seed = ndimage.binary_erosion(
        skin_core,
        structure=np.ones((3, 3), dtype=bool),
    )
    foreground_distance = ndimage.distance_transform_edt(~foreground_seed)
    wood = (
        (hue >= 29.0)
        & (hue <= 72.0)
        & (saturation >= 0.055)
        & (value >= 0.42)
        & (foreground_distance >= 18.0)
    )

    background_seed = wood.copy()
    background_seed[:5, :] = True
    background_seed[-5:, :] = True
    background_seed[:, :5] = True
    background_seed[:90, -5:] = True
    background_seed[-90:, -5:] = True
    background_seed[foreground_distance < 18.0] = False

    rgb_float = rgb.astype(np.float32)
    gradient_sq = np.zeros(rgb.shape[:2], dtype=np.float32)
    for channel in range(3):
        horizontal = ndimage.sobel(rgb_float[:, :, channel], axis=1)
        vertical = ndimage.sobel(rgb_float[:, :, channel], axis=0)
        gradient_sq += horizontal * horizontal + vertical * vertical
    gradient = np.sqrt(gradient_sq)
    gradient_scale = float(np.quantile(gradient, 0.995)) or 1.0
    gradient_u8 = np.clip(gradient / gradient_scale * 255.0, 0, 255).astype(np.uint8)

    markers = np.zeros(rgb.shape[:2], dtype=np.int32)
    markers[background_seed] = 1
    markers[foreground_seed] = 2
    watershed = ndimage.watershed_ift(gradient_u8, markers)
    # Watershed follows the real hand edge through low-saturation highlights,
    # but it can occasionally bridge into the warm desk. The desk starts around
    # 32 degrees in this recording, while the hand/nails remain below 31 degrees
    # (or wrap around red). Retain a three-pixel edge halo for antialiasing.
    skin_edge = (
        ((hue <= 31.0) | (hue >= 345.0))
        & (saturation >= 0.06)
        & (value >= 0.18)
        & (value <= 0.98)
        & ((red - green) >= 6)
        & ((red - blue) >= 13)
        & ((green - blue) >= -12)
        & ((green - blue) <= 38)
    )
    edge_halo = ndimage.binary_dilation(foreground_seed, iterations=3)
    hand = _keep_main_hand((watershed == 2) & (skin_edge | edge_halo))
    alpha = ndimage.gaussian_filter(hand.astype(np.float32), sigma=1.15)
    alpha = np.clip((alpha - 0.04) / 0.92, 0.0, 1.0)

    rgba = np.dstack((rgb, np.round(alpha * 255.0).astype(np.uint8)))
    rgba[rgba[:, :, 3] == 0, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def _write_qa(frames: list[Image.Image]) -> None:
    indices = [0, 5, 10, 15, 20, 25, 30, 35, 39]
    cell_size = 256
    sheet = Image.new("RGB", (cell_size * 3, cell_size * 3), "black")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        preview = Image.new("RGBA", OUTPUT_SIZE, (0, 0, 0, 255))
        preview.alpha_composite(frames[index])
        preview = preview.convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        x = position % 3 * cell_size
        y = position // 3 * cell_size
        sheet.paste(preview, (x, y))
        draw.text((x + 7, y + 7), str(index), fill="white")
    QA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(QA_OUTPUT)


def main() -> None:
    if not features.check("webp"):
        raise RuntimeError("This Pillow build does not support WebP output")
    paths = [FRAME_DIRECTORY / f"raw-{index:03d}.png" for index in range(FRAME_COUNT)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing extracted video frames: {missing[:3]}")

    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(_segment_frame(image))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        quality=92,
        method=4,
    )
    with Image.open(OUTPUT) as animation:
        if animation.n_frames != FRAME_COUNT:
            raise RuntimeError(
                f"Expected {FRAME_COUNT} encoded frames, got {animation.n_frames}"
            )
        animation.seek(FRAME_COUNT // 2)
        if animation.convert("RGBA").getchannel("A").getextrema()[0] != 0:
            raise RuntimeError("Encoded animation does not preserve transparency")
    _write_qa(frames)
    print(
        f"built {OUTPUT} ({FRAME_COUNT} frames, "
        f"{FRAME_COUNT * FRAME_DURATION_MS} ms)"
    )


if __name__ == "__main__":
    main()
