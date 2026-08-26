"""Build the smooth, isolated hand-cue animation from the AI sprite sheet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, features
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "assets" / "stimuli" / "hand-grasp-sprite-v3-side-16.png"
OUTPUT = PROJECT_ROOT / "assets" / "stimuli" / "hand-grasp-v3-side-16f.webp"

GRID_COLUMNS = 4
GRID_ROWS = 4
OUTPUT_SIZE = (320, 320)
FRAME_DURATION_MS = 125
CELL_INSET_PX = 3


def _source_frames(sprite: Image.Image) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            left = round(column * sprite.width / GRID_COLUMNS) + CELL_INSET_PX
            top = round(row * sprite.height / GRID_ROWS) + CELL_INSET_PX
            right = round((column + 1) * sprite.width / GRID_COLUMNS) - CELL_INSET_PX
            bottom = round((row + 1) * sprite.height / GRID_ROWS) - CELL_INSET_PX
            crop = np.asarray(
                sprite.crop((left, top, right, bottom)).convert("RGB")
            ).copy()

            # The generator encoded its transparency preview as a pale checkerboard.
            # Flood only the bright neutral region connected to the cell boundary,
            # protecting bright fingernails and skin highlights inside the hand.
            channel_max = crop.max(axis=2)
            channel_min = crop.min(axis=2)
            background_candidate = (channel_min >= 200) & (
                channel_max - channel_min <= 35
            )
            seed = np.zeros(background_candidate.shape, dtype=bool)
            seed[0, :] = background_candidate[0, :]
            seed[-1, :] = background_candidate[-1, :]
            seed[:, 0] = background_candidate[:, 0]
            seed[:, -1] = background_candidate[:, -1]
            background = ndimage.binary_propagation(
                seed,
                mask=background_candidate,
            )
            crop[background] = 0

            # A generated hand may extend a few pixels into a neighboring grid
            # cell. Keep only the largest connected foreground component so an
            # isolated fingertip or wrist fragment can never appear at an edge.
            foreground = crop.max(axis=2) > 12
            labels, component_count = ndimage.label(foreground)
            if component_count:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                main_hand = int(sizes.argmax())
                crop[labels != main_hand] = 0
            isolated = Image.fromarray(crop, mode="RGB")
            frames.append(isolated.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS))
    return frames


def main() -> None:
    if not features.check("webp"):
        raise RuntimeError("This Pillow build does not support WebP output")
    with Image.open(SOURCE) as sprite:
        frames = _source_frames(sprite)
    if len(frames) != 16:
        raise RuntimeError(f"Expected 16 frames, built {len(frames)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        quality=90,
        method=6,
    )

    with Image.open(OUTPUT) as animation:
        if int(getattr(animation, "n_frames", 1)) != len(frames):
            raise RuntimeError("Animated WebP frame count verification failed")
    duration_ms = len(frames) * FRAME_DURATION_MS
    if duration_ms != 2000:
        raise RuntimeError(f"Expected a 2000 ms animation, got {duration_ms} ms")

    print(f"built {OUTPUT} ({len(frames)} frames, {duration_ms} ms)")


if __name__ == "__main__":
    main()
