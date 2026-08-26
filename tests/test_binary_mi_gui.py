"""Contract tests for the isolated binary-MI presentation layer."""

from __future__ import annotations

import base64
from pathlib import Path
import re
import unittest

from PIL import Image

from utils.binary_mi_gui import (
    GUIDANCE_STEPS,
    HAND_ANIMATION_PATH,
    MI_BACKGROUND,
    MI_GREEN,
    STIMULUS_CSS,
    MiVisualStage,
    frame_html,
    resolve_mi_visual,
)


class BinaryMiGuiTests(unittest.TestCase):
    def test_guidance_defines_the_four_second_imagery_rhythm(self) -> None:
        guidance = " ".join(body for _frame, _title, body in GUIDANCE_STEPS)
        self.assertIn("4 秒内", guidance)
        self.assertIn("握拳—松开", guidance)
        self.assertIn("大约两轮", guidance)
        self.assertIn("不要实际运动", guidance)

    def test_protocol_messages_resolve_to_exact_visual_stages(self) -> None:
        cases = {
            "FIXATION": (MiVisualStage.FIXATION, None),
            "PROMPT HAND LEFT": (MiVisualStage.HAND_CUE, "left"),
            "PROMPT HAND RIGHT": (MiVisualStage.HAND_CUE, "right"),
            "← LEFT": (MiVisualStage.MOTOR_IMAGERY, "left"),
            "→ RIGHT": (MiVisualStage.MOTOR_IMAGERY, "right"),
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                frame = resolve_mi_visual(message)
                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual((frame.stage, frame.label), expected)

    def test_hand_cue_uses_isolated_animation_and_mirrors_left(self) -> None:
        frame = resolve_mi_visual("PROMPT HAND LEFT")
        assert frame is not None
        rendered = frame_html(frame)

        self.assertTrue(HAND_ANIMATION_PATH.is_file())
        self.assertEqual(
            HAND_ANIMATION_PATH.name,
            "hand-grasp-recorded-v2-30fps.webp",
        )
        self.assertIn("data:image/webp;base64,", rendered)
        encoded_match = re.search(r"base64,([^')]+)", rendered)
        self.assertIsNotNone(encoded_match)
        assert encoded_match is not None
        self.assertEqual(
            base64.b64decode(encoded_match.group(1)),
            HAND_ANIMATION_PATH.read_bytes(),
        )
        self.assertIn("mi-hand--left", rendered)
        self.assertNotIn("<svg", rendered)
        self.assertNotIn("🤛", rendered)
        self.assertNotIn("🤜", rendered)

    def test_hand_animation_has_sixty_recorded_frames(self) -> None:
        with Image.open(HAND_ANIMATION_PATH) as animation:
            self.assertEqual(animation.n_frames, 60)
            self.assertEqual(animation.size, (512, 512))
        encoded = HAND_ANIMATION_PATH.read_bytes()
        frame_offsets: list[int] = []
        offset = 0
        while (offset := encoded.find(b"ANMF", offset)) >= 0:
            frame_offsets.append(offset)
            offset += 4
        durations = [
            int.from_bytes(encoded[offset + 20 : offset + 23], "little")
            for offset in frame_offsets
        ]
        expected_durations = [
            round((index + 1) * 1000 / 30) - round(index * 1000 / 30)
            for index in range(60)
        ]
        self.assertEqual(durations, expected_durations)
        self.assertEqual(sum(durations), 2_000)
        self.assertIn("background-size: contain", STIMULUS_CSS)
        self.assertNotIn("background-size: 400% 200%", STIMULUS_CSS)
        self.assertNotIn("@keyframes mi-hand-sprite", STIMULUS_CSS)

    def test_formal_palette_is_black_with_green_cross_and_arrow(self) -> None:
        self.assertEqual(MI_BACKGROUND, "#000000")
        self.assertEqual(MI_GREEN, "#39FF14")
        self.assertIn("background: #000000", STIMULUS_CSS)
        self.assertIn("color: #39FF14", STIMULUS_CSS)
        self.assertIn("font-size: clamp(12rem, 26vw, 25rem)", STIMULUS_CSS)
        self.assertIn("transform: translateY(-0.08em)", STIMULUS_CSS)

    def test_formal_surface_keeps_animation_dom_stable_between_poll_reruns(self) -> None:
        component_dir = Path(__file__).resolve().parents[1] / "components" / "stimulus_surface"
        component_html = (component_dir / "index.html").read_text(encoding="utf-8")
        component_animation = component_dir / HAND_ANIMATION_PATH.name

        self.assertTrue(component_animation.is_file())
        self.assertEqual(component_animation.read_bytes(), HAND_ANIMATION_PATH.read_bytes())
        self.assertIn("if (signature === currentSignature) return;", component_html)
        self.assertIn("restartHand(label);", component_html)
        self.assertIn("clamp(12rem, 26vw, 25rem)", component_html)
        self.assertIn("transform: translateY(-0.08em)", component_html)
        self.assertIn("requestFullscreen", component_html)
        self.assertIn("进入电脑全屏", component_html)
        self.assertIn("fullscreenchange", component_html)
        self.assertIn('id="trial-progress-page"', component_html)
        self.assertIn("overflow-y: auto", component_html)
        self.assertIn("scrollbar-width: none", component_html)
        self.assertIn("updateTrialProgress(args);", component_html)
        self.assertIn('streamlit:setComponentValue", { value: true', component_html)
        self.assertLess(
            component_html.index("updateTrialProgress(args);"),
            component_html.index("if (signature === currentSignature) return;"),
        )
        self.assertNotIn("data:image/webp;base64", component_html)


if __name__ == "__main__":
    unittest.main()
