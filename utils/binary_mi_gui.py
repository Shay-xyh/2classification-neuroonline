"""Pure binary motor-imagery stimuli used by the Streamlit presentation layer.

The formal task has exactly three visual stages: a green fixation cross on a
black background, a left/right hand grasping animation, and a green direction
arrow.  Keeping this mapping outside ``gui.py`` prevents operator logs and
legacy decoder messages from silently changing the subject-facing stimulus.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import html
from pathlib import Path
import re


MI_BACKGROUND = "#000000"
MI_GREEN = "#39FF14"
MI_FOREGROUND = "#F8FAFC"
HAND_ANIMATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "stimuli"
    / "hand-grasp-recorded-v2-30fps.webp"
)


class MiVisualStage(StrEnum):
    READY = "ready"
    FIXATION = "fixation"
    HAND_CUE = "hand_cue"
    MOTOR_IMAGERY = "motor_imagery"
    BREAK = "break"
    COMPLETE = "complete"
    ERROR = "error"
    BLANK = "blank"


@dataclass(frozen=True, slots=True)
class MiVisualFrame:
    stage: MiVisualStage
    label: str | None = None
    message: str = ""

    @property
    def fallback_symbol(self) -> str:
        if self.stage is MiVisualStage.READY:
            return "+"
        if self.stage is MiVisualStage.FIXATION:
            return "+"
        if self.stage is MiVisualStage.HAND_CUE:
            return "手"
        if self.stage is MiVisualStage.MOTOR_IMAGERY:
            return "←" if self.label == "left" else "→"
        if self.stage is MiVisualStage.COMPLETE:
            return "✓"
        if self.stage is MiVisualStage.ERROR:
            return "✕"
        return ""


GUIDANCE_STEPS: tuple[tuple[MiVisualFrame, str, str], ...] = (
    (
        MiVisualFrame(MiVisualStage.FIXATION),
        "注视十字",
        "绿色十字出现时只注视屏幕中央，等待下一条动作提示。",
    ),
    (
        MiVisualFrame(MiVisualStage.HAND_CUE, label="left"),
        "观察手部动作",
        "手部开合动作出现后，确认接下来要想象左手还是右手；此时先不要开始想象。",
    ),
    (
        MiVisualFrame(MiVisualStage.MOTOR_IMAGERY, label="left"),
        "箭头出现后持续想象约两轮",
        "左箭头对应左手，右箭头对应右手。4 秒内按照刚才动画示范的节奏，持续重复想象相应手完成“握拳—松开”，大约两轮；不要实际运动。",
    ),
    (
        MiVisualFrame(MiVisualStage.BREAK, message="休息"),
        "按 block 完成会话",
        "每个 block 100 个 trial，共 9 个 block；每组完成后自动休息 3 分钟。需要额外休息时可随时按“我要休息”。",
    ),
)


def resolve_mi_visual(message: str) -> MiVisualFrame | None:
    """Resolve one explicit protocol message to a subject-facing frame."""

    normalized = re.sub(r"\s+", " ", message.strip())
    upper = normalized.upper()
    if upper == "FIXATION":
        return MiVisualFrame(MiVisualStage.FIXATION)
    if upper == "PROMPT HAND LEFT":
        return MiVisualFrame(MiVisualStage.HAND_CUE, label="left")
    if upper == "PROMPT HAND RIGHT":
        return MiVisualFrame(MiVisualStage.HAND_CUE, label="right")
    if upper == "← LEFT":
        return MiVisualFrame(MiVisualStage.MOTOR_IMAGERY, label="left")
    if upper == "→ RIGHT":
        return MiVisualFrame(MiVisualStage.MOTOR_IMAGERY, label="right")
    if "开始左右手二分类运动想象采集" in normalized:
        return MiVisualFrame(MiVisualStage.READY)
    if normalized.startswith("休息"):
        return MiVisualFrame(MiVisualStage.BREAK, message="休息")
    if "采集完成" in normalized or "测试结束" in normalized:
        return MiVisualFrame(MiVisualStage.COMPLETE, message="实验结束")
    if "执行失败" in normalized:
        return MiVisualFrame(MiVisualStage.ERROR, message="执行失败")
    if normalized.startswith("Block "):
        return MiVisualFrame(MiVisualStage.BLANK)
    return None


@lru_cache(maxsize=4)
def _versioned_hand_animation_data_uri(
    animation_path: str,
    modified_ns: int,
    size_bytes: int,
) -> str:
    del modified_ns, size_bytes
    encoded = base64.b64encode(Path(animation_path).read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def _hand_animation_data_uri() -> str:
    stat = HAND_ANIMATION_PATH.stat()
    return _versioned_hand_animation_data_uri(
        str(HAND_ANIMATION_PATH),
        stat.st_mtime_ns,
        stat.st_size,
    )


def _hand_animation(label: str) -> str:
    mirrored = " mi-hand--left" if label == "left" else ""
    accessible = "左手开合动作" if label == "left" else "右手开合动作"
    return (
        f"<div class='mi-hand{mirrored}' role='img' aria-label='{accessible}' "
        f"style=\"background-image:url('{_hand_animation_data_uri()}')\"></div>"
    )


def frame_html(frame: MiVisualFrame) -> str:
    """Return the inner HTML for one full-screen stimulus frame."""

    if frame.stage in {MiVisualStage.READY, MiVisualStage.FIXATION}:
        return "<div class='mi-fixation' role='img' aria-label='绿色注视十字'>+</div>"
    if frame.stage is MiVisualStage.HAND_CUE:
        return _hand_animation(frame.label or "right")
    if frame.stage is MiVisualStage.MOTOR_IMAGERY:
        arrow = "←" if frame.label == "left" else "→"
        accessible = "左箭头" if frame.label == "left" else "右箭头"
        return f"<div class='mi-arrow' role='img' aria-label='{accessible}'>{arrow}</div>"
    if frame.stage is MiVisualStage.BREAK:
        return "<div class='mi-state-message'>休息</div>"
    if frame.stage is MiVisualStage.COMPLETE:
        return "<div class='mi-state-message'>实验结束<br><small>请等待工作人员</small></div>"
    if frame.stage is MiVisualStage.ERROR:
        safe = html.escape(frame.message or "执行失败")
        return f"<div class='mi-state-message mi-state-message--error'>{safe}</div>"
    return ""


STIMULUS_CSS = f"""
.oi-experiment-stage {{ background: {MI_BACKGROUND} !important; }}
.oi-guidance-stimulus {{
  position: relative;
  width: min(34rem, 70vw);
  height: min(21rem, 38vh);
  margin: 0 auto 1.5rem;
  overflow: hidden;
  border-radius: 0.5rem;
  background: {MI_BACKGROUND};
}}
.oi-guidance-stimulus .mi-hand {{ width: min(26vh, 22vw); }}
.oi-guidance-stimulus .mi-fixation {{ font-size: clamp(8rem, 18vw, 15rem); }}
.oi-guidance-stimulus .mi-arrow {{ font-size: clamp(8rem, 18vw, 15rem); }}
.oi-guidance-stimulus .mi-state-message {{ font-size: clamp(2rem, 4vw, 4rem); }}
.mi-fixation,
.mi-arrow {{
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: {MI_GREEN};
  font-family: Arial, Helvetica, sans-serif;
  font-weight: 500;
  line-height: 1;
  user-select: none;
}}
.mi-fixation {{ font-size: clamp(12rem, 26vw, 25rem); }}
.mi-arrow {{
  font-size: clamp(12rem, 30vw, 28rem);
  transform: translateY(-0.08em);
}}
.mi-hand {{
  position: absolute;
  left: 50%;
  top: 50%;
  width: min(42vh, 38vw);
  aspect-ratio: 1;
  background-repeat: no-repeat;
  background-size: contain;
  background-position: center;
  transform: translate(-50%, -50%);
}}
.mi-hand--left {{ transform: translate(-50%, -50%) scaleX(-1); }}
.mi-state-message {{
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: {MI_FOREGROUND};
  font-size: clamp(3rem, 7vw, 7rem);
  font-weight: 700;
  text-align: center;
}}
.mi-state-message small {{ margin-top: 1.5rem; font-size: 0.36em; font-weight: 400; }}
.mi-state-message--error {{ color: #fb7185; }}
"""
