"""Fixed binary hand motor-imagery data-collection protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

TASK_PARADIGM = "binary_hand_mi"
TASK_LABELS = ("left", "right")
LABEL_TO_ID = {label: index for index, label in enumerate(TASK_LABELS)}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_DISPLAY = {"left": "LEFT", "right": "RIGHT"}
LABEL_SYMBOL = {"left": "←", "right": "→"}
LABEL_DESCRIPTION = {
    "left": "左手重复抓握运动想象",
    "right": "右手重复抓握运动想象",
}

FIXATION_SEC = 2.0
MOVEMENT_PROMPT_SEC = 2.0
MOTOR_IMAGERY_SEC = 4.0
MOTOR_IMAGERY_WINDOW_SEC = 4.0
MOTOR_IMAGERY_WINDOW_START_SEC = 0.0
MOTOR_IMAGERY_WINDOW_STOP_SEC = 4.0

RECOMMENDED_INSTRUCTIONS = [
    "每个 trial 先注视屏幕中央的绿色十字 2 秒；此时可眨眼或做轻微调整。",
    "随后观察 2 秒左手或右手抓握动作提示，请确认接下来要想象哪只手。",
    "箭头出现后的 4 秒内，按照刚才动画示范的节奏，持续重复想象对应手完成“握拳—松开”，大约两轮；双手不要实际运动。",
    "从动作提示出现到箭头消失，请保持身体、面部和双手不动，并尽量不要眨眼。",
    "注视十字和 block 间休息不属于分类标签；正式标签只有左手和右手。",
]


@dataclass(slots=True)
class TrialTiming:
    fixation_sec: float = FIXATION_SEC
    cue_sec: float = MOVEMENT_PROMPT_SEC
    control_sec: float = MOTOR_IMAGERY_SEC

    @property
    def total_sec(self) -> float:
        return self.fixation_sec + self.cue_sec + self.control_sec


@dataclass(slots=True)
class ProtocolConfig:
    window_sec: float
    stride_sec: float
    motor_imagery_start_offset_sec: float
    motor_imagery_stop_offset_sec: float
    trial_timing: TrialTiming
    collection_blocks: int
    collection_trials_per_class_per_block: int
    rest_between_blocks_sec: float
    random_seed: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProtocolConfig:
        protocol = dict(config.get("protocol", {}))
        collection_blocks = int(protocol.get("collection_blocks", 9))
        collection_trials_per_class_per_block = int(
            protocol.get("collection_trials_per_class_per_block", 50)
        )
        return cls(
            window_sec=MOTOR_IMAGERY_WINDOW_SEC,
            stride_sec=MOTOR_IMAGERY_WINDOW_SEC,
            motor_imagery_start_offset_sec=MOTOR_IMAGERY_WINDOW_START_SEC,
            motor_imagery_stop_offset_sec=MOTOR_IMAGERY_WINDOW_STOP_SEC,
            # The acquisition paradigm is fixed at 2 s + 2 s + 4 s. It is not
            # configurable because changing it would silently change the task.
            trial_timing=TrialTiming(),
            collection_blocks=collection_blocks,
            collection_trials_per_class_per_block=collection_trials_per_class_per_block,
            rest_between_blocks_sec=float(protocol.get("rest_between_blocks_sec", 180.0)),
            random_seed=int(protocol.get("random_seed", 17)),
        )


@dataclass(slots=True)
class SessionPlan:
    subject_mode: str
    blocks: list[list[str]]
    rest_between_blocks_sec: float
    trial_timing: TrialTiming

    @property
    def total_formal_trials(self) -> int:
        return sum(len(block) for block in self.blocks)

    @property
    def total_formal_minutes(self) -> float:
        return self.total_formal_trials * self.trial_timing.total_sec / 60.0


def build_session_plan(protocol: ProtocolConfig) -> SessionPlan:
    """Build one balanced binary hand-MI collection session."""
    rng = random.Random(protocol.random_seed)
    blocks = [
        generate_block_sequence(
            {
                label: protocol.collection_trials_per_class_per_block
                for label in TASK_LABELS
            },
            rng=rng,
        )
        for _ in range(protocol.collection_blocks)
    ]
    return SessionPlan(
        subject_mode="fixed_session",
        blocks=blocks,
        rest_between_blocks_sec=protocol.rest_between_blocks_sec,
        trial_timing=protocol.trial_timing,
    )


def generate_block_sequence(class_counts: dict[str, int], *, rng: random.Random, max_attempts: int = 1000) -> list[str]:
    labels = sorted(class_counts)
    total = sum(int(count) for count in class_counts.values())
    for _ in range(max_attempts):
        remaining = {label: int(count) for label, count in class_counts.items()}
        sequence: list[str] = []
        while len(sequence) < total:
            candidates = []
            for label in labels:
                if remaining[label] <= 0:
                    continue
                if len(sequence) >= 2 and sequence[-1] == label and sequence[-2] == label:
                    continue
                candidates.append(label)
            if not candidates:
                break
            candidates.sort(
                key=lambda label: (
                    _first_half_deficit(sequence, class_counts, label),
                    remaining[label],
                    rng.random(),
                ),
                reverse=True,
            )
            chosen = candidates[0]
            sequence.append(chosen)
            remaining[chosen] -= 1
        if len(sequence) != total:
            continue
        if len(set(sequence[:3])) < 2:
            continue
        if not _is_half_balanced(sequence, class_counts):
            continue
        return sequence
    raise RuntimeError(f"Failed to generate a valid block sequence for counts={class_counts}")


def _first_half_deficit(sequence: list[str], class_counts: dict[str, int], candidate: str) -> float:
    midpoint = sum(class_counts.values()) // 2
    if len(sequence) >= midpoint:
        return 0.0
    target = class_counts[candidate] / 2.0
    return target - sequence[:midpoint].count(candidate)


def _is_half_balanced(sequence: list[str], class_counts: dict[str, int]) -> bool:
    midpoint = len(sequence) // 2
    first = sequence[:midpoint]
    second = sequence[midpoint:]
    for label, count in class_counts.items():
        target = count / 2.0
        if abs(first.count(label) - target) > 1:
            return False
        if abs(second.count(label) - target) > 1:
            return False
    return True
