"""Streamlit presentation for realtime online-adaptation telemetry."""

from __future__ import annotations

from typing import Any

_DEFAULT_LABELS = ("LEFT", "RIGHT")
_CUE_SYMBOLS = {"left": "←", "right": "→"}


def render_online_cue_panel(status: dict[str, Any] | None, *, ui: Any) -> None:
    """Render the continuous car-scene source of truth."""

    if not isinstance(status, dict) or status.get("source") != "cued-protocol":
        return
    phase = str(status.get("phase", "preparing"))
    raw_label_name = status.get("label_name")
    label_name = (
        "pending"
        if raw_label_name is None
        else str(raw_label_name).strip().lower()
    )
    remaining = float(status.get("phase_remaining_sec", 0.0))
    phase_text = {
        "preparing": "准备",
        "control": "运动想象 / 小车控制",
    }.get(phase, phase)
    if phase == "control":
        prompt = (
            "·"
            if label_name == "pending"
            else f"{_CUE_SYMBOLS.get(label_name, '·')}  {label_name.upper()}"
        )
    else:
        prompt = "·"
    ui.markdown("### 连续统一场景")
    columns = ui.columns(8)
    columns[0].metric("阶段", phase_text)
    columns[1].metric("累计 Scene", int(status.get("scene_number", 0)))
    columns[2].metric("相对真值", label_name.upper())
    columns[3].metric("起始→空路", _lane_transition(status))
    columns[4].metric("Unity 同步", "已同步" if status.get("scene_synced") else "等待")
    columns[5].metric("场景剩余", f"{remaining:.1f}s")
    columns[6].metric("本Scene结果", "已失败" if status.get("scene_failed") else "进行中")
    columns[7].metric(
        "横向控制",
        (
            "等待 Scene"
            if phase == "preparing"
            else "采集主决策窗"
            if status.get("lateral_control_gate_active")
            else "已放行"
        ),
    )
    sync_error = status.get("scene_sync_error")
    if sync_error:
        ui.error(f"Unity 场景同步失败：{sync_error}")
    timing = status.get("timing_alignment")
    if isinstance(timing, dict) and timing:
        jitter_ms = float(timing.get("queueing_jitter_sec", 0.0)) * 1000.0
        compensation_ms = (
            float(timing.get("transport_delay_compensation_sec", 0.0)) * 1000.0
        )
        ui.caption(
            "脑电源时钟已对齐："
            f"当前排队抖动 {jitter_ms:.1f} ms，"
            f"固定延迟补偿 {compensation_ms:.1f} ms。"
        )
    ui.caption(
        "LEFT/RIGHT 均以 Scene 开始时小车实际车道为参照；"
        "只有 Unity ACK 同时确认起始车道、空路和相对动作后才产生训练标签；"
        "每个 Scene 采集一个因果主决策窗，完成后才放行横向控制。"
    )
    ui.markdown(f"<div style='text-align:center;font-size:5rem'>{prompt}</div>", unsafe_allow_html=True)


def _lane_transition(status: dict[str, Any]) -> str:
    names = {-1: "左", 0: "中", 1: "右"}
    try:
        start_lane = int(status.get("start_lane"))
        safe_lane = int(status.get("safe_lane"))
    except (TypeError, ValueError):
        return "等待 ACK"
    return f"{names.get(start_lane, '?')}→{names.get(safe_lane, '?')}"


def render_online_adaptation_panel(adaptation: dict[str, Any] | None, *, ui: Any) -> None:
    """Render the dashboard for the active adaptation strategy, if any."""

    if not isinstance(adaptation, dict) or not adaptation.get("enabled"):
        return
    _render_neuroonline(adaptation, ui=ui)


def _render_neuroonline(adaptation: dict[str, Any], *, ui: Any) -> None:
    ui.markdown("### NeuroOnline 在线适配")
    prequential = adaptation.get("prequential", {}) or {}
    operational = adaptation.get("operational_prequential", {}) or {}
    last_result = adaptation.get("last_result") or {}
    top = ui.columns(7)
    top[0].metric("状态", str(adaptation.get("state", "-")))
    top[1].metric("更新次数", int(adaptation.get("update_count", 0)))
    buffered_seconds = float(adaptation.get("buffered_window_seconds", 0.0))
    top[2].metric(
        "缓冲训练时长",
        f"{buffered_seconds:g}s",
        delta=f"{int(adaptation.get('buffered_windows', 0))} 窗",
        delta_color="off",
    )
    top[3].metric("原始 Acc.", f"{float(prequential.get('accuracy', 0.0)):.3f}")
    top[4].metric("原始 Bal.Acc.", f"{float(prequential.get('balanced_accuracy', 0.0)):.3f}")
    top[5].metric("控制覆盖率", f"{float(operational.get('coverage', 0.0)):.3f}")
    top[6].metric("选择性 Acc.", f"{float(operational.get('selective_accuracy', 0.0)):.3f}")
    ui.caption(f"最近更新耗时 {float(last_result.get('duration_sec', 0.0)):.2f}s")
    if not bool(prequential.get("all_classes_observed", False)):
        ui.info("三类尚未全部出现；固定三类 Bal.Acc. 将未出现类别的召回率按 0 计。")

    progress = float(adaptation.get("progress", 0.0))
    ui.progress(min(max(progress, 0.0), 1.0))
    ui.caption(
        f"累计训练窗口 {float(adaptation.get('seen_labeled_window_seconds', 0.0)):g}s "
        f"({int(adaptation.get('seen_labeled_windows', 0))} 窗) · "
        f"距下次更新 {float(adaptation.get('window_seconds_until_update', 0.0)):g}s "
        f"({int(adaptation.get('samples_until_update', 0))} 窗) · "
        f"下一触发 {float(adaptation.get('next_update_window_seconds', 0.0)):g}s"
    )
    duplicate_rejections = int(adaptation.get("duplicate_windows_rejected", 0))
    stale_rejections = int(adaptation.get("stale_windows_rejected", 0))
    if duplicate_rejections or stale_rejections:
        ui.warning(
            "已阻止重复/过期窗口进入在线训练："
            f"重复 {duplicate_rejections}，时间戳非递增 {stale_rejections}。"
        )

    labels = _labels_for(adaptation, prequential)
    detail_left, detail_right = ui.columns(2)
    with detail_left:
        ui.caption("类别覆盖与累计表现")
        ui.dataframe(_class_rows(adaptation, prequential, labels), hide_index=True, width="stretch")
    with detail_right:
        ui.caption("累计混淆矩阵（行=真实，列=预测）")
        ui.dataframe(_confusion_rows(prequential, labels), hide_index=True, width="stretch")

    history = adaptation.get("update_history", []) or []
    if history:
        ui.caption("更新损失轨迹")
        ui.line_chart(
            history,
            x="update",
            y=["loss", "classification_loss", "consistency_loss"],
            width="stretch",
        )
        chart_left, chart_right = ui.columns(2)
        with chart_left:
            ui.caption("CRM gate 轨迹")
            ui.line_chart(history, x="update", y=["gate_alpha", "gate_beta"], width="stretch")
        with chart_right:
            ui.caption("在线累计性能")
            ui.line_chart(
                history,
                x="update",
                y=["prequential_accuracy", "prequential_balanced_accuracy"],
                width="stretch",
            )

    if last_result:
        ui.success(
            "最近一次更新完成："
            f"loss={float(last_result.get('loss', 0.0)):.4f}，"
            f"classification={float(last_result.get('classification_loss', 0.0)):.4f}，"
            f"consistency={float(last_result.get('consistency_loss', 0.0)):.4f}"
        )


def _labels_for(adaptation: dict[str, Any], prequential: dict[str, Any]) -> tuple[str, ...]:
    counts = adaptation.get("class_counts", {}) or {}
    confusion = prequential.get("confusion_matrix", []) or []
    class_count = max(len(counts), len(confusion), len(_DEFAULT_LABELS))
    return tuple(
        _DEFAULT_LABELS[index] if index < len(_DEFAULT_LABELS) else f"class-{index}"
        for index in range(class_count)
    )


def _class_rows(
    adaptation: dict[str, Any],
    prequential: dict[str, Any],
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    counts = adaptation.get("class_counts", {}) or {}
    per_class = prequential.get("per_class_accuracy", {}) or {}
    return [
        {
            "类别": label,
            "主决策窗口": int(counts.get(str(index), 0)),
            "在线准确率": float(per_class.get(str(index), 0.0)),
        }
        for index, label in enumerate(labels)
    ]


def _confusion_rows(
    prequential: dict[str, Any],
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    confusion = prequential.get("confusion_matrix", []) or []
    rows: list[dict[str, Any]] = []
    for true_index, values in enumerate(confusion):
        true_label = labels[true_index] if true_index < len(labels) else f"class-{true_index}"
        row: dict[str, Any] = {"真实类别": true_label}
        for predicted_index, value in enumerate(values):
            predicted_label = (
                labels[predicted_index]
                if predicted_index < len(labels)
                else f"class-{predicted_index}"
            )
            row[f"预测 {predicted_label}"] = int(value)
        rows.append(row)
    return rows
