"""Streamlit smoke coverage for the operator-facing car workflow."""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest


class GuiSmokeTests(unittest.TestCase):
    def test_sidebar_has_no_impedance_page(self) -> None:
        import gui

        self.assertNotIn("阻抗检查", gui.SIDEBAR_NAV_PAGES)
        self.assertFalse(hasattr(gui, "render_impedance"))

    def test_hardware_free_rehearsal_is_short_and_isolated(self) -> None:
        import gui

        original = {
            "subject_id": "S001",
            "device_type": "neuracle",
            "hardware_dummy_mode": False,
            "protocol": {
                "collection_blocks": 9,
                "collection_trials_per_class_per_block": 50,
                "rest_between_blocks_sec": 180.0,
            },
        }

        rehearsal = gui._hardware_free_rehearsal_config(original)

        self.assertEqual(original["device_type"], "neuracle")
        self.assertFalse(original["hardware_dummy_mode"])
        self.assertEqual(rehearsal["device_type"], "dummy")
        self.assertTrue(rehearsal["hardware_dummy_mode"])
        self.assertTrue(rehearsal["collection_rehearsal"])
        self.assertEqual(rehearsal["subject_id"], "S001-rehearsal")
        self.assertEqual(rehearsal["protocol"]["collection_blocks"], 2)
        self.assertEqual(
            rehearsal["protocol"]["collection_trials_per_class_per_block"],
            2,
        )
        self.assertEqual(rehearsal["protocol"]["rest_between_blocks_sec"], 3.0)

    def test_trial_test_previews_each_hand_once(self) -> None:
        import gui

        protocol = gui.ProtocolConfig.from_config(
            {
                "window_sec": 4.0,
                "protocol": {
                    "trial_timing": {
                        "fixation_sec": 2.0,
                        "cue_sec": 2.0,
                        "control_sec": 4.0,
                    }
                },
            }
        )
        shown_trials: list[str] = []
        with (
            patch.object(gui, "init_live_view", return_value=(object(), lambda: None)),
            patch.object(gui, "_run_preview_event") as preview_event,
            patch.object(
                gui,
                "_run_visual_trial",
                side_effect=lambda *_args, **kwargs: shown_trials.append(
                    str(kwargs["label"])
                ),
            ),
        ):
            gui.run_collection_trial_test(protocol)

        self.assertEqual(shown_trials, ["left", "right"])
        preview_event.assert_not_called()

    def test_trial_preview_redraws_fixation_hand_and_arrow(self) -> None:
        import gui

        events: list[dict[str, object]] = []
        timing = gui.TrialTiming(
            fixation_sec=2.0,
            cue_sec=2.0,
            control_sec=4.0,
        )
        with patch.object(
            gui,
            "_run_preview_event",
            side_effect=lambda *_args, **kwargs: events.append(kwargs),
        ):
            gui._run_visual_trial(
                object(),
                lambda: None,
                label="left",
                timing=timing,
                trial_number="正式左手 trial",
            )

        self.assertEqual(
            [(event["message"], event["duration_sec"]) for event in events],
            [
                ("FIXATION", 2.0),
                ("PROMPT HAND LEFT", 2.0),
                ("← LEFT", 4.0),
            ],
        )

    def test_only_protocol_cues_can_change_subject_symbol(self) -> None:
        import gui

        self.assertEqual(gui._resolve_cue_symbol("← LEFT", event_type="cue"), ("←", False))
        self.assertEqual(gui._resolve_cue_symbol("→ RIGHT", event_type="cue"), ("→", False))
        self.assertIsNone(gui._resolve_cue_symbol("PRACTICE → RIGHT", event_type="cue"))
        self.assertEqual(gui._resolve_cue_symbol("PROMPT HAND LEFT", event_type="cue"), ("手", False))
        self.assertEqual(gui._resolve_cue_symbol("FIXATION", event_type="cue"), ("+", False))
        self.assertIsNone(gui._resolve_cue_symbol("○", event_type="cue"))
        self.assertEqual(gui._resolve_cue_symbol("想象左手重复握拳、松开", event_type="log"), None)
        self.assertEqual(gui._resolve_cue_symbol("Block 1/4 共 15 个 trial", event_type="log"), ("", False))

    def test_batched_stage_updates_render_only_the_latest_symbol(self) -> None:
        import gui

        placeholder = type("Placeholder", (), {"markdown": lambda *args, **kwargs: None, "code": lambda *args, **kwargs: None})()
        console = gui.StreamlitConsole(placeholder, placeholder, fullscreen=True)
        console._ui_thread_id = threading.get_ident() + 1
        rendered: list[tuple[str, bool]] = []
        console._render_cue = lambda msg, *, prediction: rendered.append((msg, prediction))

        console.print("← LEFT")
        console.print("FIXATION")
        console.render_pending()

        self.assertEqual(rendered, [("FIXATION", False)])
        self.assertEqual(console.logs[-2:], ["← LEFT", "FIXATION"])

    def test_gui_opens_and_realtime_page_exposes_car_recovery(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        app = AppTest.from_file(str(gui_path), default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        app.button(key="nav_btn_实时解码").click().run()

        self.assertEqual(list(app.exception), [])
        buttons = {button.key: button.label for button in app.button}
        self.assertEqual(buttons.get("ar_test_open_car"), "启动/重置并进入小车")
        self.assertIn("开始实时解码", buttons.values())
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("状态"), "等待启动")
        self.assertEqual(metrics.get("更新次数"), "0")
        self.assertEqual(metrics.get("缓冲训练时长"), "0s")

        app.button(key="nav_btn_数据采集").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.radio), [])
        self.assertFalse(any("正式 trial" in info.value for info in app.info))
        self.assertFalse(any("block 间休息" in info.value for info in app.info))
        self.assertFalse(any("可继续追加采集" in info.value for info in app.info))
        calibration_buttons = {button.label for button in app.button}
        self.assertIn("进入正式采集流程", calibration_buttons)
        self.assertIn("画面测试", calibration_buttons)
        self.assertIn("无硬件演练", calibration_buttons)

        app.button(key="nav_btn_设置").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertNotIn(
            "持续采集，人工结束",
            {checkbox.label for checkbox in app.checkbox},
        )
        self.assertNotIn(
            "允许结束前的最少训练窗口秒数",
            {number_input.label for number_input in app.number_input},
        )
        settings_metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(settings_metrics.get("单个 trial"), "2 + 2 + 4 秒")
        self.assertEqual(settings_metrics.get("会话结构"), "9 × 100 trial")
        self.assertEqual(settings_metrics.get("类别"), "左手 / 右手")

    def test_formal_collection_requires_guidance_then_ready_confirmation(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        app = AppTest.from_file(str(gui_path), default_timeout=30).run()

        app.button(key="nav_btn_数据采集").click().run()
        next(
            button for button in app.button if button.label == "进入正式采集流程"
        ).click().run()
        self.assertEqual(list(app.exception), [])

        for _ in range(4):
            app.button(key="collection_guidance_next").click().run()
            self.assertEqual(list(app.exception), [])

        buttons = {button.key: button.label for button in app.button}
        self.assertEqual(buttons.get("collection_start_formal"), "开始正式采集")
        self.assertNotIn("calibration_pause", buttons)
        self.assertNotIn("calibration_finish", buttons)

    def test_hardware_free_rehearsal_reaches_the_real_collection_gate(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        app = AppTest.from_file(str(gui_path), default_timeout=30).run()

        app.button(key="nav_btn_数据采集").click().run()
        next(button for button in app.button if button.label == "无硬件演练").click().run()

        self.assertEqual(list(app.exception), [])
        buttons = {button.key: button.label for button in app.button}
        self.assertEqual(buttons.get("collection_start_formal"), "开始无硬件演练")
        rendered_text = "\n".join(markdown.value for markdown in app.markdown)
        self.assertIn("2 个 block，共 8 个 trial", rendered_text)

    def test_hardware_free_collection_pauses_discards_and_recollects(self) -> None:
        import gui

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "subject_id": "integration-rehearsal",
                "device_type": "dummy",
                "hardware_dummy_mode": True,
                "sfreq": 200.0,
                "window_sec": 0.5,
                "step_sec": 0.5,
                "buffer_sec": 10.0,
                "protocol": {
                    "collection_stride_sec": 0.5,
                    "motor_imagery_start_offset_sec": 0.0,
                    "motor_imagery_stop_offset_sec": 0.5,
                    "collection_blocks": 1,
                    "collection_trials_per_class_per_block": 1,
                    "rest_between_blocks_sec": 0.0,
                    "random_seed": 17,
                },
                "device": {
                    "dummy_label_aware": False,
                    "dummy_source_sfreq": 250,
                    "dummy_eeg_channels": 59,
                },
                "storage": {
                    "runtime_dir": str(root / "runtime"),
                    "records_dir": str(root / "records"),
                },
            }
            protocol = gui.ProtocolConfig.from_config(config)
            # Keep the production config fixed at 2-2-4; shorten only this
            # in-memory test plan so the integration test finishes quickly.
            protocol.trial_timing = gui.TrialTiming(0.15, 0.15, 0.5)
            protocol.window_sec = 0.5
            protocol.stride_sec = 0.5
            protocol.motor_imagery_start_offset_sec = 0.0
            protocol.motor_imagery_stop_offset_sec = 0.5
            control = gui.CollectionPauseControl()
            messages: list[str] = []
            console = type(
                "Console",
                (),
                {
                    "print": lambda self, message, *args, **kwargs: messages.append(str(message)),
                    "set_stage_progress": lambda self, **kwargs: None,
                },
            )()
            outcome_box: list[dict[str, object]] = []
            worker = threading.Thread(
                target=lambda: outcome_box.append(
                    gui.run_collection_session(
                        config,
                        protocol,
                        console=console,
                        pause_control=control,
                    )
                )
            )
            worker.start()
            time.sleep(0.35)
            control.request_pause()
            deadline = time.monotonic() + 3.0
            while not control.paused and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(control.paused)
            control.resume()
            worker.join(timeout=10.0)

            self.assertFalse(worker.is_alive())
            self.assertEqual(len(outcome_box), 1)
            self.assertTrue(outcome_box[0]["ok"], outcome_box[0])
            session_dir = Path(str(outcome_box[0]["session_dir"]))
            events = json.loads((session_dir / "events.json").read_text(encoding="utf-8"))
            event_names = [event["name"] for event in events]
            self.assertIn("trial_discarded", event_names)
            self.assertIn("manual_pause_start", event_names)
            self.assertIn("manual_pause_end", event_names)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["formal_trial_count"], 2)
            self.assertEqual(metadata["source_sfreq"], 250.0)
            self.assertEqual(metadata["sfreq"], 200.0)
            self.assertEqual(metadata["n_channels"], 59)
            self.assertEqual(len(metadata["channel_names"]), 59)
            self.assertNotIn("ECG", metadata["channel_names"])
            expected_starts = np.asarray(
                [trial["motor_imagery_on_sample"] for trial in metadata["trials"]],
                dtype=np.int64,
            )
            with np.load(session_dir / "mi_windows.npz") as windows:
                self.assertEqual(windows["raw_windows"].shape[1:], (59, 100))
                self.assertEqual(windows["processed_windows"].shape[1:], (59, 100))
                self.assertTrue(np.all(windows["source_sfreq"] == 250.0))
                self.assertTrue(np.all(windows["sfreq"] == 200.0))
                np.testing.assert_array_equal(
                    windows["window_stop_samples"] - windows["window_start_samples"],
                    np.full(windows["window_start_samples"].shape, 125, dtype=np.int64),
                )
                self.assertTrue(
                    np.all(np.isin(windows["window_start_samples"], expected_starts))
                )
                np.testing.assert_array_equal(
                    windows["window_offsets_sec"],
                    np.zeros(windows["window_start_samples"].shape[0], dtype=np.float32),
                )

    def test_fixed_collection_run_has_no_legacy_operator_controls(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        source = gui_path.read_text(encoding="utf-8")

        self.assertIn("if not is_running:", source)
        self.assertIn("render_experiment_return_button(", source)
        self.assertIn('key="collection_return_from_experiment"', source)
        self.assertIn('key="test_mode_return_from_experiment"', source)
        self.assertIn('is_running = collection_view == "run"', source)
        self.assertNotIn('key="calibration_pause"', source)
        self.assertNotIn('key="calibration_resume"', source)
        self.assertNotIn('key="calibration_finish"', source)
        self.assertIn("_start_collection_worker", source)
        self.assertIn("collection_stimulus_surface_epoch", source)
        self.assertIn('key=f"collection_stimulus_surface_{surface_epoch}"', source)
        self.assertIn("_render_computer_fullscreen_control()", source)
        self.assertIn('key="collection_computer_fullscreen_control"', source)
        self.assertIn(".st-key-collection_computer_fullscreen_control", source)
        self.assertIn("result = collector.collect(", source)
        self.assertIn('key="collection_request_pause"', source)
        self.assertIn('key="collection_resume"', source)
        self.assertNotIn("train_after_collection", source)
        self.assertIn("st.session_state.collection_last_outcome = outcome", source)
        self.assertIn('st.session_state.gui_nav_mode = "数据采集"', source)
        self.assertIn("st.rerun()", source)

    def test_running_collection_renders_surface_pause_and_return_together(self) -> None:
        script = r'''
import streamlit as st
from types import SimpleNamespace
import gui
from adaptation.calibrator import CollectionPauseControl
from utils.binary_mi_gui import MiVisualFrame, MiVisualStage

console = gui.StreamlitConsole(st.empty(), st.empty(), fullscreen=True, stable_surface=True)
console._fullscreen_frame = MiVisualFrame(MiVisualStage.HAND_CUE, label="left")
console.render_pending = lambda: None
console.attach = lambda *args, **kwargs: None
handle = SimpleNamespace(
    console=console,
    pause_control=CollectionPauseControl(),
    outcome=lambda: None,
)
gui._get_collection_worker = lambda config: handle
gui.time.sleep = lambda seconds: None
gui.st.rerun = lambda: None
gui._render_running_collection({}, SimpleNamespace())
'''
        app = AppTest.from_string(script, default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        buttons = {button.key: button.label for button in app.button}
        self.assertEqual(buttons["collection_return_from_running"], "≪")
        self.assertEqual(buttons["collection_request_pause"], "我要休息")
        components = [node for node in app._tree if node.type == "component_instance"]
        self.assertEqual(len(components), 1)
        self.assertIn('"stage": "hand_cue"', components[0].proto.json_args)
        self.assertIn('"label": "left"', components[0].proto.json_args)
        progress_markup = [
            node.value
            for node in app.markdown
            if "oi-collection-trial-progress" in node.value
        ]
        self.assertEqual(len(progress_markup), 1)
        self.assertIn("已完成 0 / 900 个有效 trial", progress_markup[0])
        gui_source = (Path(__file__).resolve().parents[1] / "gui.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("margin: calc(100dvh + 2rem) auto 4rem", gui_source)

    def test_completed_collection_is_recovered_after_browser_disconnect(self) -> None:
        import gui

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "subject_id": "S001",
                "storage": {
                    "runtime_dir": str(root / "runtime"),
                    "records_dir": str(root / "records"),
                },
            }

            session_dir = (
                root
                / "records"
                / "S001"
                / "collection"
                / "session_test"
            )
            session_dir.mkdir(parents=True)
            (session_dir / "continuous_eeg.npy").write_bytes(b"eeg")
            (session_dir / "events.json").write_text("[]", encoding="utf-8")
            (session_dir / "mi_windows.npz").write_bytes(b"windows")
            metadata = {
                "trials_collected": 284,
                "windows_collected": 284,
            }

            gui._write_collection_status(
                config,
                {"state": "running", "session_id": "session_test"},
            )
            (session_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            recovered = gui._recover_completed_collection(config)

            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertTrue(recovered["ok"])
            self.assertTrue(recovered["recovered_after_reconnect"])
            self.assertEqual(recovered["trials_collected"], 284)
            self.assertEqual(recovered["windows_collected"], 284)
            self.assertEqual(
                gui._read_collection_status(config)["state"],
                "completed",
            )

    def test_collection_success_requires_saved_data(self) -> None:
        import gui

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_dir = root / "session"
            session_dir.mkdir()
            (session_dir / "metadata.json").write_text("{}", encoding="utf-8")
            eeg_path = session_dir / "continuous_eeg.npy"
            events_path = session_dir / "events.json"

            with self.assertRaisesRegex(RuntimeError, "连续 EEG"):
                gui._validate_collection_outcome(
                    {
                        "continuous_eeg_path": str(eeg_path),
                        "events_path": str(events_path),
                        "windows_path": str(session_dir / "mi_windows.npz"),
                        "session_dir": str(session_dir),
                    }
                )

    def test_collection_success_does_not_require_model_files(self) -> None:
        import gui

        with TemporaryDirectory() as temporary:
            session_dir = Path(temporary) / "session"
            session_dir.mkdir()
            (session_dir / "metadata.json").write_text("{}", encoding="utf-8")
            eeg_path = session_dir / "continuous_eeg.npy"
            eeg_path.write_bytes(b"eeg")
            events_path = session_dir / "events.json"
            events_path.write_text("[]", encoding="utf-8")
            windows_path = session_dir / "mi_windows.npz"
            windows_path.write_bytes(b"windows")

            gui._validate_collection_outcome(
                {
                    "continuous_eeg_path": str(eeg_path),
                    "events_path": str(events_path),
                    "windows_path": str(windows_path),
                    "session_dir": str(session_dir),
                }
            )


if __name__ == "__main__":
    unittest.main()
