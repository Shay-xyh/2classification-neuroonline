# 当前数据采集协议

本文件是正式数据采集的唯一口径。采集流程不启动、加载或更新任何模型；训练与评价属于采集完成后的独立工作。

## 0. 采集信号

- 正式采集输入：250 Hz、59 个纯 EEG 通道。
- 排除辅助通道：ECG、HEOR、HEOL、VEOU、VEOL。
- 保留 250 Hz 连续原始 EEG 与相应事件样本号；采后预处理并降采样到 200 Hz。
- 每个有效 trial 从运动想象开始点切取完整 4 秒窗，即源数据 1000 点、处理后 800 点。
- 无硬件模式使用完全相同的采样率和通道边界。

## 1. 固定 trial 流程

每个 trial 固定 8 秒，时间轴不可在 GUI、CLI 或配置文件中修改：

1. `0–2 s`：屏幕中央显示绿色注视十字。被试可眨眼或做轻微调整，不赋分类标签。
2. `2–4 s`：播放左手或右手抓握动作提示。该阶段只提示下一任务，不进入带标签 MI 数据。
3. `4–8 s`：显示对应方向的绿色箭头。被试按提示节奏持续重复想象对应手“握拳—松开”，约两轮，但不得实际运动。

箭头消失即表示本 trial 结束，随后立即显示下一 trial 的注视十字。trial 内没有其他阶段或空白间隔。

## 2. 单次采集结构

- 正式类别固定为 `left=0`、`right=1`，不存在静息类别。
- 单次正式采集共 900 个有效 trial，分成 9 个 block。
- 每个 block 有 100 个有效 trial，其中左手 50 个、右手 50 个并伪随机排列。
- 每完成 100 个有效 trial 自动休息 180 秒，共 8 次自动休息。
- 纯 trial 时长为 900 × 8 秒，即 120 分钟；加上自动休息后，计划总运行时间为 144 分钟，手动休息会继续延长总时长。
- 自动休息和手动休息期间连续 EEG 仍然采集，但不产生左右手分类标签。

## 3. 手动暂停

- 被试可在正式采集中点击“我要休息”。
- 如果按钮在 trial 进行中生效，该次 attempt 立即作废并记录 `trial_discarded`。
- 暂停结束后重新采集同一个计划 trial；`trial_index` 不变，`attempt_index` 加一。
- 被丢弃 attempt 的连续 EEG 和事件仍保留用于质量审计，但不会进入有效 trial 列表或 MI 窗口。

## 4. 事件与标签

所有事件按连续 EEG 的 `sample_index` 对齐。事件文件不保存墙钟时间或绝对日期时间。
本实验不使用外置 TriggerBox，也不发送硬件触发码。事件编号仅作为软件协议标识保存在
`events.json` 中；采集程序利用 JellyFish 设备包时间与电脑单调时钟的近期下包络关系，
将事件投影到 250 Hz 连续 EEG 样本号。近期窗口会跟踪两套时钟的缓慢漂移，固定传输
延迟则只在经过独立测量后通过 `neuracle_transport_delay_sec` 作一次性补偿。

| 事件 | 编码 | 含义 |
|---|---:|---|
| `session_start` | 101 | 正式 session 开始 |
| `session_end` | 102 | 正式 session 正常结束 |
| `block_start` | 120 | 一个 100-trial block 开始 |
| `block_end` | 121 | 一个 100-trial block 结束 |
| `automatic_break_start` | 122 | 180 秒自动休息开始 |
| `automatic_break_end` | 123 | 180 秒自动休息结束 |
| `fixation_on` | 130 | 2 秒注视阶段开始 |
| `cue_left_on` | 131 | 左手动作动画开始 |
| `cue_right_on` | 132 | 右手动作动画开始 |
| `motor_imagery_left_on` | 134 | 左手 4 秒 MI 与有效标签开始 |
| `motor_imagery_right_on` | 135 | 右手 4 秒 MI 与有效标签开始 |
| `motor_imagery_off` | 136 | 当前 4 秒 MI 结束 |
| `manual_pause_start` | 140 | 手动休息开始 |
| `manual_pause_end` | 141 | 手动休息结束 |
| `trial_discarded` | 142 | 当前 attempt 作废 |

只有 `motor_imagery_left_on` 和 `motor_imagery_right_on` 开始后的完整 4 秒区间具有分类标签。提示动画事件虽然包含方向信息，但不是有效分类区间。

## 5. 采集与采后处理顺序

1. 采集中持续接收 250 Hz、59 通道连续 EEG，并按源数据 `sample_index` 记录事件；不实时运行模型、预处理或切窗。
2. 每个有效 trial 完成后由后台线程增量保存新增 EEG、事件和进度检查点；自动休息及手动暂停期间每 10 秒增量保存一次。检查点不包含墙钟或绝对时间。
3. `session_end` 后停止设备流并原子生成 `continuous_eeg.npy`、`events.json` 和初始 `metadata.json`。
4. 对整段连续数据统一处理：非有限值修复、去除逐通道直流、坏通道检测与修复、平均参考、250→200 Hz 重采样和带通滤波。
5. 将每个有效 trial 的 `motor_imagery_on_sample` 映射到 200 Hz 连续处理结果，从该点切取完整 4 秒。
6. 对每个窗口执行质量判定和数值裁剪；不合格窗口计入质量报告，但不进入最终有效窗口集合。
7. 保存 `mi_windows.npz`，最后计算文件校验和并把完整性状态写回 `metadata.json`。

因此，采集线程不会实时切窗或实时预处理；预处理和切窗都在刺激会话结束后完成。处理顺序是“先对整段连续信号统一预处理，再按事件边界切窗”，避免对每个独立 4 秒片段滤波造成边缘伪迹。

## 6. 保存文件

```text
records_storage/<subject>/collection/<session_id>/
├── continuous_eeg.npy
├── events.json
├── metadata.json
├── mi_windows.npz
└── checkpoint.json
```

- `continuous_eeg.npy`：未经采后预处理的 250 Hz、59 通道连续 EEG。
- `events.json`：事件名称、源数据样本号和协议上下文；不包含绝对时间。
- `metadata.json`：采样率、通道名、trial 清单、预处理参数、质量统计、丢包诊断和文件校验和。
- `mi_windows.npz/raw_windows`：逐通道去直流并降采样到 200 Hz、尚未完成完整滤波的 4 秒窗口。
- `mi_windows.npz/processed_windows`：完成坏通道处理、平均参考、降采样、带通滤波、质量控制和裁剪后的 4 秒窗口。
- `mi_windows.npz/window_start_samples` 与 `window_stop_samples`：窗口在 250 Hz 连续源数据中的边界，每个完整窗口相差 1000 点。
- `checkpoint.json`：增量保存状态和已持久化有效 trial 数；不包含绝对时间。

整个正式采集入口不创建模型对象，不读取模型权重，也不调用训练、推理或在线更新代码。采后训练必须由独立命令显式启动。

输出结构和时间映射公式见 [数据文件与时间对齐](docs/DATA_FORMAT.md)，实验当天的操作顺序
见 [正式实验操作清单](docs/FORMAL_COLLECTION.md)。
