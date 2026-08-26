# 数据文件与时间对齐

## 1. Session目录

每次正常完成的正式采集写入：

```text
records_storage/<subject>/collection/<session_id>/
├── continuous_eeg.npy
├── events.json
├── metadata.json
└── mi_windows.npz
```

正式采集不创建模型文件。

## 2. 连续EEG

`continuous_eeg.npy` 为 `float32` NumPy数组：

```text
shape = (59, 连续源样本数)
source_sfreq = 250 Hz
```

它只包含59个头皮EEG通道，不包含 `ECG/HEOR/HEOL/VEOU/VEOL` 或硬件触发通道。自动休息、
手动休息和作废attempt期间的连续EEG仍在其中。

## 3. 事件文件

`events.json` 是事件对象数组。典型事件：

```json
{
  "name": "motor_imagery_left_on",
  "sample_index": 25008,
  "payload": {
    "event_code": 134,
    "block_index": 0,
    "trial_index": 12,
    "attempt_index": 0,
    "label": "left",
    "label_id": 0,
    "alignment_method": "source-clock-projection"
  }
}
```

事件文件不保存日期、北京时间、Unix时间或其他绝对时间戳。`event_code` 只是内部协议ID，
不会发送到外部硬件。

完整事件表见 [正式实验协议](../CURRENT_EXPERIMENT_PROTOCOL.md)。

## 4. 时间对齐链路

JellyFish数据包提供设备端毫秒时间和250Hz数据。项目同时记录数据包抵达电脑时的
`time.monotonic()`，再建立设备时间到电脑单调时间的映射。

对于一个数据包：

```text
device_end = device_start + sample_count / 250
observed_offset = arrival_monotonic - device_end
```

项目使用最近120秒中最低的 `observed_offset` 作为当前时钟偏移。排队和线程调度只会让包
更晚到达，因此近期最低值用于抑制正向排队抖动；滚动窗口同时允许映射跟踪设备与电脑
时钟的缓慢漂移。

映射后的包末端时间为：

```text
mapped_end = device_end + clock_offset - transport_delay
```

收到新数据后，记录器维护：

- 已追加的连续样本数 `sample_count`；
- 最近源数据块末端的电脑单调时间 `latest_sample_end`。

事件发生时取得 `event_time = time.monotonic()`，并计算：

```text
event_sample_index = sample_count
                   + round((event_time - latest_sample_end) × 250)
```

因此网络包尚未到达时，事件也可以投影到即将到来的源样本位置。250Hz的样本间隔为4ms，
单纯四舍五入的量化误差最多约2ms。

## 5. 固定延迟、抖动和漂移

- 固定传输延迟：每个事件产生相同样本偏移，不会随trial累计；只能通过独立测量确定。
- 队列抖动：每个数据包变化，由近期下包络尽量排除。
- 时钟漂移：设备与电脑晶振速率差可能随长时间累计；由120秒滚动估计持续跟踪。
- 浏览器显示延迟：事件对齐的是程序提交刺激切换的时刻，不是显示器实际发光时刻。

没有光电二极管、共享硬件时钟或其他独立测量时，软件无法自动确定绝对固定传输延迟，
因此默认 `neuracle_transport_delay_sec: 0.0`。

## 6. MI窗口

每个成功trial只使用 `motor_imagery_left_on` 或 `motor_imagery_right_on`：

```text
源窗口 = [motor_imagery_on_sample, motor_imagery_on_sample + 1000)
```

1000点对应250Hz下4秒。项目先对完整连续记录统一处理，再映射到200Hz并切取800点：

```text
target_start = round(source_start × 200 / 250)
target_window = 59 × 800
```

`motor_imagery_off` 用于审计，窗口长度以开始样本加固定1000点为准。作废attempt不会进入
有效trial列表，也不会产生MI窗口。

## 7. `mi_windows.npz`

主要数组：

| 数组 | 含义 |
|---|---|
| `raw_windows` | 去直流并降到200Hz、尚未完成完整滤波的窗口 |
| `processed_windows` | 完成坏通道修复、平均参考、滤波与质量控制的窗口 |
| `labels` | 左手0、右手1 |
| `trial_ids` | trial分组编号 |
| `window_start_samples` | 250Hz连续源数据中的起点 |
| `window_stop_samples` | 250Hz连续源数据中的终点，和起点相差1000 |
| `quality_rejected_windows` | 被质量控制拒绝的窗口数 |

如果质量控制拒绝窗口，最终 `processed_windows` 数量可能少于900，但
`continuous_eeg.npy` 和审计事件仍保留。

## 8. 完整性

`metadata.json` 保存协议、通道、trial、时间诊断、预处理信息、包丢失计数、文件大小和
SHA-256校验和。正常session应满足：

```text
model_activity = none
integrity.status = complete
integrity.packet_loss_count = 0
source_sfreq = 250
sfreq = 200
n_channels = 59
```

`source_packet_loss` 表示JellyFish数据包时间不连续。此时不要仅凭窗口数量判定数据可用，
应结合博睿康原生备份和丢包位置进一步审计。

