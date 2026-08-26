# 正式采集配置参考

正式运行默认读取项目根目录的 `config.yaml`。也可以显式指定：

```powershell
.\.venv\Scripts\python.exe cli.py --config D:\configs\subject.yaml gui
```

本页只说明Neuracle正式采集相关参数。模型、BrainCo、Unity和在线适配参数不会进入正式
采集线程，不应在实验当天调整。

## 顶层参数

| 参数 | 正式值 | 含义 |
|---|---:|---|
| `subject_id` | 每名被试不同 | 输出目录中的被试编号 |
| `device_type` | `neuracle` | 使用Neuracle/JellyFish适配器 |
| `hardware_dummy_mode` | `false` | `true` 时强制改用模拟EEG |
| `sfreq` | `200` | 采后目标采样率，不是硬件采样率 |
| `task_paradigm` | `binary_hand_mi` | 左右手二分类范式 |
| `n_classes` | `2` | 左手和右手 |
| `window_sec` | `4.0` | 每个有效MI窗口长度 |
| `buffer_sec` | `60` | JellyFish内存环形缓冲长度 |

`step_sec` 属于采后/实时解码兼容参数。正式采集固定每个trial只保留一个完整4秒窗，实际
采集步长由协议代码固定为4秒。

## 会话参数

```yaml
protocol:
  collection_blocks: 9
  collection_trials_per_class_per_block: 50
  rest_between_blocks_sec: 180.0
  random_seed: 17
```

每个block的trial数为：

```text
每类数量50 × 类别数2 = 100
```

`random_seed` 控制可复现的伪随机顺序。每个block左右各50个、连续同类不超过2个，且前后
半段近似平衡。

trial内部的 `2+2+4` 秒时序固定在代码中，不能通过YAML修改。

## JellyFish参数

```yaml
device:
  neuracle_host: 127.0.0.1
  neuracle_port: 8712
  neuracle_source_sfreq: 250
  neuracle_transport_delay_sec: 0.0
  neuracle_eeg_channels: 59
  neuracle_eeg_channel_names:
    # 固定59个名称，使用项目现有清单
```

### `neuracle_host`

- JellyFish和项目同机：`127.0.0.1`。
- JellyFish在另一台电脑：填写那台电脑由 `ipconfig` 显示的局域网IPv4。
- 它不是放大器IP。

### `neuracle_port`

JellyFish TCP转发端口，默认8712。以JellyFish界面的实际配置为准。可以验证：

```powershell
Test-NetConnection 127.0.0.1 -Port 8712
```

### `neuracle_source_sfreq`

固定250Hz。项目在session结束后才降采样到200Hz；不能让JellyFish直接转发200Hz。

### `neuracle_transport_delay_sec`

设备采样到JellyFish交付之间的可选固定延迟补偿。它是对整条时间轴的一次性固定偏移，
不会逐trial累计。没有独立测量时必须保持 `0.0`，不能凭经验猜测。

设备时钟与电脑时钟的缓慢漂移由近期120秒数据包下包络自动跟踪，与这个固定补偿分开。

### 通道清单

项目按名称选择59个头皮EEG通道，顺序以 `neuracle_eeg_channel_names` 为准。缺少、重复或
被JellyFish标为非EEG的必需通道都会阻止正式采集。以下辅助通道不进入输出：

```text
ECG HEOR HEOL VEOU VEOL
```

不要为了通过检查而删除或重命名必需通道；应修复JellyFish通道设置。

## 保存路径

```yaml
storage:
  records_dir: records_storage
```

相对路径相对于启动命令的当前目录。推荐始终先 `Set-Location` 到项目根目录再启动，或使用
明确的绝对路径，例如：

```yaml
storage:
  records_dir: D:/EEG/oi-mi-records
```

不要在正式采集过程中把输出目录设到网络盘或同步盘。

## 不使用的硬件

当前正式协议不使用外置TriggerBox，也没有串口配置项。事件编号只保存在
`events.json`，不会发送硬件码。

