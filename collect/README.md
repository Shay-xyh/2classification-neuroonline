# `collect` 目录说明

这个目录只保留当前 Neuracle / JellyFish 采集链路需要的底层代码。

## 文件

- `neuracle_api.py`
  - 提供 `DataServerThread`
  - 当前被 `NeuracleAcquirer` 直接复用

正式入口位于 GUI“数据采集”页或 CLI `collect`，设备适配位于
`acquisition/neuracle_acquirer.py`。当前链路从 JellyFish 接收 250 Hz 数据，按通道名
选择 59 个 EEG 通道，并把未经采后预处理的连续数据交给 `SessionRecorder`。

预处理和 4 秒切窗不在本目录执行，也不在刺激呈现期间执行；它们由
`adaptation/calibrator.py` 在设备流停止后统一完成。本目录不保存旧 GUI、旧范式或
模型训练入口。
