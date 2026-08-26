# 2classification-neuroonline

`oi-mi` 是用于左右手二分类运动想象 EEG 实验的数据采集程序。当前正式采集只使用
Neuracle（博睿康）设备与 JellyFish 数据转发；采集入口不会加载、训练、推理或更新模型。

## 当前正式协议

- 输入：250 Hz、59 个纯 EEG 通道；排除 `ECG/HEOR/HEOL/VEOU/VEOL`。
- 单个 trial：2 秒绿色注视十字 + 2 秒手部动作提示 + 4 秒箭头运动想象。
- 类别：左手 `0`、右手 `1`，没有静息类别和额外 ITI。
- 会话：9 个 block × 100 个有效 trial，共 900 个；前8个 block 结束后自动休息3分钟。
- 手动暂停：丢弃当前 attempt，恢复后重采同一个计划 trial。
- 输出：连续原始 EEG、按样本号对齐的事件、元数据和采后生成的4秒窗口。

完整定义以 [CURRENT_EXPERIMENT_PROTOCOL.md](CURRENT_EXPERIMENT_PROTOCOL.md) 为准。

## 已配置电脑上的启动方式

先启动博睿康原生软件与 JellyFish 数据转发，再运行：

```powershell
Set-Location D:\path\to\oi-mi
.\.venv\Scripts\python.exe cli.py gui
```

正式实验使用 GUI 的“数据采集”页。设备连通测试也可以使用：

```powershell
.\.venv\Scripts\python.exe cli.py probe-device --device neuracle --duration 5
```

## 文档

- [首次部署](INSTALL.md)：新Windows电脑从零安装Python、JellyFish连接和首次验证。
- [正式实验操作](docs/FORMAL_COLLECTION.md)：实验前、实验中、暂停、结束和备份清单。
- [配置参考](docs/CONFIGURATION.md)：正式采集相关配置项及 JellyFish 地址确定方法。
- [数据与时间对齐](docs/DATA_FORMAT.md)：输出文件、事件、切窗、固定延迟和时钟漂移。
- [正式实验协议](CURRENT_EXPERIMENT_PROTOCOL.md)：唯一有效的范式和标签定义。

模型训练、实时解码、Unity小车与 `volc_upload/` 属于采后或独立研究流程，不是正式数据
采集的依赖，也不应在采集过程中启动。
