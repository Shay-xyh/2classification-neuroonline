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

## 获取与更新代码

唯一使用的仓库和分支：

```text
https://github.com/Shay-xyh/2classification-neuroonline.git
main
```

新电脑首次克隆：先在资源管理器中进入你希望存放项目的文件夹，在该文件夹中打开
PowerShell，然后执行：

```powershell
git clone https://github.com/Shay-xyh/2classification-neuroonline.git oi-mi
Set-Location .\oi-mi
py -3.12 .\setup_local.py
```

实验电脑强制同步到远端 `main` 时，先在项目根目录（能够看到 `cli.py` 的目录）打开
PowerShell。以下命令从当前目录自动取得项目路径，并保留本机 `config.yaml`：

```powershell
$ProjectPath = (Get-Location).Path
if (-not (Test-Path -LiteralPath "$ProjectPath\cli.py")) { throw '当前目录不是项目根目录' }
$ConfigBackup = Join-Path (Split-Path -Parent $ProjectPath) 'oi-mi-config.local.yaml'
Copy-Item -LiteralPath "$ProjectPath\config.yaml" -Destination $ConfigBackup -Force
git -C $ProjectPath remote set-url origin https://github.com/Shay-xyh/2classification-neuroonline.git
git -C $ProjectPath fetch origin
git -C $ProjectPath reset --hard origin/main
Copy-Item -LiteralPath $ConfigBackup -Destination "$ProjectPath\config.yaml" -Force
py -3.12 "$ProjectPath\setup_local.py"
```

该操作不会删除被 `.gitignore` 排除的 `records_storage`、`.venv` 或 `.runtime`。完整更新说明、
采用远端默认配置的方法和安全检查见 [首次部署](INSTALL.md)。

## 已配置电脑上的启动方式

先启动博睿康原生软件与 JellyFish 数据转发，再在项目根目录打开 PowerShell 并运行：

```powershell
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
