# 首次部署：Windows + Neuracle/JellyFish

本文用于在一台新Windows电脑上从零配置正式采集环境。正式采集不需要GPU、CUDA、
模型权重、Unity、BrainCo、TriggerBox或Git。

## 1. 电脑与代码准备

建议使用Windows 10/11 64位、至少16GB内存并预留20GB以上磁盘空间。正式实验期间应
接通电源，并关闭睡眠、屏保、自动更新和通知。

安装Git（已安装可以跳过）：

```powershell
winget install --id Git.Git --exact --source winget --accept-package-agreements --accept-source-agreements
```

关闭并重新打开PowerShell，然后从个人仓库克隆采集版代码。把下面地址替换成实际仓库地址：

```powershell
New-Item -ItemType Directory -Path D:\Projects -Force
Set-Location D:\Projects
git clone https://github.com/<你的账号>/<仓库名>.git oi-mi
Set-Location .\oi-mi
```

私有仓库会要求登录GitHub；使用浏览器或Git Credential Manager完成登录，不要把访问令牌
写进配置文件或提交到仓库。克隆后的项目位置例如：

```text
D:\Projects\oi-mi
```

仓库不应包含 `.venv`、采集记录、运行缓存、模型权重、原始视频或临时动画帧。Python
虚拟环境必须在新电脑重新创建。

## 2. 安装博睿康软件

使用实验室已经验证的同一版本安装：

1. 博睿康设备驱动；
2. 博睿康原生采集软件；
3. JellyFish数据转发软件。

先不启动本项目，在原生软件中完成以下检查：

- 放大器能够连接并显示EEG波形；
- 硬件采样率为250Hz；
- 电极名称和类型正确；
- 原生软件可以录制文件；
- JellyFish能够开始数据转发。

项目当前不通过SDK读取博睿康阻抗，阻抗检查必须在博睿康原生软件中完成。

## 3. 确定 JellyFish 地址

如果JellyFish与 `oi-mi` 在同一台电脑，使用：

```yaml
neuracle_host: 127.0.0.1
neuracle_port: 8712
```

`127.0.0.1` 表示当前电脑，`8712` 是JellyFish接口的默认TCP端口。最终端口必须与
JellyFish界面中的实际转发设置一致。JellyFish开始转发后运行：

```powershell
Test-NetConnection 127.0.0.1 -Port 8712
```

正常应显示：

```text
TcpTestSucceeded : True
```

只有JellyFish运行在另一台电脑时，才把 `neuracle_host` 改成那台电脑由 `ipconfig`
显示的局域网IPv4地址，并在JellyFish电脑的防火墙中放行TCP端口8712。

## 4. 安装 Python 3.12 和采集环境

项目固定使用Python 3.12.x。新电脑如果还没有Python，先在PowerShell安装：

```powershell
winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
```

安装后关闭并重新打开PowerShell，再执行：

```powershell
Set-Location D:\Projects\oi-mi
py -3.12 setup_local.py
```

脚本会：

1. 查找Python 3.12；如果脚本由其他Python版本启动且3.12缺失，会尝试通过 `winget` 安装；
2. 在项目目录创建 `.venv`；
3. 安装Neuracle正式采集所需的最小依赖；
4. 运行采集环境检查。

默认安装不会下载模型权重，也不会安装Torch、BrainCo SDK或Unity。它们只属于可选的
采后功能：

```powershell
# 只有需要采后训练/实时解码时才使用
py -3.12 setup_local.py --with-decoding

# 只有需要BrainCo设备适配时才使用
py -3.12 setup_local.py --with-brainco
```

如果不希望脚本自动安装Python：

```powershell
py -3.12 setup_local.py --no-install-python
```

## 5. 核对正式配置

打开 `config.yaml`，至少确认：

```yaml
device_type: neuracle
hardware_dummy_mode: false
sfreq: 200
window_sec: 4.0
buffer_sec: 60

protocol:
  collection_blocks: 9
  collection_trials_per_class_per_block: 50
  rest_between_blocks_sec: 180.0

device:
  neuracle_host: 127.0.0.1
  neuracle_port: 8712
  neuracle_source_sfreq: 250
  neuracle_transport_delay_sec: 0.0
  neuracle_eeg_channels: 59

storage:
  records_dir: records_storage
```

不要修改59通道名称清单。没有经过独立测量时，
`neuracle_transport_delay_sec` 必须保持 `0.0`。详细说明见
[配置参考](docs/CONFIGURATION.md)。

## 6. 第一次启动与无硬件验证

启动GUI：

```powershell
.\.venv\Scripts\python.exe cli.py gui
```

第一次先不要连接被试，在“数据采集”页依次验证：

1. “查看范式”；
2. “左右手练习”；
3. “画面测试”；
4. “无硬件演练”；
5. 电脑全屏；
6. “我要休息”后当前trial作废，继续后重采同一trial；
7. 演练完成后生成四个session文件。

无硬件演练使用独立被试ID并模拟250Hz、59通道，不会覆盖正式记录。

## 7. 第一次硬件验证

启动顺序固定为：

```text
放大器 → 博睿康原生软件 → JellyFish开始转发 → oi-mi
```

然后执行：

```powershell
.\.venv\Scripts\python.exe cli.py probe-device --device neuracle --duration 5
```

正常应看到：

```text
设备转发正常 shape=(59, 800)
```

`59×800` 是连通测试返回的4秒、200Hz视图；连续原始采集仍保存为250Hz。

### 短硬件预采集

复制正式配置作为临时文件：

```powershell
Copy-Item .\config.yaml .\config.preflight.yaml
```

只在 `config.preflight.yaml` 中改为：

```yaml
subject_id: HARDWARE_PREFLIGHT
protocol:
  collection_blocks: 1
  collection_trials_per_class_per_block: 1
  rest_between_blocks_sec: 0.0
```

然后使用临时配置启动GUI：

```powershell
.\.venv\Scripts\python.exe cli.py --config .\config.preflight.yaml gui
```

完成2个trial后检查四个输出文件、59通道、250Hz源采样率、200Hz窗口和零丢包。正式实验
必须关闭这个GUI，再使用不带 `--config` 的默认正式配置重新启动，避免误采只有2个trial。

## 8. 环境复查

只检查正式采集环境：

```powershell
.\.venv\Scripts\python.exe tools\check_environment.py
.\.venv\Scripts\python.exe -m pip check
```

启动失败时不要改用全局 `streamlit`；始终通过项目虚拟环境中的Python运行 `cli.py`。
正式实验当天的完整操作见 [正式实验操作](docs/FORMAL_COLLECTION.md)。
