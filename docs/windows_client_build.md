# Windows 客户端构建指南

本文档说明如何从源码在 Windows 上构建 Lada 客户端（CLI 和 GUI 的可执行文件）。

> [!NOTE]
> 本文档面向需要在 Windows 上**构建可执行文件**的开发者。如果只是想从源码运行，请参考 [Windows 安装指南](windows_install.md)。

## 前置条件

### 系统依赖

构建过程需要以下工具。你可以通过一键脚本自动安装，也可以手动安装：

```powershell
# 以管理员身份运行 PowerShell
winget install --id Gyan.FFmpeg -e --source winget
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
winget install --id=7zip.7zip -e --source winget

# 如果构建 GUI（而非仅 CLI），还需要：
winget install --id MSYS2.MSYS2 -e --source winget
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Rustlang.Rustup -e --source winget
winget install --id Microsoft.VCRedist.2013.x64  -e --source winget
winget install --id Microsoft.VCRedist.2013.x86  -e --source winget
```
安装完python后，配置路径到环境变量PATH，以执行 python 和 python3 命令。 

安装完成后**重启电脑**。

## 方式一：一键构建脚本（推荐）

项目提供了自动化构建脚本，会依次完成依赖安装、模型下载、编译和打包。

```powershell
powershell -ExecutionPolicy Bypass ./packaging/windows/package_executable.ps1 -extra nvidia
```

### 脚本参数

| 参数 | 说明 |
|------|------|
| `-extra <EXTRA>` | 选择 GPU 后端，目前支持 `nvidia` 或 `intel` |
| `-skipWinget` | 跳过通过 winget 安装/更新系统依赖 |
| `-skipGvsbuild` | 跳过 gvsbuild 依赖构建（如果已有预编译的 GTK 依赖） |
| `-skipArchive` | 跳过创建 7z 压缩包 |
| `-skipTranslations` | 跳过翻译编译，构建结果不含翻译文件 |
| `-cleanGvsbuild` | 执行 gvsbuild 的干净构建（更新 gvsbuild/uv/python 后建议使用） |
| `-cliOnly` | 仅构建 `lada-cli.exe`，跳过 GUI |

构建完成后，可执行文件和压缩包会输出到 `dist/` 目录。

## 方式二：手动逐步构建

### 1. 获取源码

```powershell
git clone https://codeberg.org/ladaapp/lada.git
cd lada
```

### 2. 下载模型权重

```powershell
# 下载模型
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v2.pt"
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v4_accurate.pt"
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v4_fast.pt"
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' -OutFile ".\model_weights\lada_mosaic_restoration_model_generic_v1.2.pth"

# 可选，运行 DeepMosaics 修复模型
Invoke-WebRequest 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' -OutFile ".\model_weights\3rd_party\clean_youknow_video.pth"
```

### 3. 构建 GTK 系统依赖（仅 GUI 需要）

> 如果你只需要 CLI 或者已有预编译的 `build_gtk_release` 目录，可跳过此步骤。

```powershell
# 创建构建环境
uv venv venv_gtk_release
.\venv_gtk_release\Scripts\Activate.ps1

uv pip install gvsbuild==2026.4.1
uv pip install patch
uv run --no-project python -m patch -p1 -d venv_gtk_release/lib/site-packages patches/gvsbuild_ffmpeg.patch
uv pip uninstall patch

# 设置环境变量，引导 FFmpeg 使用 MSVC 工具链（好像不需要且没有效果）
set MSYS2_ARG_CONV_EXCL=*
set CC=cl
set LD=link

# 构建 GTK 及相关依赖（耗时较长）
gvsbuild build --configuration=release --build-dir='./build_gtk_release' --enable-gi --py-wheel gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext

# 构建 GTK 及相关依赖（通过 --fast-build 跳过已构建的包，与上面指令二选一，未验证）
gvsbuild build --configuration=release --build-dir='./build_gtk_release' --enable-gi --py-wheel --fast-build gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext

deactivate
```

### 4. 安装 Python 依赖并构建可执行文件

```powershell
# 创建发布用的虚拟环境
uv venv --clear --python 3.13 venv_release_win
.\venv_release_win\Scripts\Activate.ps1

# 安装项目依赖（根据 GPU 选择 extra）
uv sync --active --frozen --extra nvidia --no-editable --no-install-project

# 安装项目本身
uv pip install --no-deps '.'

# 安装 GUI 所需的 PyGObject/PyCairo（仅 GUI 构建需要）
uv pip install --force-reinstall (Resolve-Path ".\build_gtk_release\gtk\x64\release\python\pygobject*.whl").Path
uv pip install --force-reinstall (Resolve-Path ".\build_gtk_release\gtk\x64\release\python\pycairo*.whl").Path

# 安装 PyInstaller 并应用必要的补丁
uv pip install pyinstaller==6.18.0 "setuptools<81.0.0"
uv pip install patch
uv run --no-project python -m patch -p1 -d venv_release_win/lib/site-packages patches/increase_mms_time_limit.patch
uv run --no-project python -m patch -p1 -d venv_release_win/lib/site-packages patches/remove_ultralytics_telemetry.patch
uv run --no-project python -m patch -p1 -d venv_release_win/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
uv pip uninstall patch

# 兼容性修复：构建机器若无 AVX512，需替换 polars
uv pip uninstall polars
uv pip install polars-lts-cpu

deactivate
```

### 5. 编译翻译文件（可选）

```powershell
powershell -ExecutionPolicy Bypass .\translations\compile_po.ps1
```

### 6. 运行 PyInstaller 打包

```powershell
# 激活虚拟环境
.\venv_release_win\Scripts\Activate.ps1

# 设置 GTK 相关环境变量（仅 GUI 构建需要）
$release_dir = (Resolve-Path ".\build_gtk_release\gtk\x64\release").Path
$env:Path = $release_dir + "\bin;" + $env:Path
$env:LIB = $release_dir + "\lib;" + $env:LIB
$env:INCLUDE = $release_dir + "\include;" + $release_dir + "\include\cairo;" + $release_dir + "\include\glib-2.0;" + $release_dir + "\include\gobject-introspection-1.0;" + $release_dir + "\lib\glib-2.0\include;" + $env:INCLUDE

# 执行打包（选择 lada.spec 用于完整构建）
uv run --no-project pyinstaller --noconfirm ./packaging/windows/lada.spec -- --extra=nvidia

deactivate
```

打包完成后，输出位于 `dist/lada/` 目录。

### 7. 创建压缩包（可选）

```powershell
# 将 dist/lada 目录打包为 7z 格式
$env:Path = ($env:Programfiles + "\7-Zip;") + $env:Path
7z.exe a -v1999m ./dist/lada-v<version>_windows_nvidia.7z ./dist/lada
```

## GPU 后端选择

| Extra | 支持的 GPU 架构 | 驱动要求 |
|-------|----------------|---------|
| `nvidia` | Nvidia Volta(7.0), Turing(7.5), Ampere(8.0/8.6), Hopper(9.0), Blackwell(10.0/12.0) | Nvidia 驱动 >= 570 |
| `intel` | Intel Arc 独立显卡 (Alchemist/Battlemage), Core Ultra 集成显卡 (Meteor Lake-H, Arrow Lake-H, Lunar Lake) | 参考 [Intel GPU 驱动安装指南](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-9.html) |

## 常见问题

**Clock skew 构建错误**

检查系统日期和时间设置。如果仍有问题，可以重写时间戳：

```powershell
$now = (Get-Date)
Get-ChildItem -Path ./build_gtk_release/build -Recurse | ForEach-Object { $_.LastWriteTime = $now }
```

**更新 gvsbuild 后**

建议使用 `--clean-gvsbuild` 参数执行干净构建，并在另一台干净的 Windows 虚拟机上测试生成的可执行文件，确保 PyInstaller 捕获了所有必要的依赖。
