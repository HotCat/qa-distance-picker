# Windows 10 部署指南

以下步骤在全新 Windows 10（64位）工作站上部署距离测量质检系统。

---

## 方案 A：直接部署（推荐）

### 1. 安装 Miniconda

1. 下载 Miniconda 安装包：
   - 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
   - 官方：https://docs.conda.io/en/latest/miniconda.html
2. 运行安装程序，全部默认选项
   - 勾选 "Add Miniconda to my PATH environment variable"
3. 安装完成后打开 **Anaconda Prompt**（开始菜单搜索）

验证：
```
conda --version
python --version
```

---

### 2. 创建 Python 虚拟环境

```
conda create -n qa python=3.11 -y
conda activate qa
```

---

### 3. 安装项目依赖

将项目代码复制到工作站（或 git clone），进入项目目录：

```
cd C:\Users\你的用户名\qualityAssurance
pip install -r requirements.txt
```

> 下载慢时使用清华镜像：
> ```
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

`requirements.txt` 内容：
```
PySide6
diplib
opencv-python
numpy
pyyaml
pillow
scipy
```

---

### 4. 安装 MindVision 相机 SDK

1. 从 MindVision 官网下载 SDK：
   - http://www.mindvision.com.cn/ → 技术支持 → SDK下载
   - 或使用随相机附带的 U 盘/光盘中的 SDK 安装包
2. 运行 `MVCAMSDK.exe` 安装程序
3. 默认安装路径：`C:\Program Files\MindVision\MVCAMSDK`
4. 确认 DLL 存在：
   ```
   dir "C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64\MVCAMSDK_X64.dll"
   ```
5. 将 Runtime 路径添加到系统 PATH：
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在"系统变量"中找到 `Path`，编辑，添加：
     ```
     C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64
     ```
   - 确定保存

> 不使用相机（仅用图片测试）可跳过此步。程序无相机时不会崩溃。

---

### 5. 运行程序

```
conda activate qa
cd C:\Users\你的用户名\qualityAssurance
python app.py
```

如果一切正常，PySide6 窗口弹出。

---

### 6. 使用流程

#### 6.1 标定像素尺寸

1. 打印一张棋盘格标定板（默认参数：11×8 内角点，格子大小 5mm）
2. 将标定板放在相机下方
3. 切换到 **Image Processing** 模式，点击 **Grab** 采集一帧
4. 点击 **Calibration**，程序自动检测棋盘格并计算 `pixel_size`
5. 标定结果自动保存到 `config.yaml`

#### 6.2 创建测量模板

1. 将完美产品放在相机下方
2. 切换到 **Image Processing** 模式，点击 **Grab** 采集一帧
3. 点击 **Arclines** 运行线/弧检测，等待完成
4. 在 **Detection Results** 窗口中用 Ctrl+点击选择两个特征（线或弧），自动创建 Feature Pair
5. 重复步骤 4 添加所有需要的测量对
6. 在 **Feature Pair Measurements** 窗口中输入公差上下限（Lower / Upper 列）
7. 点击 **Confirm** 保存模板，输入模板名称（如产品型号）

#### 6.3 批量检测

1. 点击 **BatchInspect** 打开批量检测窗口
2. 在下拉框中选择模板
3. 放置待测产品
4. 点击 **Inspect**（或按键盘 **I** 键）执行检测
5. 结果表格中显示：
   - Distance：实测距离
   - Pass：✓（合格）或 ✗（不合格）

#### 6.4 RANSAC 对齐（可选）

产品放置有旋转或偏移时，勾选工具栏 **Align** 复选框，检测时自动先做刚体对齐再匹配。

---

### 7. 创建桌面快捷方式

创建 `QADistancePicker.bat` 放到桌面：

```bat
@echo off
call C:\Users\你的用户名\miniconda3\Scripts\activate.bat qa
cd /d C:\Users\你的用户名\qualityAssurance
python app.py
pause
```

双击即可启动。

---

## 方案 B：EXE 部署（免安装 Python）

### 1. 下载构建产物

1. 打开 GitHub 仓库 → **Actions** → **Build Windows EXE** → 最近一次成功运行
2. 在 Artifacts 区域下载 `qa-distance-picker-windows.zip`

### 2. 安装 MindVision 相机 SDK

同方案 A 步骤 4。

### 3. 解压并运行

1. 将 zip 解压到目标目录，如 `C:\QA\qa-distance-picker\`
2. 双击 `qa-distance-picker.exe` 运行

> EXE 已内置 VC++ 运行时 DLL，无需额外安装。

### 4. 配置文件

程序首次运行会在 exe 同目录下生成 `config.yaml`，可手动编辑：

```yaml
processing:
  pixel_size: 0.117027    # mm/pixel，标定后自动更新
detection:
  line_min_mm: 20.5       # 线长度过滤下限
  line_max_mm: 170.0      # 线长度过滤上限
  arc_min_mm: 2.0         # 弧半径过滤下限
  arc_max_mm: 10.0        # 弧半径过滤上限
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError` | 确认已 `conda activate qa` 再运行 |
| 相机打不开 | 确认 SDK 已安装且 Runtime 路径在系统 PATH 中 |
| 画面全黑 | 检查镜头盖是否取下，在 Camera Settings 中调大曝光时间 |
| `vcruntime140.dll not found`（方案A） | 安装 VC++ 运行时：https://aka.ms/vs/17/release/vc_redist.x64.exe |
| 检测结果全为 N/A | 检查线长度/弧半径过滤范围是否覆盖实际特征尺寸 |
| Inspect 报 "infeasible cost matrix" | 当前帧没有特征落在过滤范围内，放宽过滤参数 |
| pip 下载超时 | 使用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
