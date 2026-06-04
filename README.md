# remove-watermarks-skill

一个轻量的图片水印去除工具：用户或 Agent 指定水印所在区域，脚本先生成预览图，确认后使用 LaMA/iopaint 对该区域进行修复。

它不使用 Florence-2 这类大型识别模型，因此首次准备更轻；但它也不会自动识别任意水印位置，需要你提供大致区域，例如“右下角”“中间”或具体坐标。

> 请仅用于你拥有版权、已获授权，或有权编辑的图片。

## 适合处理的场景

- 右下角、左上角等角落小 logo
- 图片底部或顶部的平台水印
- 半透明文字水印
- 面积较小、背景相对自然的水印

不太适合：

- 大面积居中水印
- 覆盖在人脸、文字、商品标签、表格、UI 截图上的水印
- 需要精确还原线条或文字的图片

## 安装与初始化

进入项目目录后，首次运行：

```bash
python scripts/setup_runtime.py --warmup
```

国内网络可以使用镜像：

```bash
python scripts/setup_runtime.py --warmup --china-mirror
```

如果需要 CUDA 12.4：

```bash
python scripts/setup_runtime.py --warmup --cuda cu124
```

LaMA 模型首次下载约 196 MB。完整依赖会更大，尤其是 CUDA 版 PyTorch。

## 用户如何让 Agent 使用

你可以直接这样说：

```text
帮我去掉这张图片右下角的水印
```

```text
帮我去掉图片底部的半透明文字水印，先生成预览图给我看
```

```text
这张图中间有水印，帮我先预览一下修复区域
```

推荐流程：

1. 用户说明水印位置。
2. Agent 先运行 `preview` 生成预览图。
3. 用户确认区域是否准确。
4. Agent 再运行 `remove` 输出修复后的图片。

## 命令行使用

### 1. 预览右下角水印区域

```bash
python scripts/remove_watermark_lite.py \
  --input input.png \
  --output clean.png \
  --mode preview \
  --region bottom-right
```

会生成：

```text
clean_mask.png
clean_preview.png
```

`clean_preview.png` 会用红色半透明区域标出将要修复的位置。

### 2. 确认后去除水印

```bash
python scripts/remove_watermark_lite.py \
  --input input.png \
  --output clean.png \
  --mode remove \
  --region bottom-right
```

### 3. 使用精确坐标

```bash
python scripts/remove_watermark_lite.py \
  --input input.png \
  --output clean.png \
  --mode remove \
  --bbox 1200,850,1800,980
```

### 4. 扩大修复边缘

半透明水印容易残留边缘，可以加 `--padding`：

```bash
python scripts/remove_watermark_lite.py \
  --input input.png \
  --output clean.png \
  --mode preview \
  --region bottom-right \
  --padding 8
```

### 5. 调整默认区域大小

```bash
python scripts/remove_watermark_lite.py \
  --input input.png \
  --output clean.png \
  --mode preview \
  --region bottom-right \
  --region-width 0.22 \
  --region-height 0.12
```

## 支持的区域

```text
top-left
top-right
bottom-left
bottom-right
top
bottom
left
right
center
```

默认区域大小：

| 区域 | 默认大小 |
|---|---|
| 四角 | 宽 28%，高 16% |
| 顶部 / 底部 | 宽 100%，高 18% |
| 左侧 / 右侧 | 宽 18%，高 100% |
| 中间 | 宽 35%，高 22% |

## 常用参数

| 参数 | 说明 |
|---|---|
| `--input` | 输入图片路径 |
| `--output` | 输出图片路径 |
| `--mode preview` | 只生成预览图和 mask |
| `--mode remove` | 执行水印修复 |
| `--region` | 使用预设区域 |
| `--bbox` | 使用精确坐标：`x1,y1,x2,y2` |
| `--padding` | 扩大修复区域边缘 |
| `--overwrite` | 覆盖已有输出文件 |
| `--device auto/cpu/cuda` | 指定运行设备 |

支持图片格式：`png`、`jpg`、`jpeg`、`webp`。

## 输出 JSON 示例

```json
{
  "ok": true,
  "mode": "preview",
  "input": "input.png",
  "output": "clean.png",
  "bbox": [1440, 907, 2000, 1080],
  "mask": "clean_mask.png",
  "preview": "clean_preview.png",
  "device": null,
  "warnings": []
}
```

## 测试

运行不依赖 LaMA 的测试：

```bash
python scripts/test_remove_watermark_lite.py
```
