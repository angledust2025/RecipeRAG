# IR-CLIP

Model weights are not included in this code release. From the repository root, download the desired checkpoint as described in the [release README](../README.md#download-model-weights). The training and evaluation commands below expect it under `checkpoints/base` or `checkpoints/large`.

## 环境配置

使用 Python 3.10 及支持 CUDA 的 PyTorch 环境。先按机器的 CUDA 版本安装 PyTorch，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

## 数据组织

训练和评估数据使用 JSON 数组或 JSONL。每条记录包含配对图片和食谱文本：

```json
{"image_path":"Recipe1M/train/0/0/0/0/example.jpg","seg_image_path":"Recipe1M/train_sam3_touming/0/0/0/0/example.png","title":"Recipe title","ingredients":"Ingredient list","instructions":"Recipe instructions"}
```

`image_path` 和 `seg_image_path` 可以是相对路径或绝对路径。使用相对路径时，通过 `IMAGE_ROOT`（训练）或 `--image-root`（评估）指定其根目录。仓库的 `data/` 下提供了 1,000 条训练和测试样例；图片文件需按样例中的相对目录另行准备。训练脚本默认启用前景图像对齐，因此每条训练记录都需要 `seg_image_path`。训练样例已包含分割图的相对路径；准备好数据后，该路径应能在 `IMAGE_ROOT` 下解析到对应 PNG。

前景分割使用 Hugging Face [`facebook/sam3`](https://huggingface.co/facebook/sam3) 权重，由 `Sam3Model` 和 `Sam3Processor` 加载。仓库提供的 [`data/sam3.py`](data/sam3.py) 先用提示词 `food` 分割，未检测到实例时回退到 `drink`；实例置信阈值为 `0.60`，像素掩码阈值为 `0.30`。通过阈值的实例掩码会合并为前景，背景置黑，再按 RGB 阈值 `20` 生成透明度并以高斯 `sigma=1.0` 羽化边缘。脚本输出 PNG，按输入目录结构保存；默认跳过已生成的图像。启用前景图像对齐时，数据记录中的 `seg_image_path` 应指向对应输出文件。

```bash
python data/sam3.py \
  --input-root ./dataset/Recipe1M/train \
  --output-root ./dataset/Recipe1M/train_sam3_touming
```

使用前需按 Hugging Face [`facebook/sam3`](https://huggingface.co/facebook/sam3) 页面要求取得模型访问权限，并完成 Hugging Face 认证。可通过 `--model` 指定本地权重目录；使用 `--overwrite` 可重新生成已有输出。

## 启动训练

在仓库根目录执行。`MODEL` 可选 `checkpoints/base` 或 `checkpoints/large`；根据机器资源设置 GPU 数量、数据文件和图片根目录：

训练脚本固定训练 **4 个 epoch**，并默认启用前景对齐损失。训练使用模型的短文本位置编码长度（77 tokens）；检索评估对 instructions 使用 248-token 长位置编码。

```bash
MODEL=checkpoints/base \
TRAIN_JSON=data/train_sample_1000.json \
IMAGE_ROOT=./dataset \
NUM_GPUS=8 \
OUTPUT_DIR=outputs/irclip-base \
scripts/train.sh
```

## 启动评估

用单个 JSON/JSONL 文件评估一个检索集合；将目录传给 `--data` 时，脚本会逐个评估其中的 `sample_*.json` 并报告各子集平均值。评估使用 224px 图像输入、标题和配料的 77-token 短位置编码、instructions 的 248-token 长位置编码，并对三种文本特征取归一化均值。

```bash
python retrieval_global.py \
  --checkpoint checkpoints/base \
  --data data/test_sample_1000.json \
  --image-root ./dataset

python retrieval_global.py \
  --checkpoint checkpoints/large \
  --data ./dataset/test1000 \
  --image-root ./dataset
```
