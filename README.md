# Modality-Invariant Optical–SAR Image Registration using Hierarchical Structural Cognition Prior
Code for paper "Modality-Invariant Optical–SAR Image Registration using Hierarchical Structural Cognition Prior"

# 项目说明

## 环境与依赖

- Python 版本：建议 Python `3.8+`

- 依赖安装：确保仓库根目录下已有 `requirements.txt`，然后执行：

```bash
pip install -r requirements.txt

## checkpoints
- `checkpoints`：提供了在 **OSdataset** 和 **WHU-SEN-City** 数据集下的原始训练权重。

## 数据集
- 文中进行实验的数据集为 **OSdataset** 和 **WHU-SEN-City**，图片大小均为 **512×512**。
- OSdataset 下载链接：**（请在此处补充链接）**
- WHU-SEN-City 下载链接：**（请在此处补充链接）**

## 训练
- `train.py`：用于权重训练的脚本，训练得到的权重和训练日志会存储至 `checkpoints` 文件夹下。

## 评估
- `Evaluation_OSdataset / Evaluation_WHU_SEN_City`：分别包含两个数据集下的评估文件，可进行：
  - 关键点与特征描述提取
  - 关键点匹配
  - 影像配准
  - 输出可视化结果与指标

评估目录内主要内容如下：

- `VIS-SAR/`：数据集存储文件夹
- `extract_feature.py`：关键点和特征描述提取脚本，输出 `.mat` 文件至 `feature` 文件夹
- `match.py`：读取 `extract_feature.py` 输出的 `.mat` 文件，进行光 SAR 影像关键点匹配，并将可视化与指标结果保存至 `result` 文件夹
- `reproj.py`：读取 `extract_feature.py` 输出的 `.mat` 文件，进行光 SAR 影像配准，并将可视化与指标结果保存至 `result` 文件夹

	
