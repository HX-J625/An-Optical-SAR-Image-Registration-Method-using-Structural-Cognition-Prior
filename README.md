# Modality-Invariant Optical–SAR Image Registration using Hierarchical Structural Cognition Prior
Code for paper "Modality-Invariant Optical–SAR Image Registration using Hierarchical Structural Cognition Prior"

"checkpoints"：
提供了在OSdataset和WHU-SEN-City数据集下的原始训练权重。

数据集：
文中进行实验的数据集为OSdataset和WHU-SEN-City数据集，图片大小均为512×512。
OSdataset下载链接：
WHU-SEN-City下载链接：

train.py：
用于权重训练的脚本，训练的权重和训练日志会存储至“checkpoints”文件夹下。

"Evaluation_OSdataset/Evaluation_WHU_SEN_City"：
分别包含两个数据集下的评估文件，可以进行关键点和特征描述的提取、关键点匹配以及影像配准，并输出可视化图以及指标。

	"VIS-SAR"：
	数据集存储文件夹。

	extract_feature.py：
	关键点和特征描述提取脚本，输出.mat文件至“feature”文件夹下。

	match.py：
	读取extract_feature.py输出的.mat文件，进行光SAR影像关键点匹配，并将可视化和指标结果保存至“result”文件夹。

	reproj.py：
	读取extract_feature.py输出的.mat文件，进行光SAR影像配准，并将可视化和指标结果保存至“result”文件夹。
	
