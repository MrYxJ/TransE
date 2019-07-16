# TransE

## Data
在```/data/```目录下有训练集和测试集不同划分比例的两个数据集```0_1```和```_0_5```，具体说明见该目录下```README.md```。

```/data/0_1/```数据集的结果已经跑出来存在 ```ent_embeddings.npy```，可以运行```Model.py```查看该数据集的结果。

## 任务

复现TransE模型,在/data/0_5 的数据集上跑出该KG的每个实体的 Embedding,并使用metric.py里的 get_hits()评测效果，类似Model.py。



