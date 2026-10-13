# Demo2：NER

基于 PyTorch 和 Hugging Face `BertModel` 实现的中文命名实体识别项目，支持 **Weibo** 和 **MSRA** 两个数据集，以及 `bert-base-chinese` 和 `chinese-bert-wwm-ext` 两种预训练模型。

### 本次修改说明 - V2

1. **重构评价指标**：参考 W2NER 写法，直接统计预测对的数、预测总数和真实总数，用 $2 \times \text{正确数} / (\text{预测数} + \text{真实数})$ 计算 F1。
2. **合并工具函数**：将类外的 `extract_entities_from_bio` 和 `print_metrics_table` 合并进 `NERMetric` 作为静态方法。
3. **固定依赖版本**：在 `requirements.txt` 中固定版本。
4. **调大微博截断长度**：微博最长样本有 175 字，原 96 长度会截断样本；现统一调为 192。

## 1. 目录结构

```text
├── configs/             # 四组实验的参数配置文件（JSON）
├── data/                # 数据集目录（msra/ 与 weibo/）
├── src/
│   ├── config.py        # 实验参数定义与读写
│   ├── dataset.py       # 数据读取、分词对齐、batch 内动态 padding
│   ├── model.py         # BertModel + 线性分类头、模型保存与加载
│   ├── metrics.py       # 实体级评估指标（reset / update / compute）
│   ├── trainer.py       # 训练循环、早停、保存最优权重与实验记录
│   └── utils.py         # 随机种子、设备选择、JSON 读写
├── train.py             # 训练与测试主入口
├── test.py              # 独立测试集评估入口
├── predict.py           # 单句实体预测入口
└── requirements.txt     # 环境依赖包列表
```

## 2. 环境依赖

所需依赖包已列在 `requirements.txt` 中，可通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 3. 训练与运行

在项目根目录下运行以下命令即可启动训练（默认开启 SwanLab 记录，并在训练结束后自动加载验证集最优模型评估测试集；关闭 SwanLab 可加 `--no_swanlab`）：

```bash
# MSRA 数据集
python train.py --config configs/msra_bert_base.json
python train.py --config configs/msra_bert_wwm.json

# Weibo 数据集
python train.py --config configs/weibo_bert_base.json
python train.py --config configs/weibo_bert_wwm.json
```

每次运行会在 `outputs/<实验名>/<run_id>/` 下保存当次配置 `config.json`、最优模型权重、训练汇总 `train_summary.json` 和测试集结果 `test_result.json`。

如需单独评估测试集或进行单句预测，直接传入对应的运行目录即可：

```bash
# 测试集评估
python test.py --checkpoint_dir outputs/msra_bert_wwm/20261011_184334

# 单句预测（不传 --text 则进入交互输入模式）
python predict.py --checkpoint_dir outputs/msra_bert_wwm/20261011_184334 --text "北京大学代表团访问微软亚洲研究院。"
```

## 4. 实验结果

SwanLab 实验看板：[https://swanlab.cn/@qychen553728/demo2-ner/v0/runs/35e7bgon/chart](https://swanlab.cn/@qychen553728/demo2-ner/v0/runs/35e7bgon/chart)

| 实验配置 | 数据集 | 预训练模型 | Best Epoch | Dev Micro-F1 | Test Precision | Test Recall | Test Micro-F1 | Test Macro-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `msra_bert_base` | MSRA (5k) | `bert-base-chinese` | 3 / 3 | 0.9246 | 0.9021 | **0.8993** | 0.9007 | 0.8996 |
| `msra_bert_wwm` | MSRA (5k) | `chinese-bert-wwm-ext` | 2 / 3 | **0.9288** | **0.9250** | 0.8803 | **0.9021** | **0.9036** |
| `weibo_bert_base` | Weibo | `bert-base-chinese` | 3 / 5 | 0.7037 | 0.6439 | 0.6555 | 0.6497 | 0.4109 |
| `weibo_bert_wwm` | Weibo | `chinese-bert-wwm-ext` | 4 / 5 | **0.7260** | **0.6692** | **0.6864** | **0.6777** | **0.4629** |

