# LoRA on DistilBERT (Jittor)

本项目演示如何在 Jittor 中实现 LoRA（Low-Rank Adaptation），并在 GLUE 基准的 SST-2 数据集上微调 DistilBERT，最终进行验证。

## 环境依赖

- 操作系统：Windows / Linux  
- Python ≥ 3.9  
- CUDA ≥ 11.2 + cuDNN ≥ 8.0（如需 GPU）  
- Jittor  
- PyTorch  
- transformers  
- datasets  
- peft  
- numpy  
- tensorboard  

安装依赖：  
```bash
pip install jittor torch transformers datasets peft numpy tensorboard
```

## 目录结构

```
d:\lora\jittor_test
├── data
│   └── processed_sst2            # 预处理后保存的 SST-2 数据集
├── models
│   └── lora_weights.pkl          # 保存的 LoRA 权重
├── results
│   └── logs                      # TensorBoard 日志
├── scripts
│   ├── data_process.py           # 数据加载与预处理脚本
│   ├── train.py                  # 训练脚本（Jittor + LoRA）
│   └── evaluate.py               # 验证脚本
├── lora_jittor
│   ├── layers.py                 # LoRA 核心层实现
│   └── utils.py                  # 冻结/提取 LoRA 参数工具
└── README.md                     # 项目说明
```

## 脚本说明

### scripts/data_process.py

- 功能：加载 GLUE SST-2，分词并保存到 `data/processed_sst2`  
- 使用：  
  ```bash
  python scripts/data_process.py
  ```  
- 输出：  
  - `./data/processed_sst2/train`  
  - `./data/processed_sst2/validation`

### scripts/train.py

- 功能：基于 Jittor 与 LoRA 对 DistilBERT 进行微调  
- 默认配置：  
  - LoRA rank = 4, alpha = 8, dropout = 0.05  
  - batch size = 16, 梯度累积步数 = 2  
  - 学习率 = 2e-4, 训练轮数 = 5  
- 使用：  
  ```bash
  python scripts/train.py
  ```  
- 输出：  
  - `./models/lora_weights.pkl` —— 保存 LoRA 增量权重与分类头权重  
  - `./results/logs` —— TensorBoard 日志  

### scripts/evaluate.py

- 功能：加载基础模型 + LoRA 权重，对验证集做推理并计算准确率  
- 默认 batch size = 8  
- 使用：  
  ```bash
  python scripts/evaluate.py
  ```  
- 输出：  
  - 控制台打印验证集准确率，例如 `验证集准确率: 92.30%`

## 快速开始

1. 数据预处理  
   ```bash
   python scripts/data_process.py
   ```
2. 模型训练  
   ```bash
   python scripts/train.py
   ```
3. 可视化训练过程（可选）  
   ```bash
   tensorboard --logdir=results/logs --port=6006
   ```
   浏览器访问 `http://localhost:6006`  
4. 模型评估  
   ```bash
   python scripts/evaluate.py
   ```

## 自定义配置

- 修改 LoRA 超参：编辑 `scripts/train.py` 中 `Linear(..., r=…, lora_alpha=…, lora_dropout=…)`  
- 调整训练超参：在 `scripts/train.py` 中 `TrainingArguments` 构造时设置 `per_device_train_batch_size`、`learning_rate`、`num_train_epochs` 等  
- 推理时仅用基础模型：可注释或删除 `evaluate.py` 中加载 LoRA 权重部分  

## 常见问题

- 如果遇到 GPU 不可用，检查 `jt.flags.use_cuda = 1` 是否生效或更新 Jittor 版本  
- 权重加载报错，请确认 `lora_weights.pkl` 路径是否正确，并使用相同的模型配置  

## 致谢

- LoRA 原论文：“LoRA: Low-Rank Adaptation of Large Language Models”  
- HuggingFace Transformers & Datasets  
- Jittor 框架  