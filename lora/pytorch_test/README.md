# LoRA on DistilBERT (PyTorch)

本项目展示如何使用 PyTorch + PEFT 实现 LoRA（Low-Rank Adaptation），并在 GLUE 基准的 SST-2 数据集上微调 DistilBERT，最后在验证集上评估性能。

## 环境要求

- 操作系统：Windows 10/11 或 Linux  
- Python：3.8 及以上  
- CUDA：11.2  
- cuDNN：8.9  
- PyTorch：2.1.0+cu118  
- 依赖库：  
  - transformers  
  - datasets  
  - peft  
  - scikit-learn  
  - tensorboard  

安装示例（Windows + CUDA 11.8）：  
```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft scikit-learn tensorboard
```

## 项目结构

```
pytorch_test/
├── data/
│   └── processed_sst2/     # 分词并保存后的 SST-2 train/validation/test
├── results/
│   ├── logs/               # TensorBoard 日志
│   └── model/              # 最终保存的 LoRA Adapter + 基础模型权重
├── scripts/
│   ├── data_process.py     # 数据预处理脚本
│   ├── train.py            # LoRA 微调训练脚本
│   └── evaluate.py         # 模型评估脚本
└── README.md               # 本说明文件
```

---

## 1. 数据预处理

脚本：`scripts/data_process.py`

功能：
- 加载 GLUE SST-2（训练/验证/测试）  
- 定义 `tokenize_fn`：使用 `AutoTokenizer` 将 `sentence` 分词  
  - padding 到 `max_length=128`，截断超长序列  
- map 批量调用，删除原始 `sentence` 列  
- 将处理结果保存到 `./data/processed_sst2`

运行命令：  
```bash
python scripts/data_process.py
```

---

## 2. 模型训练

脚本：`scripts/train.py`

主要步骤：

1. **构造 Dataset & DataLoader**  
   - 自定义 `HFDataset`：将 HF 数据集封装为 PyTorch `Dataset`，在 `__getitem__` 中转换为 `torch.tensor`  
   - `collate_fn`：批次内 `input_ids`/`attention_mask`/`labels` 堆叠  
   - DataLoader 示例：  
     ```python
     train_loader = DataLoader(
       train_ds,
       batch_size=16,
       shuffle=True,
       num_workers=4,
       pin_memory=True,
       collate_fn=collate_fn
     )
     ```

2. **LoRA Adapter 注入**  
   ```python
   from peft import LoraConfig, get_peft_model
   model = AutoModelForSequenceClassification.from_pretrained(
     "distilbert-base-uncased", num_labels=2
   )
   lora_cfg = LoraConfig(
     r=4,
     lora_alpha=8,
     target_modules=["q_lin", "v_lin"],
     lora_dropout=0.05,
     modules_to_save=["classifier"],
   )
   model = get_peft_model(model, lora_cfg)
   model.to(device)
   ```

3. **训练循环**  
   - 梯度累计步数 = 2  
   - 混合精度 (AMP) + `GradScaler`  
   - 每 100 步记录训练 `loss` 和 `accuracy` 至 TensorBoard  
   - 调用 `nvidia-smi` 采集 GPU 利用率和显存占用  
   - 训练结束后打印整体训练耗时  

4. **日志与模型保存**  
   ```bash
   # TensorBoard 可视化
   tensorboard --logdir=./results/logs
   ```
   ```python
   # 保存最终模型（包含 LoRA Adapter）
   save_dir = r"D:\lora\pytorch_test\models"
   model.save_pretrained(save_dir)
   ```

运行命令：  
```bash
python scripts/train.py
```

---

## 3. 模型评估

脚本：`scripts/evaluate.py`

主要流程：

1. **加载基础模型 & 分词器**  
   ```python
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   base_model = AutoModelForSequenceClassification.from_pretrained(
     "distilbert-base-uncased", num_labels=2
   )
   ```

2. **加载 LoRA Adapter 并合并**  
   ```python
   from peft import PeftModel
   model = PeftModel.from_pretrained(base_model, r"D:\lora\pytorch_test\models")
   model = model.merge_and_unload()  # 合并权重并卸载 Adapter
   model.to(device).eval()
   ```

3. **加载验证集**  
   ```python
   ds = load_from_disk("./data/processed_sst2")["validation"]
   input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
   attention_mask = torch.tensor(ds["attention_mask"], dtype=torch.long)
   labels = ds["label"]
   ```

4. **逐样本推理 & 计算准确率**  
   ```python
   preds = []
   with torch.no_grad():
     for i in range(len(input_ids)):
       ids = input_ids[i].unsqueeze(0).to(device)
       mask = attention_mask[i].unsqueeze(0).to(device)
       outputs = model(input_ids=ids, attention_mask=mask)
       preds.append(outputs.logits.argmax(dim=-1).item())
   from sklearn.metrics import accuracy_score
   acc = accuracy_score(labels, preds)
   print(f"验证集准确率: {acc:.2%}")
   ```

运行命令：  
```bash
python scripts/evaluate.py
```

---

## 自定义与常见问题

- 修改 LoRA 超参：在 `scripts/train.py` 中调整 `LoraConfig(r, lora_alpha, target_modules, lora_dropout)`  
- 调整训练超参：批量大小、学习率、轮数可通过脚本中 `args_bs`、`acc_steps`、`num_epochs` 修改  
- 若遇到 GPU 不可用，请检查 `torch.cuda.is_available()` 与 CUDA 驱动版本  
- 日志目录 & 模型保存路径可在脚本顶部或最后 `save_pretrained` 路径中修改

## 致谢

- LoRA 原论文：“Low-Rank Adaptation of Large Language Models”  
- HuggingFace Transformers & Datasets  
- PEFT + PyTorch 社区  