import os                                    # 用于目录创建、文件操作
import time                                  # 用于计时
import subprocess                            # 用于调用外部命令（nvidia-smi 监控 GPU）

import numpy as np                           # 用于计算平均值等数值统计
import torch                                 # PyTorch 主框架
from torch.utils.data import Dataset, DataLoader   # PyTorch 数据集与数据加载器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志记录
from transformers import AutoModelForSequenceClassification  # 预训练模型接口
from peft import LoraConfig, get_peft_model   # LoRA 微调配置与接口
from datasets import load_from_disk           # 加载已保存的 HuggingFace 数据集

# 优化 cuDNN 性能（对输入大小固定时加速卷积等操作）
torch.backends.cudnn.benchmark = True
# 自动选择可用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFDataset(Dataset):
    """
    将 HuggingFace Dataset 格式数据封装成 PyTorch Dataset。
    该类负责将原始字典数据转换为 PyTorch 张量。
    """
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def collate_fn(batch):
    """
    自定义批次拼接函数，将一个 batch 中的多个样本在第 0 维度拼接。
    """
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. 加载预处理好的训练数据
    # -------------------------------------------------------------------------
    raw = load_from_disk("./data/processed_sst2")
    train_ds = HFDataset(raw["train"])

    # -------------------------------------------------------------------------
    # 2. 构造 PyTorch DataLoader
    # -------------------------------------------------------------------------
    args_bs = 16
    acc_steps = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=args_bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # -------------------------------------------------------------------------
    # 3. 构建并初始化 LoRA 微调模型
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4. 构造优化器、混合精度工具和 TensorBoard 日志
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir="./results/logs")

    global_step = 0
    num_epochs = 5

    # -------------------------------------------------------------------------
    # *Modified*: 记录训练循环开始时间（不包含模型加载等前置消耗）
    # -------------------------------------------------------------------------
    train_loop_start = time.time()

    # -------------------------------------------------------------------------
    # 5. 训练循环
    # -------------------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        gpu_utils, gpu_mems = [], []

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / acc_steps
                logits = outputs.logits

            scaler.scale(loss).backward()

            if (step + 1) % acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                loss_item = (loss.item() * acc_steps)
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean().item()

                if global_step % 100 == 0:
                    writer.add_scalar("train/loss", loss_item, global_step)
                    writer.add_scalar("train/accuracy", acc, global_step)

                try:
                    out = subprocess.check_output(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,nounits,noheader"],
                        encoding="utf-8"
                    ).strip().split(", ")
                    gpu_utils.append(float(out[0]))
                    gpu_mems.append(float(out[1]))
                except:
                    pass

        # ---------------------------------------------------------------------
        # 6. 记录并写入该 epoch 的平均指标
        # ---------------------------------------------------------------------
        epoch_time = time.time() - epoch_start
        util_avg = np.mean(gpu_utils) if gpu_utils else 0.0
        mem_avg = np.mean(gpu_mems) if gpu_mems else 0.0

        writer.add_scalar("epoch/epoch_time", epoch_time, epoch)
        writer.add_scalar("epoch/gpu_utilization", util_avg, epoch)
        writer.add_scalar("epoch/gpu_memory", mem_avg, epoch)

    # -------------------------------------------------------------------------
    # *Modified*: 记录训练循环结束时间，并打印总耗时
    # -------------------------------------------------------------------------
    train_loop_end = time.time()
    print(f"Training loop time (excluding model loading): {train_loop_end - train_loop_start:.2f}s")

    writer.close()

    # -------------------------------------------------------------------------
    # 7. 保存最终微调模型
    # -------------------------------------------------------------------------
    save_dir = r"D:\lora\pytorch_test\models"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"训练完成，模型已保存至 {save_dir}，使用 tensorboard --logdir=./results/logs 查看日志")