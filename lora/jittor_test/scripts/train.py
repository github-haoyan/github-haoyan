import os
import sys
import time
import subprocess
from typing import Optional, Dict, Any

import numpy as np
import torch
import jittor as jt
import jittor.nn as nn

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
from jittor.dataset import Dataset, DataLoader

# 将项目根目录加入 sys.path，方便导入自定义 lora_jittor 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lora_jittor.layers import Linear
from lora_jittor.utils import mark_only_lora_as_trainable, lora_state_dict

# 启用 Jittor 的 CUDA 加速
jt.flags.use_cuda = 1


class HFDataset(Dataset):
    """将预处理好的 HuggingFace 数据集包装为 Jittor Dataset。"""
    def __init__(self, hf_dataset: Any):
        super().__init__()
        self.data = hf_dataset

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        item = self.data[int(idx)]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["label"],
        }

    def __len__(self) -> int:
        return len(self.data)


class DistilBertConfig:
    """手动定义 DistilBERT 配置，匹配 transformers.Config。"""
    def __init__(self):
        self.vocab_size = 30522
        self.max_position_embeddings = 512
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.num_hidden_layers = 6
        self.num_labels = 2


class MultiHeadAttention(nn.Module):
    """多头注意力层，Q/V 上集成 LoRA 低秩适配。"""
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_lin = Linear(config.hidden_size, config.hidden_size,
                            r=4, lora_alpha=8, lora_dropout=0.05)
        self.k_lin = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_lin = Linear(config.hidden_size, config.hidden_size,
                            r=4, lora_alpha=8, lora_dropout=0.05)
        self.out_lin = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def execute(self, x: jt.Var, mask: Optional[jt.Var] = None) -> jt.Var:
        B, T, _ = x.shape
        # 线性变换并拆分多头
        q = self.q_lin(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_lin(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,3,1)
        v = self.v_lin(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        # 计算注意力得分
        scores = jt.matmul(q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            fill = jt.full_like(scores, -1e9)
            scores = jt.where(mask.unsqueeze(1).unsqueeze(1)==0, fill, scores)
        attn = nn.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = jt.matmul(attn, v).permute(0,2,1,3).reshape(B, T, -1)
        return self.out_lin(out)


class TransformerBlock(nn.Module):
    """Transformer Block：多头注意力 + 前馈网络 + 残差 + LayerNorm + Dropout。"""
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, x: jt.Var, mask: Optional[jt.Var] = None) -> jt.Var:
        # Attention + 残差 + LayerNorm
        y = self.attn(x, mask)
        x = self.ln1(x + self.dropout(y))
        # 前馈 + 残差 + LayerNorm
        z = self.ffn(x)
        return self.ln2(x + self.dropout(z))


class DistilBertModel(nn.Module):
    """DistilBERT 主干：embedding + 多层 TransformerBlock。"""
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, input_ids: jt.Var, attention_mask: Optional[jt.Var] = None) -> jt.Var:
        B, T = input_ids.shape
        pos = jt.arange(T).unsqueeze(0).expand_as(input_ids)
        x = self.embeddings(input_ids) + self.position_embeddings(pos)
        x = self.embeddings_layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class DistilBertForSequenceClassification(nn.Module):
    """序列分类模型：DistilBERT + 分类头。"""
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.bert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self,
                input_ids: jt.Var,
                attention_mask: Optional[jt.Var] = None,
                labels: Optional[jt.Var] = None
               ) -> Any:
        # 取 [CLS] 向量做分类
        hidden = self.bert(input_ids, attention_mask)[:, 0]
        x = nn.relu(self.pre_classifier(hidden))
        x = self.dropout(x)
        logits = self.classifier(x)
        if labels is not None:
            loss = nn.cross_entropy_loss(logits, labels, reduction="mean")
            return loss, logits
        return logits


def convert_pt_to_jt(pt_state_dict: Dict[str, torch.Tensor]) -> Dict[str, jt.Var]:
    """
    将 PyTorch 模型权重映射并转换为 Jittor 格式的 state_dict。
    """
    jt_state: Dict[str, jt.Var] = {}
    for k, v in pt_state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        name = (k
                .replace("distilbert.embeddings.word_embeddings.weight", "bert.embeddings.weight")
                .replace("distilbert.embeddings.position_embeddings.weight", "bert.position_embeddings.weight")
                .replace("distilbert.embeddings.LayerNorm.weight", "bert.embeddings_layer_norm.weight")
                .replace("distilbert.embeddings.LayerNorm.bias", "bert.embeddings_layer_norm.bias")
                .replace("distilbert.transformer.layer.", "bert.layers.")
                .replace(".attention.", ".attn.")
                .replace("sa_layer_norm.weight", "ln1.weight")
                .replace("sa_layer_norm.bias", "ln1.bias")
                .replace("output_layer_norm.weight", "ln2.weight")
                .replace("output_layer_norm.bias", "ln2.bias")
                .replace(".ffn.lin1.weight", ".ffn.0.weight")
                .replace(".ffn.lin1.bias", ".ffn.0.bias")
                .replace(".ffn.lin2.weight", ".ffn.2.weight")
                .replace(".ffn.lin2.bias", ".ffn.2.bias"))
        jt_state[name] = jt.array(v.cpu().numpy())
    return jt_state


class TrainingArguments:
    """简单封装训练超参数。"""
    def __init__(self,
                 output_dir: str,
                 per_device_train_batch_size: int = 16,
                 gradient_accumulation_steps: int = 1,
                 learning_rate: float = 5e-5,
                 num_train_epochs: int = 3,
                 logging_dir: Optional[str] = None,
                 report_to: Optional[str] = None):
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_dir = logging_dir
        self.report_to = report_to


class JittorTrainer:
    """封装训练循环、日志记录及可选的 TensorBoard 输出。"""
    def __init__(self,
                 model: nn.Module,
                 args: TrainingArguments,
                 train_loader: DataLoader):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.optimizer = jt.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        os.makedirs(args.output_dir, exist_ok=True)
        self.writer = (SummaryWriter(log_dir=args.logging_dir)
                       if args.report_to == "tensorboard" else None)

    def train(self) -> None:
        self.model.train()
        acc_steps = self.args.gradient_accumulation_steps
        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            start_time = time.time()
            gpu_utils, gpu_mems = [], []
            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].int64()
                attention_mask = batch["attention_mask"].int64()
                labels = batch["labels"].int64()

                loss, logits = self.model(input_ids, attention_mask, labels)
                loss = loss / acc_steps
                self.optimizer.backward(loss)

                if (step + 1) % acc_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # TensorBoard 日志
                    if self.writer and global_step % 100 == 0:
                        preds = logits.numpy().argmax(axis=1)
                        acc = float((preds == labels.numpy()).mean())
                        self.writer.add_scalar("train/accuracy", acc, global_step)
                        self.writer.add_scalar("train/loss", (loss * acc_steps).item(), global_step)

                    # 采集 GPU 利用率和显存
                    try:
                        out = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                             "--format=csv,nounits,noheader"],
                            encoding="utf-8"
                        ).strip().split(", ")
                        gpu_utils.append(float(out[0]))
                        gpu_mems.append(float(out[1]))
                    except Exception:
                        pass

            epoch_time = time.time() - start_time
            if self.writer:
                self.writer.add_scalar("epoch/time", epoch_time, epoch)
                self.writer.add_scalar("epoch/gpu_utilization",
                                       np.mean(gpu_utils) if gpu_utils else 0.0, epoch)
                self.writer.add_scalar("epoch/gpu_memory",
                                       np.mean(gpu_mems) if gpu_mems else 0.0, epoch)
        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    # 加载和格式化数据集
    data = load_from_disk("./data/processed_sst2")
    data["train"].set_format(type="numpy")

    # 初始化模型和权重转换
    config = DistilBertConfig()
    model = DistilBertForSequenceClassification(config)
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=config.num_labels
    )
    model.load_state_dict(convert_pt_to_jt(hf_model.state_dict()))

    # 仅微调 LoRA 层和分类头
    mark_only_lora_as_trainable(model, bias="all")
    for p in model.pre_classifier.parameters():
        p.start_grad()
    for p in model.classifier.parameters():
        p.start_grad()

    # 训练参数
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=5,
        logging_dir="./results/logs",
        report_to="tensorboard",
    )

    # 构建 DataLoader
    train_ds = HFDataset(data["train"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0
    )

    # 启动训练
    trainer = JittorTrainer(model, args, train_loader)
    trainer.train()

    # 保存 LoRA 权重和分类头权重
    os.makedirs("./models", exist_ok=True)
    peft_state = lora_state_dict(model, bias="all")
    peft_state.update({
        'pre_classifier.weight': model.pre_classifier.weight.data,
        'pre_classifier.bias':   model.pre_classifier.bias.data,
        'classifier.weight':     model.classifier.weight.data,
        'classifier.bias':       model.classifier.bias.data
    })
    jt.save(peft_state, "./models/lora_weights.pkl")
    print("训练完成，使用 tensorboard --logdir=./results/logs 查看日志")