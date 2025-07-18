import os
import numpy as np
import jittor as jt
from datasets import load_from_disk                 # 加载预处理好的 HuggingFace 数据集
from jittor.dataset import DataLoader, Dataset       # Jittor 的 DataLoader 和 Dataset
from transformers import AutoModelForSequenceClassification  # 用于下载 PyTorch 预训练模型权重
from train import DistilBertConfig, \
                  DistilBertForSequenceClassification, \
                  convert_pt_to_jt  # 导入模型配置类、自定义模型与权重转换函数

# 开启 Jittor GPU 计算
jt.flags.use_cuda = 1

class HFDataset(Dataset):
    """包装预处理好的 HF dataset，使其可被 Jittor DataLoader 使用"""
    def __init__(self, hf_ds):
        super().__init__()
        self.data = hf_ds  # 接受一个 DatasetDict 子集（train/validation/test）

    def __getitem__(self, idx):
        # Jittor 可能传入浮点索引，需要转换为 int
        item = self.data[int(idx)]
        # 返回字典格式，DataLoader 会按键合并成 batch
        return {
            "input_ids": item["input_ids"],         # 预先分词好的 token id 列表
            "attention_mask": item["attention_mask"],# attention mask 列表
            "labels": item["label"],                 # 情感标签
        }

    def __len__(self):
        # 返回样本数
        return len(self.data)

if __name__ == "__main__":
    # 1. 加载已预处理的验证集
    data = load_from_disk("./data/processed_sst2")
    # 将 Dataset 格式设置为 numpy，方便 Jittor DataLoader 直接读取
    data["validation"].set_format(type="numpy")
    # 构造 Jittor 数据集与 DataLoader
    val_ds = HFDataset(data["validation"])
    val_loader = DataLoader(
        val_ds,
        batch_size=8,     # 推理时每批 8 个样本
        shuffle=False     # 验证时不需要打乱
    )

    # 2. 初始化模型并加载预训练权重
    config = DistilBertConfig()  # 构造 DistilBERT 配置
    model = DistilBertForSequenceClassification(config)  # 实例化自定义 Jittor 模型
    # 从 HuggingFace Hub 下载 PyTorch 版 DistilBERT，并提取 state_dict
    hf_pt = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=config.num_labels
    )
    # 将 PyTorch 权重转换并加载到 Jittor 模型
    jt_state = convert_pt_to_jt(hf_pt.state_dict())
    model.load_state_dict(jt_state)

    # 3. 加载 LoRA 微调后的权重（如果已存在）
    peft_path = "./models/lora_weights.pkl"
    if os.path.exists(peft_path):
        peft_state = jt.load(peft_path)   # 从文件读取 LoRA 权重
        state = model.state_dict()        # 原模型参数
        state.update(peft_state)          # 覆盖 LoRA 子层与分类头权重
        model.load_state_dict(state)      # 重新加载全模型权重

    # 切换模型到评估模式，关闭 Dropout 等训练专用行为
    model.eval()

    # 4. 批量推理并收集预测与真实标签
    all_preds = []
    all_labels = []
    for batch in val_loader:
        # 将 numpy batch 转为 jt.array，放到 GPU 上
        input_ids = jt.array(batch["input_ids"])
        attention_mask = jt.array(batch["attention_mask"])
        labels = batch["labels"]  # 真实标签保留 numpy 数组
        # 前向计算 logits (inference 模式不返回 loss)
        logits = model(input_ids, attention_mask)
        # 收集 logits 和 labels
        all_preds.append(logits.data)  # jt.Tensor.data 转为 numpy
        all_labels.append(labels)

    # 5. 合并所有批次并计算准确率
    logits_all = np.vstack(all_preds)                # (N, num_labels)
    labels_all = np.concatenate(all_labels)          # (N,)
    pred_ids = np.argmax(logits_all, axis=-1)        # 选最大值索引
    accuracy = (pred_ids == labels_all).mean()       # 计算准确率

    # 6. 打印结果
    print(f"验证集准确率: {accuracy:.2%}")