from datasets import load_dataset         # 从 HuggingFace Datasets 加载数据集
from transformers import AutoTokenizer    # 从 Transformers 导入分词器

# 实例化 DistilBERT 分词器，用于将文本转换为输入模型所需的 token ids
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(examples):
    """
    批量分词函数，用于 dataset.map 调用
    参数:
        examples: dict，包含键 "sentence"（文本列表）和 "label"（对应标签列表）
    返回:
        dict，包含三项：
          - "input_ids": 分词后的 token id 列表
          - "attention_mask": attention mask 列表（1 表示真实 token，0 表示 padding）
          - "label": 原始标签列表，保持不变
    """
    # 对每条 sentence 批量调用 tokenizer
    outputs = tokenizer(
        examples["sentence"],   # 文本列表，长度等于 batch 大小
        padding="max_length",   # 将所有序列填充到相同的 max_length
        truncation=True,        # 截断超过 max_length 的文本
        max_length=128          # 指定最大序列长度
    )
    # 返回新生成的字段，同时保留原始 label
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
        "label": examples["label"],
    }

if __name__ == "__main__":
    # 1. 加载 SST-2 原始数据集
    print("正在加载 SST-2 数据集…")
    # load_dataset 会返回一个 DatasetDict，包含 train/validation/test 三个子集
    dataset = load_dataset("glue", "sst2")

    # 2. 执行分词并预处理
    print("正在处理数据…")
    # 使用 map 批量调用 tokenize_fn，batched=True 表示 examples 是一个 batch
    # remove_columns 指定删除原始文本列，节省存储空间
    tokenized_data = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["sentence"]
    )

    # 3. 将处理后的数据保存到本地，以便后续训练和评估直接加载
    print("正在保存预处理结果…")
    tokenized_data.save_to_disk("./data/processed_sst2")
    print("已将预处理数据保存到 ./data/processed_sst2")