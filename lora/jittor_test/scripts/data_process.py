from datasets import load_dataset       # 从 HuggingFace datasets 库加载数据集
from transformers import AutoTokenizer  # 导入 HuggingFace 分词器

# 使用 DistilBERT 的分词器实例，基于预训练模型 distilbert-base-uncased
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(examples):
    """
    对传入的批量文本示例进行分词处理
    参数:
        examples: dict, 包含 "sentence" 和 "label" 两个字段的批量数据
    返回:
        dict, 包含分词后的 "input_ids", "attention_mask" 和原始 "label"
    """
    # 将文本句子批量转为 token id，统一填充到 max_length=128，并截断超长部分
    outputs = tokenizer(
        examples["sentence"],   # 文本内容列表
        padding="max_length",   # 使用最大长度进行填充
        truncation=True,        # 截断超过最大长度的文本
        max_length=128          # 最大序列长度
    )
    # 返回分词结果和标签
    return {
        "input_ids": outputs["input_ids"],          # 分词后的 token id 列表
        "attention_mask": outputs["attention_mask"],# attention mask 列表
        "label": examples["label"],                 # 保留原始标签
    }

if __name__ == "__main__":
    # 主流程: 加载、分词并保存预处理数据
    print("正在加载 SST-2 数据集…")
    dataset = load_dataset("glue", "sst2")  # 加载 GLUE benchmark 中的 SST-2 数据集

    print("正在处理数据…")
    tokenized_data = dataset.map(
        tokenize_fn,               # 应用自定义的 tokenize_fn 函数
        batched=True,              # 批量处理，提高效率
        remove_columns=["sentence"]# 处理后删除原始的 "sentence" 列
    )

    print("正在保存预处理结果…")
    tokenized_data.save_to_disk("./data/processed_sst2")  # 将处理后的数据保存到本地磁盘
    print("已将预处理数据保存到 ./data/processed_sst2")