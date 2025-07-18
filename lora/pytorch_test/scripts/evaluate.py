import torch                                        # PyTorch 主框架，用于张量运算与模型推理
from transformers import AutoModelForSequenceClassification, AutoTokenizer
                                                   # 从 Transformers 导入预训练模型和分词器接口
from peft import PeftModel                         # PEFT 包，用于加载 LoRA 适配器
from datasets import load_from_disk                # HuggingFace Datasets，用于加载预处理数据
from sklearn.metrics import accuracy_score         # 计算模型准确率

if __name__ == "__main__":
    # -----------------------------
    # 1. 加载基础模型和分词器
    # -----------------------------
    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    # -----------------------------
    # 2. 加载 LoRA 适配器并合并
    # -----------------------------
    # 修改：调整 LoRA 权重路径为训练脚本保存目录
    lora_weights_path = r"D:\lora\pytorch_test\models"
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    # 合并 LoRA 微调的低秩权重到基础模型，并卸载 adapter
    model = model.merge_and_unload()

    # -----------------------------
    # 3. 模型准备：移动设备并设为评估模式
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -----------------------------
    # 4. 加载预处理好的验证集
    # -----------------------------
    ds = load_from_disk("./data/processed_sst2")["validation"]
    input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(ds["attention_mask"], dtype=torch.long)
    labels = ds["label"]

    # -----------------------------
    # 5. 批量推理并收集预测结果
    # -----------------------------
    preds = []
    with torch.no_grad():
        for i in range(len(input_ids)):
            ids = input_ids[i].unsqueeze(0).to(device)
            mask = attention_mask[i].unsqueeze(0).to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            pred_label = logits.argmax(dim=-1).cpu().item()
            preds.append(pred_label)

    # -----------------------------
    # 6. 计算并打印准确率
    # -----------------------------
    acc = accuracy_score(labels, preds)
    print(f"验证集准确率: {acc:.2%}")