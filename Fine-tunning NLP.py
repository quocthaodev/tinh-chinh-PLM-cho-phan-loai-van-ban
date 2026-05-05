import os
import gc
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

def clear_memory():
    """Hàm hỗ trợ dọn dẹp bộ nhớ RAM và VRAM (GPU) giữa các lần chạy"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =====================================================================
# PHẦN 1: TẢI VÀ LƯU MÔ HÌNH FINBERT
# =====================================================================
print("\n" + "="*50)
print("PHẦN 1: TẢI VÀ LƯU MÔ HÌNH FINBERT")
print("="*50)

model_name_finbert = "ProsusAI/finbert"
print("Đang tải mô hình FinBERT...")
model_p1 = AutoModelForSequenceClassification.from_pretrained(model_name_finbert)
tokenizer_p1 = AutoTokenizer.from_pretrained(model_name_finbert)

save_path = './nlp_project/finetuned_model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Đang lưu mô hình vào máy tính...")
model_p1.save_pretrained(save_path)
tokenizer_p1.save_pretrained(save_path)
print(f"Tuyệt vời! Mô hình đã được lưu an toàn tại: {save_path}")

# Dọn dẹp bộ nhớ trước khi sang Phần 2
del model_p1, tokenizer_p1
clear_memory()


# =====================================================================
# PHẦN 2: HUẤN LUYỆN BERT CƠ BẢN VỚI LORA
# =====================================================================
print("\n" + "="*50)
print("PHẦN 2: HUẤN LUYỆN BERT VỚI LORA")
print("="*50)

print("1. Đang tải tập dữ liệu Tin tức Tài chính...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

print("2. Đang tải Tokenizer và Base Model (bert-base-uncased)...")
model_name_bert = "bert-base-uncased"
tokenizer_p2 = AutoTokenizer.from_pretrained(model_name_bert)
model_p2 = AutoModelForSequenceClassification.from_pretrained(model_name_bert, num_labels=3)

print("3. Đang mã hóa dữ liệu văn bản...")
def tokenize_function_p2(examples):
    return tokenizer_p2(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train_p2 = dataset["train"].map(tokenize_function_p2, batched=True)
tokenized_eval_p2 = dataset["validation"].map(tokenize_function_p2, batched=True)

print("4. Cài đặt kỹ thuật LoRA (PEFT)...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)
model_p2 = get_peft_model(model_p2, lora_config)
model_p2.print_trainable_parameters()

print("5. Thiết lập hàm tính điểm Macro F1...")
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

print("6. Cấu hình các thông số Huấn luyện...")
training_args_p2 = TrainingArguments(
    output_dir="./results_lora",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10
)

trainer_p2 = Trainer(
    model=model_p2,
    args=training_args_p2,
    train_dataset=tokenized_train_p2,
    eval_dataset=tokenized_eval_p2,
    compute_metrics=compute_metrics,
)

print("7. BẮT ĐẦU HUẤN LUYỆN LoRA!")
trainer_p2.train()

# Dọn dẹp bộ nhớ trước khi sang Phần 3
del model_p2, tokenizer_p2, trainer_p2
clear_memory()


# =====================================================================
# PHẦN 3: FULL FINE-TUNING FINBERT
# =====================================================================
print("\n" + "="*50)
print("PHẦN 3: FULL FINE-TUNING FINBERT")
print("="*50)

print("1. Đang tải lại Tokenizer và Mô hình FinBERT chuyên ngành...")
tokenizer_p3 = AutoTokenizer.from_pretrained(model_name_finbert)
model_p3 = AutoModelForSequenceClassification.from_pretrained(model_name_finbert, num_labels=3)

print("2. Mã hóa dữ liệu cho FinBERT...")
def tokenize_function_p3(examples):
    return tokenizer_p3(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train_p3 = dataset["train"].map(tokenize_function_p3, batched=True)
tokenized_eval_p3 = dataset["validation"].map(tokenize_function_p3, batched=True)

print("3. Cấu hình tham số huấn luyện (Full Fine-tuning)...")
training_args_p3 = TrainingArguments(
    output_dir="./finbert_results_full",
    learning_rate=2e-5, 
    num_train_epochs=3,  
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch"
)

trainer_p3 = Trainer(
    model=model_p3,
    args=training_args_p3,
    train_dataset=tokenized_train_p3,
    eval_dataset=tokenized_eval_p3,
    compute_metrics=compute_metrics,
)

print("4. BẮT ĐẦU HUẤN LUYỆN FULL FINE-TUNING FINBERT!")
trainer_p3.train()

print("\n" + "="*50)
print("HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH!")
print("="*50)