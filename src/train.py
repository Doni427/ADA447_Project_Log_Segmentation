import os
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from tokenizer_utils import tokenize_and_align_labels

# Load metrics via seqeval
metric = evaluate.load("seqeval")
label_list = ["O", "B-PAR"] # 0: Text, 1: Parameter

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    dataset_path = "data/processed_logs.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset profile! Run python src/dataset_processor.py first.")

    # 1. Veriyi düz bir Dataset olarak yükle
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 2. Dataset nesnesinin kendi içindeki train_test_split metodunu dogrudan çagir
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # 3. Map fonksiyonunu split_dataset üzerinden çalistir
    tokenized_datasets = split_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=split_dataset["train"].column_names
    )

    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=len(label_list)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # use_mps_device=torch.backends.mps.is_available()
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting training pipeline...")
    trainer.train()

    # Save finalized model outputs
    output_dir = "./models/fine_tuned_logbert/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" Model artifact exported to {output_dir}")

if __name__ == "__main__":
    main()