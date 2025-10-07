"""Train fine-tuned BERT model for query routing using brooks_queries.csv."""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
from typing import Dict


class QueryDataset(Dataset):
    """Dataset for query classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(csv_path: str = "brooks_queries.csv"):
    """Load and prepare data from CSV."""
    print(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create category to ID mapping
    categories = sorted(df['label'].unique())
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    id_to_category = {idx: cat for cat, idx in category_to_id.items()}
    
    print(f"📊 Total samples: {len(df)}")
    print(f"📋 Categories: {categories}")
    print("\n📊 Label distribution:")
    print(df['label'].value_counts())
    
    # Convert labels to IDs
    texts = df['text'].tolist()
    labels = [category_to_id[label] for label in df['label']]
    
    return texts, labels, category_to_id, id_to_category


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def train_model(
    csv_path: str = "brooks_queries.csv",
    model_name: str = "bert-base-uncased",
    output_dir: str = "./models/fine_tuned_router",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    test_size: float = 0.2,
    seed: int = 42
):
    """Train the router model."""
    
    # Load data
    texts, labels, category_to_id, id_to_category = load_data(csv_path)
    
    # Split data
    print(f"\n🔀 Splitting data (test_size={test_size})...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=seed, 
        stratify=labels
    )
    
    # Further split training into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=0.1,
        random_state=seed,
        stratify=train_labels
    )
    
    print(f"✅ Training samples: {len(train_texts)}")
    print(f"✅ Validation samples: {len(val_texts)}")
    print(f"✅ Test samples: {len(test_texts)}")
    
    # Load tokenizer and model
    print(f"\n🔄 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(category_to_id)
    )
    
    # Detect device (MPS for Mac, CUDA for others, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("⚠️ Using CPU (GPU not available)")
    
    # Create datasets
    train_dataset = QueryDataset(train_texts, train_labels, tokenizer)
    val_dataset = QueryDataset(val_texts, val_labels, tokenizer)
    test_dataset = QueryDataset(test_texts, test_labels, tokenizer)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mappings
    import json
    mappings = {
        'category_to_id': category_to_id,
        'id_to_category': id_to_category
    }
    with open(f"{output_dir}/label_mappings.json", 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Training arguments optimized for Mac GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        warmup_steps=200,
        save_total_limit=2,
        fp16=False,  # MPS doesn't support fp16 yet
        use_cpu=False if device != "cpu" else True,
        dataloader_num_workers=0,  # MPS works better with 0
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n" + "="*80)
    print("🚀 Starting training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("📊 Validation Set Evaluation")
    print("="*80)
    val_results = trainer.evaluate(val_dataset)
    print(f"Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("📊 Test Set Evaluation")
    print("="*80)
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Detailed evaluation
    print("\n🔍 Generating detailed classification report...")
    model.eval()
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Classification report
    print("\n" + "="*80)
    print("📋 Classification Report")
    print("="*80)
    target_names = [id_to_category[i] for i in range(len(id_to_category))]
    print(classification_report(test_labels, pred_labels, target_names=target_names))
    
    # Confusion matrix
    print("\n" + "="*80)
    print("🔢 Confusion Matrix")
    print("="*80)
    cm = confusion_matrix(test_labels, pred_labels)
    print("      ", "  ".join([f"{cat:12s}" for cat in target_names]))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:12s}", "  ".join([f"{val:12d}" for val in row]))
    
    # Save test results
    results = {
        'validation_accuracy': val_results['eval_accuracy'],
        'test_accuracy': test_results['eval_accuracy'],
        'num_train_samples': len(train_texts),
        'num_val_samples': len(val_texts),
        'num_test_samples': len(test_texts),
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    
    with open(f"{output_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print(f"📁 Model saved to: {output_dir}")
    print(f"🎯 Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print("="*80)
    
    return model, tokenizer, category_to_id, id_to_category


if __name__ == "__main__":
    # Train the model
    model, tokenizer, category_to_id, id_to_category = train_model(
        csv_path="brooks_queries.csv",
        model_name="bert-base-uncased",
        output_dir="./models/fine_tuned_router",
        num_epochs=5,
        batch_size=16,
        learning_rate=2e-5,
        test_size=0.2,
        seed=42
    )
