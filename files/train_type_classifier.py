# train_type_classifier.py
import numpy as np
import random
import logging
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Custom Trainer to incorporate class weights
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute the weighted cross-entropy loss
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Main training function
def train_classifier(input_file, model_output_dir):
    set_seed()

    # Map labels to numeric IDs
    label2id = {"yes_no": 0, "entity": 1}
    id2label = {v: k for k, v in label2id.items()}

    dataset = load_dataset(
        'csv',
        data_files=input_file,
        delimiter='\t',
        column_names=["id", "text", "label"]
    )

    # Map textual labels to numeric IDs
    dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

    # Split dataset into training and validation sets
    train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train'].rename_column("label", "labels")
    val_dataset = train_test_split['test'].rename_column("label", "labels")

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Tokenize the dataset
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="longest",
            max_length=128
        )

    # Apply tokenization and format datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.remove_columns(["id", "text"])
    train_dataset.set_format("torch")

    val_dataset = val_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.remove_columns(["id", "text"])
    val_dataset.set_format("torch")

    # Compute class weights for handling class imbalance
    labels = train_dataset["labels"].numpy()
    if labels is None or len(labels) == 0:
        raise ValueError("Training set labels are empty; cannot compute class weights.")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")

    # Define metrics for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save the model at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=3,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
    )

    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate the model
    logger.info("Starting evaluation...")
    trainer.evaluate()

    # Save the model and tokenizer
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info("Model saved successfully.")


def main():
    input_file = 'question_augmented_format.csv'
    model_output_dir = "./question_type_classifier"
    train_classifier(input_file, model_output_dir)


if __name__ == "__main__":
    main()
