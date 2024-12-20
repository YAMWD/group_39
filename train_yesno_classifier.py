# train_yesno_classifier.py
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class YesNoDataset(Dataset):
    """
    A custom PyTorch Dataset for Yes/No/Uncertain classification.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initializes the dataset with texts and labels.

        Parameters:
        - texts (list): List of text samples.
        - labels (list): Corresponding list of label IDs.
        - tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the texts.
        - max_length (int): Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized text and label for a given index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - dict: Dictionary containing input IDs, attention masks, and labels.
        """
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred):
    """
    Computes evaluation metrics.

    Parameters:
    - pred (transformers.EvalPrediction): Object containing predictions and label IDs.

    Returns:
    - dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    """Main function to train the Yes/No/Uncertain classifier."""
    set_seed()

    # Define the path to your dataset
    input_file = 'yesno_dataset_format.csv'  # Replace with your actual file path

    # Define valid labels and their corresponding IDs
    label2id = {'No': 0, 'Uncertain': 1, 'Yes': 2}
    id2label = {v: k for k, v in label2id.items()}

    try:
        # Attempt to read the CSV file using multiple spaces as the delimiter
        data = pd.read_csv(
            input_file,
            sep='\s{2,}',  # Regex for two or more spaces
            engine='python',
            header=None,
            names=['serial', 'question', 'label']
        )
        print(f"Successfully loaded {len(data)} samples from '{input_file}'.")
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Map textual labels to numerical IDs
    data['label_id'] = data['label'].map(label2id)

    # Drop any rows with missing labels
    data = data.dropna(subset=['label_id'])
    print(f"After dropping missing labels, {len(data)} samples remain.")

    # Convert label IDs to integers
    data['label_id'] = data['label_id'].astype(int)

    # Extract texts and labels
    texts = data['question'].tolist()
    labels = data['label_id'].tolist()

    # Split the dataset into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels
    )
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Create dataset objects
    train_dataset = YesNoDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = YesNoDataset(val_texts, val_labels, tokenizer, max_length=128)

    # Initialize the model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./yesno_classifier',  # Output directory
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=12,  # Batch size per device during training
        per_device_eval_batch_size=12,  # Batch size for evaluation
        evaluation_strategy='epoch',  # Evaluation strategy to adopt during training
        save_strategy='epoch',  # Save checkpoint every epoch
        load_best_model_at_end=True,  # Load the best model when finished training
        metric_for_best_model='accuracy',  # Use accuracy to evaluate the best model
        greater_is_better=True,  # Whether the metric is to be maximized
        logging_dir='./logs_yesno',  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
        save_total_limit=2,  # Limit the total amount of checkpoints
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping
    )

    # Start training
    trainer.train()

    # Evaluate the model on the validation set
    results = trainer.evaluate()
    print(f"Validation results: {results}")

    # Save the trained model and tokenizer
    trainer.save_model('./yesno_classifier')
    tokenizer.save_pretrained('./yesno_classifier')
    print("Model and tokenizer saved to './yesno_classifier'.")


if __name__ == "__main__":
    main()
