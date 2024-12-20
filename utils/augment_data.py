# augment_data.py

import random
import nlpaug.augmenter.word as naw
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

def set_seed(seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

def load_data(file_path):
    # Load data from a tab-separated file
    try:
        data = pd.read_csv(file_path, sep='\t', names=["id", "text", "label"])
        print(f"Loaded {len(data)} samples from '{file_path}'.")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit(1)

def initialize_device():
    # Determine the device to use (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def augment_sentence(text, augmenter_list):
    # Perform augmentation using a list of augmenters
    augmented_texts = []
    for aug in augmenter_list:
        try:
            augmented = aug.augment(text)
            augmented_texts.append(augmented)
        except Exception as e:
            print(f"Augmentation failed for text: {text[:50]}... Error: {e}")
            augmented_texts.append(text)  # Fallback to original text
    return augmented_texts

def load_translation_model(src_lang, tgt_lang, device):
    # Load translation model for specified source and target languages
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        print(f"Loaded translation model: {model_name}")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load translation model {model_name}: {e}")
        exit(1)

def translate(texts, tokenizer, model, device, batch_size=16):
    # Translate a batch of texts using the given model and tokenizer
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs)
        translated_batch = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_texts.extend(translated_batch)
    return translated_texts

def back_translate_batch(texts, tokenizer_fr, model_fr, tokenizer_en, model_en, device, batch_size=16):
    # Translate from English to French
    try:
        translated_fr = translate(texts, tokenizer_fr, model_fr, device, batch_size)
    except Exception as e:
        print(f"Batch translation to French failed: {e}")
        translated_fr = texts  # Fallback to original texts

    # Translate back from French to English
    try:
        back_translated = translate(translated_fr, tokenizer_en, model_en, device, batch_size)
    except Exception as e:
        print(f"Batch back translation to English failed: {e}")
        back_translated = translated_fr  # Fallback to translated texts

    return back_translated

def process_row(row, augmenter_list, tokenizer_fr, model_fr, tokenizer_en, model_en, device):
    # Process a single row of data to perform augmentation
    original_id = row['id']
    original_text = row['text'][:512]  # Limit text length
    label = row['label']

    # Perform synonym replacement and random deletion
    augmented_texts = augment_sentence(original_text, augmenter_list)

    # Perform back-translation
    try:
        augmented_bt = back_translate_batch([original_text], tokenizer_fr, model_fr, tokenizer_en, model_en, device, batch_size=1)[0]
        augmented_texts.append(augmented_bt)
    except Exception as e:
        print(f"Back translation failed for {original_id}: {e}")

    augmented_entries = []
    for i, aug_text in enumerate(augmented_texts):
        new_id = f"{original_id}_aug_{i + 1}"
        augmented_entries.append({"id": new_id, "text": aug_text, "label": label})

    return augmented_entries

def augment_data(input_df, augmenters, tokenizer_fr, model_fr, tokenizer_en, model_en, device, output_file):
    # Perform data augmentation on the entire dataset
    augmented_data = []
    print("Starting data augmentation...")

    for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Augmenting"):
        augmented_entries = process_row(row, augmenters, tokenizer_fr, model_fr, tokenizer_en, model_en, device)
        augmented_data.extend(augmented_entries)

    # Convert to DataFrame
    augmented_df = pd.DataFrame(augmented_data)

    # Combine original and augmented data
    combined_df = pd.concat([input_df, augmented_df], ignore_index=True)

    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save augmented data to a file
    try:
        combined_df.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"Data augmentation completed. Augmented data saved to '{output_file}'.")
    except Exception as e:
        print(f"Failed to save augmented data: {e}")

def main():
    set_seed()
    device = initialize_device()
    data = load_data('../training_input.csv')


    aug_syn = naw.SynonymAug(aug_p=0.3)
    aug_delete = naw.RandomWordAug(action="delete", aug_p=0.3)
    augmenters = [aug_syn, aug_delete]

    # Load translation models
    tokenizer_fr, model_fr = load_translation_model('en', 'fr', device)
    tokenizer_en, model_en = load_translation_model('fr', 'en', device)

    # Perform data augmentation
    augment_data(
        input_df=data,
        augmenters=augmenters,
        tokenizer_fr=tokenizer_fr,
        model_fr=model_fr,
        tokenizer_en=tokenizer_en,
        model_en=model_en,
        device=device,
        output_file='../question_augmented_format.csv'  # Update to your desired format
    )

if __name__ == "__main__":
    main()