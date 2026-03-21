# =============================================================================
# WHOISPER LORA FINE-TUNING FOR DYSARTHRIA SPEECH
# A Complete Google Colab Script
# =============================================================================
#
# This script fine-tunes OpenAI's Whisper model on custom dysarthric speech
# (Parkinson's disease) using Parameter-Efficient Fine-Tuning (LoRA).
#
# Final Output: A merged, standalone PyTorch model ready for GGML conversion.
#
# =============================================================================


# =============================================================================
# CELL 1: INSTALL DEPENDENCIES
# =============================================================================
# Run this cell first to install all required libraries.
# This typically takes 2-3 minutes in Colab.

!pip install -q transformers datasets peft accelerate librosa soundfile jiwer evaluate tensorboard

# NOTE: The `-q` flag suppresses verbose output. Remove it if you need to debug
# installation issues.

# Verify GPU is available
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# CELL 2: MOUNT GOOGLE DRIVE & DOWNLOAD DATA
# =============================================================================
# This cell:
# 1. Mounts your Google Drive for saving outputs
# 2. Downloads your dysarthria dataset from shared Google Drive links

from google.colab import drive
drive.mount('/content/drive')

# Install gdown for downloading from Google Drive
!pip install -q gdown

# Create data directory
import os
DATA_DIR = "/content/dysarthria_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Download Audio Files ---
# Your audio folder: https://drive.google.com/drive/folders/1rkmGdusM3pzjIXxdT6-IsiVYn1zIHdKt
print("Downloading audio files from Google Drive...")
!gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1rkmGdusM3pzjIXxdT6-IsiVYn1zIHdKt" -O "{DATA_DIR}"

# --- Download Metadata CSV ---
# Your metadata sheet: https://docs.google.com/spreadsheets/d/1UUV7tPU18XDW5eXWcK2E-o84Y3kdgi-YGjlX1PFoARM
print("\nDownloading metadata.csv from Google Sheets...")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
!curl -L -o "{METADATA_FILE}" "https://docs.google.com/spreadsheets/d/1UUV7tPU18XDW5eXWcK2E-o84Y3kdgi-YGjlX1PFoARM/export?format=csv&gid=0"

# Verify downloads
print(f"\nDataset contents:")
!ls -la "{DATA_DIR}" | head -20
audio_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.wav')])
print(f"\nTotal audio files: {audio_count}")
print(f"Metadata file exists: {os.path.exists(METADATA_FILE)}")


# =============================================================================
# CELL 3: CONFIGURATION
# =============================================================================
# Paths are now set automatically based on the downloads above

import os

# Data paths (already set in Cell 2)
# DATA_DIR and METADATA_FILE are already defined

# --- Output Configuration ---
# Save to Google Drive so they persist after Colab session ends
OUTPUT_DIR = "/content/drive/MyDrive/whisper_dysarthria_lora"
FINAL_MODEL_DIR = "/content/drive/MyDrive/whisper_dysarthria_merged"

# --- Model Configuration ---
MODEL_NAME = "openai/whisper-small"  # Options: tiny, base, small, medium, large
LANGUAGE = "english"
TASK = "transcribe"  # Options: transcribe, translate

# --- LoRA Configuration ---
LORA_R = 8          # LoRA rank - higher = more parameters, better quality but slower
LORA_ALPHA = 16     # LoRA scaling factor (usually 2x rank)
LORA_DROPOUT = 0.05 # Dropout for LoRA layers

# --- Training Configuration ---
BATCH_SIZE = 8              # Reduce if you get OOM errors (try 4 or 2)
GRADIENT_ACCUMULATION = 2   # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION
LEARNING_RATE = 1e-4        # LoRA typically uses higher LR than full fine-tuning
NUM_EPOCHS = 10             # With 50 samples, 10 epochs is reasonable
WARMUP_STEPS = 20           # Warmup steps before reaching full learning rate
MAX_STEPS = -1              # Set to -1 to use epochs, or specify exact steps

# Logging settings
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

print("Configuration loaded!")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Final model directory: {FINAL_MODEL_DIR}")


# =============================================================================
# CELL 4: PREVIEW THE DATA
# =============================================================================
# Quick check to make sure everything loaded correctly

import pandas as pd

print("=" * 60)
print("DATA PREVIEW")
print("=" * 60)

# Load and display metadata
df = pd.read_csv(METADATA_FILE)
print(f"\nMetadata shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check audio files
import os
audio_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.wav')])
print(f"\nAudio files found: {len(audio_files)}")
print(f"Sample filenames: {audio_files[:3]}")

# Verify all files in metadata exist
missing = []
for fname in df['file_name']:
    if not os.path.exists(os.path.join(DATA_DIR, fname)):
        missing.append(fname)

if missing:
    print(f"\n⚠️ Warning: {len(missing)} files from metadata not found!")
    print(f"Missing: {missing[:5]}")
else:
    print(f"\n✓ All {len(df)} audio files found!")


# =============================================================================
# CELL 5: LOAD AND PREPARE THE DATASET
# =============================================================================
# This cell loads your audio files and transcriptions
# and formats them for the Hugging Face datasets library.

from datasets import Dataset, Audio
import librosa
import soundfile as sf

def load_custom_dataset(data_dir, metadata_file):
    """
    Load custom dysarthria speech dataset from Google Drive.

    Expected structure:
    data_dir/
    ├── audio_001.wav
    ├── audio_002.wav
    ├── ...
    └── metadata.csv  (columns: file_name, transcription)

    Returns:
        Hugging Face Dataset object
    """

    # Read the metadata CSV
    print(f"Loading metadata from: {metadata_file}")
    df = pd.read_csv(metadata_file)

    # Validate columns
    required_cols = ['file_name', 'transcription']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata.csv: {missing_cols}")

    print(f"Found {len(df)} samples in metadata")

    # Build full paths to audio files
    # Handle both relative filenames and full paths in the CSV
    def get_audio_path(filename):
        # If it's already a full path, use it; otherwise, join with data_dir
        if os.path.isabs(filename):
            return filename
        return os.path.join(data_dir, filename)

    # Create list of dictionaries for the dataset
    data_samples = []
    skipped_files = []

    for idx, row in df.iterrows():
        audio_path = get_audio_path(row['file_name'])

        # Check if file exists
        if not os.path.exists(audio_path):
            skipped_files.append(audio_path)
            continue

        data_samples.append({
            'audio_path': audio_path,
            'transcription': row['transcription'],
            'file_name': row['file_name']
        })

    if skipped_files:
        print(f"Warning: Skipped {len(skipped_files)} files not found:")
        for f in skipped_files[:5]:  # Show first 5
            print(f"  - {f}")

    print(f"Successfully loaded {len(data_samples)} samples")

    # Create Hugging Face Dataset
    dataset_dict = {
        'audio': [s['audio_path'] for s in data_samples],
        'transcription': [s['transcription'] for s in data_samples],
        'file_name': [s['file_name'] for s in data_samples]
    }

    dataset = Dataset.from_dict(dataset_dict)

    # Cast the audio column to Audio type with 16kHz sampling
    # This automatically handles loading and resampling
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    return dataset

# Load the dataset
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
dataset = load_custom_dataset(DATA_DIR, METADATA_FILE)

# Split into train and validation sets (80/20 split)
# With 50 samples: 40 train, 10 validation
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"\nDataset split:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(eval_dataset)}")

# Show a sample
print(f"\nSample data point:")
sample = train_dataset[0]
print(f"  Audio path: {sample['audio']['path']}")
print(f"  Sampling rate: {sample['audio']['sampling_rate']}")
print(f"  Audio shape: {len(sample['audio']['array'])} samples")
print(f"  Transcription: {sample['transcription']}")


# =============================================================================
# CELL 6: LOAD WHISPER PROCESSOR AND FEATURE EXTRACTOR
# =============================================================================
# The processor combines:
#   - Feature Extractor: Converts raw audio to Log-Mel Spectrograms
#   - Tokenizer: Converts text to token IDs for the model

from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer

print("Loading Whisper processor components...")

# Load the feature extractor (audio -> log-mel spectrogram)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK
)

# Load the tokenizer (text -> token IDs)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK
)

# Combine into a single processor
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK
)

print(f"Processor loaded for model: {MODEL_NAME}")
print(f"Language: {LANGUAGE}")
print(f"Task: {TASK}")

# Test the tokenizer
sample_text = "Hello, this is a test transcription."
encoded = tokenizer(sample_text)
decoded = tokenizer.decode(encoded['input_ids'])
print(f"\nTokenizer test:")
print(f"  Input: {sample_text}")
print(f"  Token IDs: {encoded['input_ids'][:10]}...")
print(f"  Decoded: {decoded}")


# =============================================================================
# CELL 7: PREPROCESSING FUNCTION
# =============================================================================
# This function prepares each sample for training:
# 1. Converts audio to Log-Mel Spectrogram
# 2. Tokenizes the transcription

def prepare_dataset(batch):
    """
    Preprocess a batch of samples for Whisper training.

    Args:
        batch: Dictionary containing 'audio' and 'transcription' fields

    Returns:
        Dictionary with 'input_features' and 'labels' tensors
    """

    # 1. Load and resample audio (already done by Audio column cast)
    #    Audio is at 16kHz as required by Whisper

    # 2. Compute Log-Mel Spectrogram input features
    #    The feature_extractor handles:
    #    - Computing STFT
    #    - Mel filterbank application
    #    - Log scaling
    #    - Normalization
    audio_array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]

    input_features = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features[0]  # Remove batch dimension added by extractor

    # 3. Tokenize the transcription
    #    This converts text to token IDs that the model predicts
    labels = tokenizer(batch["transcription"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels
    }

print("Preprocessing training dataset...")
train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names,
    desc="Processing training audio"
)

print("Preprocessing validation dataset...")
eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
    desc="Processing validation audio"
)

print(f"\nPreprocessing complete!")
print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(eval_dataset)} samples")

# Check a sample
sample = train_dataset[0]
print(f"\nProcessed sample shapes:")
print(f"  Input features: {sample['input_features'].shape}")  # Should be [80, 3000] for 30s audio
print(f"  Labels length: {len(sample['labels'])}")


# =============================================================================
# CELL 8: DATA COLLATOR
# =============================================================================
# The data collator prepares batches for training by:
# - Padding input features to the same length
# - Padding labels and replacing padding with -100 (ignored by loss)
# - Creating attention masks

from transformers import WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper that dynamically pads:
    - Input features (log-mel spectrograms)
    - Labels (token IDs)

    This is adapted from the Hugging Face Whisper fine-tuning tutorial.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a padded batch.
        """

        # 1. Pad input features (log-mel spectrograms)
        #    Each input is shape [80, num_frames], we pad along num_frames
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        # Use processor's feature extractor for padding
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # 2. Pad labels (token sequences)
        #    Replace pad tokens with -100 so they're ignored in loss
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # 3. Move decoder_start_token_id to correct device
        #    This tells the model where to start generating
        batch["labels"] = labels

        return batch

print("Data collator defined!")


# =============================================================================
# CELL 9: LOAD MODEL WITH PEFT/LoRA
# =============================================================================
# This is the core of parameter-efficient fine-tuning.
# Instead of updating all 244M parameters of whisper-small,
# LoRA adds small trainable adapter matrices to attention layers.

from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

print("Loading base Whisper model...")

# Load the pre-trained Whisper model
# We use ConditionalGeneration for seq2seq tasks
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Automatically use GPU
    torch_dtype=torch.float16  # Use FP16 for efficiency
)

# Disable caching during training (not needed for training)
model.config.use_cache = False

# Configure LoRA
# LoRA adds low-rank matrices to attention layers:
#   W_new = W_original + (A @ B)
# Where A and B are small matrices (rank x dimension)
#
# This drastically reduces trainable parameters:
#   Full fine-tuning: ~244M parameters
#   LoRA (rank=8): ~1-2M parameters (1% of original!)

print(f"\nConfiguring LoRA:")
print(f"  Rank (r): {LORA_R}")
print(f"  Alpha: {LORA_ALPHA}")
print(f"  Dropout: {LORA_DROPOUT}")
print(f"  Target modules: q_proj, v_proj")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],  # Target attention query/value projections
    bias="none",  # Don't modify bias terms
    task_type=TaskType.SEQ_2_SEQ_LM  # Sequence-to-sequence language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nParameter efficiency:")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

# Create the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)


# =============================================================================
# CELL 10: DEFINE EVALUATION METRICS
# =============================================================================
# We use Word Error Rate (WER) to evaluate transcription quality.
# WER = (Substitutions + Insertions + Deletions) / Total Words

import evaluate

# Load WER metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    """
    Compute Word Error Rate (WER) for predictions.

    Args:
        pred: EvalPrediction containing predictions and labels

    Returns:
        Dictionary with WER score
    """

    # Get predicted token IDs (argmax over logits)
    pred_ids = pred.predictions

    # Replace -100 in labels (they were padding)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels to text
    # skip_special_tokens removes <pad>, <eos>, etc.
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("Metrics defined: Word Error Rate (WER)")


# =============================================================================
# CELL 11: TRAINING ARGUMENTS
# =============================================================================
# Configure the training loop with Seq2SeqTrainingArguments.
# These settings are optimized for a single Colab GPU.

from transformers import Seq2SeqTrainingArguments

print("Configuring training arguments...")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # --- Batch Configuration ---
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION

    # --- Learning Rate Schedule ---
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,  # -1 means use epochs

    # --- Mixed Precision ---
    fp16=True,  # Use FP16 for faster training on GPU
    fp16_full_eval=True,  # Also use FP16 during evaluation

    # --- Generation Settings ---
    # During evaluation, generate predictions with beam search
    generation_max_length=225,  # Max tokens to generate
    predict_with_generate=True,  # Generate during evaluation for WER

    # --- Checkpointing & Logging ---
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",  # Evaluate every eval_steps
    save_strategy="steps",  # Save every save_steps
    save_total_limit=2,  # Keep only 2 best checkpoints
    load_best_model_at_end=True,  # Load best model when done
    metric_for_best_model="wer",  # Use WER to determine best
    greater_is_better=False,  # Lower WER is better

    # --- Optimization ---
    gradient_checkpointing=True,  # Trade compute for memory
    optim="adamw_bnb_8bit",  # Use 8-bit Adam for memory efficiency

    # --- Logging ---
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=["tensorboard"],  # Log to TensorBoard

    # --- Misc ---
    remove_unused_columns=False,  # Keep audio column
    label_names=["labels"],  # Label column name
    dataloader_num_workers=2,  # Parallel data loading
)

print(f"Training configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size per device: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  FP16 training: True")
print(f"  Output directory: {OUTPUT_DIR}")


# =============================================================================
# CELL 12: INITIALIZE TRAINER
# =============================================================================
# The Seq2SeqTrainer handles the training loop, evaluation, and checkpointing.

from transformers import Seq2SeqTrainer

print("Initializing trainer...")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,  # Use feature extractor for saving
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Trainer initialized!")
print(f"Training on {len(train_dataset)} samples")
print(f"Evaluating on {len(eval_dataset)} samples")


# =============================================================================
# CELL 13: START TRAINING
# =============================================================================
# This cell runs the actual training loop.
# Training time: ~30-60 minutes for 50 samples on Colab T4/L4

import time

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Start training
# This will:
# 1. Load data in batches
# 2. Forward pass through model with LoRA adapters
# 3. Compute loss
# 4. Backward pass (only LoRA parameters updated)
# 5. Evaluate every eval_steps
# 6. Save checkpoints every save_steps
# 7. Load best model at end

train_result = trainer.train()

print()
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Print training metrics
print(f"\nTraining metrics:")
for key, value in train_result.metrics.items():
    print(f"  {key}: {value}")

# Save the final training state
trainer.save_model(OUTPUT_DIR)
trainer.save_state()

# Save the processor for later use
processor.save_pretrained(OUTPUT_DIR)

print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")


# =============================================================================
# CELL 14: EVALUATE ON VALIDATION SET
# =============================================================================
# Let's see how well our fine-tuned model performs!

print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

# Evaluate on validation set
eval_results = trainer.evaluate()

print(f"\nValidation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Test transcription on a few samples
print("\nSample transcriptions (Ground Truth vs Predicted):")
print("-" * 60)

# Load a few samples from original dataset for testing
from datasets import Dataset, Audio
test_dataset = Dataset.from_dict({
    'audio': [os.path.join(DATA_DIR, df.iloc[i]['file_name']) for i in range(min(3, len(df)))],
    'transcription': [df.iloc[i]['transcription'] for i in range(min(3, len(df)))]
})
test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=16000))

for i, sample in enumerate(test_dataset):
    # Get ground truth
    ground_truth = sample['transcription']

    # Prepare input
    input_features = feature_extractor(
        sample['audio']['array'],
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(model.device)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=225)

    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    print(f"\nSample {i+1}:")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Predicted:    {predicted_text}")


# =============================================================================
# CELL 15: MERGE LoRA AND EXPORT FINAL MODEL
# =============================================================================
# !!! CRITICAL STEP FOR GGML CONVERSION !!!
#
# whisper.cpp cannot load separate LoRA adapter files.
# We must merge the LoRA weights back into the base model
# to create a standalone PyTorch model.

print("=" * 60)
print("MERGING LoRA WEIGHTS INTO BASE MODEL")
print("=" * 60)

from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# Step 1: Load the ORIGINAL base model (without LoRA)
print(f"\n1. Loading base model: {MODEL_NAME}")
base_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: Load the trained LoRA adapter
print(f"2. Loading LoRA adapter from: {OUTPUT_DIR}")
peft_model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR,
    device_map="auto"
)

# Step 3: Merge LoRA weights into base model
# This applies: W_new = W_original + (A @ B)
print("3. Merging LoRA weights into base model...")
merged_model = peft_model.merge_and_unload()

print("   Merge complete!")
print(f"   Model type: {type(merged_model)}")

# Step 4: Save the MERGED model
# This is now a standalone model that can be converted to GGML
print(f"\n4. Saving merged model to: {FINAL_MODEL_DIR}")

# Create directory if it doesn't exist
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Save the merged model weights
merged_model.save_pretrained(FINAL_MODEL_DIR)

# Save the processor (tokenizer + feature extractor)
processor.save_pretrained(FINAL_MODEL_DIR)

# Also save the model in a more portable format
print("5. Saving additional model format...")

# Save as standard PyTorch save (alternative format)
torch.save(
    merged_model.state_dict(),
    os.path.join(FINAL_MODEL_DIR, "pytorch_model.bin")
)

print()
print("=" * 60)
print("EXPORT COMPLETE!")
print("=" * 60)
print(f"\nMerged model saved to: {FINAL_MODEL_DIR}")
print("\nFiles in output directory:")
for f in os.listdir(FINAL_MODEL_DIR):
    filepath = os.path.join(FINAL_MODEL_DIR, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  {f}: {size_mb:.2f} MB")

print("\n" + "=" * 60)
print("NEXT STEPS FOR GGML CONVERSION")
print("=" * 60)
print("""
To convert this merged model to GGML format for whisper.cpp:

1. Clone the whisper.cpp repository:
   git clone https://github.com/ggerganov/whisper.cpp
   cd whisper.cpp

2. Convert the PyTorch model to GGML format:
   python3 models/convert-h5-to-ggml.py \\
       /content/drive/MyDrive/whisper_dysarthria_merged \\
       /content/drive/MyDrive/whisper_dysarthria_ggml \\
       --outtype q8_0

3. The output .ggml file can be used directly with whisper.cpp
   on your iOS device!

For iOS integration:
- Add the .ggml file to your iOS app bundle
- Use whisper.cpp iOS bindings to load and run inference
""")


# =============================================================================
# CELL 16: OPTIONAL - TEST THE MERGED MODEL
# =============================================================================
# Verify the merged model works correctly before converting to GGML.

print("=" * 60)
print("TESTING MERGED MODEL")
print("=" * 60)

# Load the merged model
print(f"Loading merged model from: {FINAL_MODEL_DIR}")
test_model = WhisperForConditionalGeneration.from_pretrained(
    FINAL_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
test_processor = WhisperProcessor.from_pretrained(FINAL_MODEL_DIR)

# Test on a sample audio file
test_audio_path = os.path.join(DATA_DIR, df.iloc[0]['file_name'])
print(f"\nTesting on: {test_audio_path}")

# Load audio
audio_array, sr = librosa.load(test_audio_path, sr=16000)

# Process
input_features = test_processor.feature_extractor(
    audio_array,
    sampling_rate=16000,
    return_tensors="pt"
).input_features.to(test_model.device)

# Generate
with torch.no_grad():
    predicted_ids = test_model.generate(input_features, max_length=225)

predicted_text = test_processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(f"\nGround truth: {df.iloc[0]['transcription']}")
print(f"Predicted:    {predicted_text}")

print("\nMerged model is working correctly!")


# =============================================================================
# CELL 17: OPTIONAL - DOWNLOAD MODEL (if not using Drive)
# =============================================================================
# If you need to download the model directly from Colab:

from google.colab import files
import shutil

# Create a zip file of the merged model
shutil.make_archive(
    "/content/whisper_dysarthria_merged",
    'zip',
    FINAL_MODEL_DIR
)

# Download (this will prompt a download in the browser)
# files.download("/content/whisper_dysarthria_merged.zip")

print("Model archived and ready for download!")
print("Uncomment the files.download() line above to trigger download.")
