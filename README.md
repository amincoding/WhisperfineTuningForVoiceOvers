## Fine-Tuning Whisper Model on Your Voice for Speech Generation
Prerequisites
1. **Data Preparation:** A dataset of your voice, preferably clean and well-segmented audio clips paired with corresponding transcripts.
2. **Python and Libraries:** Install Python and the necessary libraries (transformers, datasets, torchaudio).
3. **GPU Access:** Required for efficient fine-tuning of models like Whisper.
## Step 1: Data Preparation
Organize your audio and transcript files in a structured directory.
``` plantext
/data
    /audio
        file1.wav
        file2.wav
        ...
    /transcripts
        file1.txt
        file2.txt
        ...
```
## Step 2: Install Required Libraries
``` bash
pip install torch transformers datasets torchaudio librosa
```
## Step 3: Load and Preprocess Data
``` python
import os
import torchaudio
from datasets import load_dataset, DatasetDict

# Function to load audio files and their transcripts
def load_data(audio_dir, transcript_dir):
    data = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(audio_dir, filename)
            transcript_path = os.path.join(transcript_dir, filename.replace(".wav", ".txt"))
            
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()
            
            data.append({"audio": audio_path, "text": transcript})
    return data

# Load data
audio_dir = '/data/audio'
transcript_dir = '/data/transcripts'
data = load_data(audio_dir, transcript_dir)

# Create a DatasetDict
dataset = DatasetDict({"train": load_dataset("csv", data_files=data)})
```
## Step 4: Fine-Tuning Whisper Model
### Prepare the Dataset
``` python
from transformers import Wav2Vec2FeatureExtractor, WhisperForConditionalGeneration

# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("openai/whisper-base")

# Function to preprocess audio
def preprocess_function(examples):
    audio = examples["audio"]
    audio, _ = torchaudio.load(audio)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return {"input_values": inputs.input_values.squeeze(), "labels": examples["text"]}

# Apply preprocessing
dataset = dataset.map(preprocess_function, remove_columns=["audio"])
```
### Load the Whisper Model
``` python
from transformers import WhisperForConditionalGeneration, Trainer, TrainingArguments

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
```
### Training Arguments
``` python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)
```
### Create Trainer
``` python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=feature_extractor,
)
```
### Train the Model
``` python
trainer.train()
```
### Save the Model
``` python 
model.save_pretrained("./fine_tuned_whisper")
feature_extractor.save_pretrained("./fine_tuned_whisper")

```
## Step 5: Generate Speech Using the Fine-Tuned Model
### Using the Model to Generate Speech
``` python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

# Load the fine-tuned model and processor
processor = Wav2Vec2Processor.from_pretrained("./fine_tuned_whisper")
model = Wav2Vec2ForCTC.from_pretrained("./fine_tuned_whisper")

# Function to generate speech
def generate_speech(text, processor, model):
    inputs = processor(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Example usage
text = "The quick brown fox jumps over the lazy dog."
generated_speech = generate_speech(text, processor, model)
print(generated_speech)
```
### Tips and Additional Resources
- Data Augmentation: Use data augmentation techniques to enhance your dataset and improve model robustness.
- Voice Cloning Tools: Explore tools like VITS for voice cloning.
- Model Compatibility: Ensure your model’s training and inference procedures are compatible with the data formats and tasks.
