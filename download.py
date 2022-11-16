# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# load openai whisper model

import whisper

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    whisper.load_model("large")

if __name__ == "__main__":
    download_model()