# Importing necessary classes
from transformers import SeamlessM4Tv2Model, AutoProcessor
import torch
import asyncio
import json
import random
import soundfile as sf
from fetch_news import scrape_news

print("Telugu News Translation \n \n \n")


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

scraped_news = asyncio.run(scrape_news())

#open news_india.json

# Read the news data
with open('news_india.json', 'r') as file:
    news_data = json.load(file)

random_heading = random.choice(json.loads(news_data))["heading"]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

# Define the English text
english_text = random_heading

# Process the input text (specifying the source language as English)
text_inputs = processor(text=english_text, src_lang="eng", return_tensors="pt").to(device)

# Generate the translated speech (target language set to Telugu)
audio_array = model.generate(**text_inputs, tgt_lang="tel")[0].cpu().numpy().squeeze()


sf.write('input_audio.wav', audio_array, processor.feature_extractor.sampling_rate)

