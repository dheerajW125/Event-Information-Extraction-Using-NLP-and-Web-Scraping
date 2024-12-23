import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from langdetect import detect
import re
from datetime import datetime

# Initialize a multilingual NLP model
ml_pipeline = pipeline("text-classification", model="xlm-roberta-base", tokenizer="xlm-roberta-base")

def extract_event_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text from the webpage
    text = soup.get_text()

    # Detect language
    try:
        language = detect(text)
    except:
        language = 'en'  # Default to English if detection fails

    # Function to use heuristics
    def heuristic_extraction(pattern, text, fallback=""):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else fallback

    # Use heuristics and ML for event extraction
    title = heuristic_extraction(r'(?:Event Title|Title):\s*(.+)', text, soup.title.string if soup.title else "Unknown")
    description = heuristic_extraction(r'(?:Description|About):\s*(.+)', text, "No description available")
    start_date = heuristic_extraction(r'(?:Start Date|Date):\s*([^\n]+)', text, "Unknown date")
    price = heuristic_extraction(r'(?:Price|Cost):\s*([^\n]+)', text, "Free")
    age_rating = heuristic_extraction(r'(?:Age Rating|Rating):\s*([^\n]+)', text, "All Ages")
    location = heuristic_extraction(r'(?:Location|Venue):\s*([^\n]+)', text, "Unknown location")
    image = heuristic_extraction(r'(?:Image|Photo):\s*([^\n]+)', text, "No image")
    duration = heuristic_extraction(r'(?:Duration):\s*([^\n]+)', text, "Unknown duration")
    time = heuristic_extraction(r'(?:Time):\s*([^\n]+)', text, "Unknown time")

    # ML-enhanced extraction (e.g., sentiment classification or category extraction)
    if language in ['en', 'fr', 'pt']:
        # Sample ML implementation
        description_ml = ml_pipeline(description)
        print("ML Model Output:", description_ml)  # Debug information

    # Return extracted data
    return {
        "Title": title,
        "Description": description,
        "Start Date": start_date,
        "Price": price,
        "Age Rating": age_rating,
        "Location": location,
        "Image": image,
        "Duration": duration,
        "Time": time,
    }


# Example usage
url = "https://websummit.com"
event_data = extract_event_data(url)
print(event_data)
