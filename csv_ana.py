from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as ai
import pandas as pd
import random
import json
import os
import time

# Flask app initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize sentiment analysis models
analyzer_vader = SentimentIntensityAnalyzer()

# Configure Google Gemini API
API_KEY = 'AIzaSyCIgHspeOtytvBf0_ohZZdt43DUqNJBf2Q'
ai.configure(api_key=API_KEY)
model = ai.GenerativeModel("gemini-pro")
chat = model.start_chat()

# JSON file for history
HISTORY_FILE = 'history.json'

# Sentiment mapping for specific words
sentiment_mapping = {
    'nallathu': 'POSITIVE',
    'nalla': 'POSITIVE',
    'nanmai': 'POSITIVE',
    'could be improved': 'NEUTRAL',
    'not too bad': 'NEUTRAL',
    'shit': 'BAD WORD',
    'mairu': 'BAD WORD',
}

def read_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            return json.load(file)
    return []

def write_history(history):
    with open(HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)

# Route: Home Page
@app.route("/")
def home():
    return render_template("score.html")

# Route: CSV Sentiment Analysis
@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    print("Received a request for CSV analysis")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    text_column = request.form.get("text_column", "text")
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        print(f"CSV file loaded with {len(df)} entries")
        
        if text_column not in df.columns:
            return jsonify({"error": f"Column '{text_column}' not found in CSV"})

        sentiments = []
        start_time = time.time()
        for text in df[text_column]:
            try:
                sentiment, score, response, ai_response = perform_sentiment_analysis(text)
                sentiments.append((text, sentiment, response, ai_response))
                if len(sentiments) % 100 == 0:
                    print(f"Processed {len(sentiments)} entries...")
            except Exception as e:
                print(f"Error processing text '{text}': {e}")

        overall_sentiment = determine_overall_sentiment([s[1] for s in sentiments])
        overall_score = sum([score for _, _, score, _ in sentiments]) / len(sentiments) * 100
        print(f"Overall sentiment: {overall_sentiment}, Overall score: {overall_score}")

        history = read_history()
        for entry in sentiments:
            history_entry = {
                "text": entry[0],
                "sentiment": entry[1],
                "response": entry[2],
                "ai_response": entry[3]
            }
            history.append(history_entry)
        write_history(history)
        
        end_time = time.time()
        print(f"CSV analysis completed in {end_time - start_time} seconds")
        return jsonify({
            "overall_sentiment": overall_sentiment,
            "overall_score": overall_score,
            "ai_response": generate_dynamic_response(overall_sentiment, overall_score)
        })
    return jsonify({"error": "Invalid file format"})

# Route: View History
@app.route("/history", methods=["GET"])
def view_history():
    history = read_history()
    return jsonify(history)

# Route: Delete History
@app.route("/delete", methods=["POST"])
def delete_history():
    entry_id = request.json.get("id")
    history = read_history()
    history = [entry for i, entry in enumerate(history) if i != entry_id]
    write_history(history)
    return jsonify({"status": "success"})

# Helper Functions
def perform_sentiment_analysis(text):
    sentiment = sentiment_mapping.get(text, "NEUTRAL")  # Use predefined mappings
    score = 0.5

    # Use TextBlob for sentiment analysis if not found in predefined mappings
    if sentiment == "NEUTRAL":
        blob = TextBlob(text)
        polarity_tb = blob.sentiment.polarity

        # Use VADER for sentiment analysis
        sentiment_dict = analyzer_vader.polarity_scores(text)
        polarity_vader = sentiment_dict['compound']

        # Ensemble decision: Weighted Averaging
        final_polarity = (polarity_tb + polarity_vader) / 2

        if final_polarity > 0:
            sentiment = "POSITIVE"
        elif final_polarity < 0:
            sentiment = "NEGATIVE"
    
    response = generate_dynamic_response(sentiment, score)

    # Generate Gemini chat response
    chat_message = f"The sentiment of this feedback is {sentiment}. Provide some solutions or suggestions for {sentiment} as business advisor by using more emojis."
    ai_response = chat_response(chat_message)
    
    return sentiment, score, response, ai_response

def determine_overall_sentiment(sentiments):
    sentiment_counts = {
        "POSITIVE": sentiments.count("POSITIVE"),
        "NEGATIVE": sentiments.count("NEGATIVE"),
        "BAD WORD": sentiments.count("BAD WORD"),
        "NEUTRAL": sentiments.count("NEUTRAL"),
    }
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return overall_sentiment

def generate_dynamic_response(sentiment, score):
    responses = {
        'POSITIVE': [
            "You're on top of the world! Keep spreading positivity! 🌞",
            "What a wonderful mood! You're glowing with happiness! ✨",
        ],
        'NEGATIVE': [
            "It seems like your text reflects some negative emotions. Stay strong.",
            "It's okay to feel down sometimes. Take a deep breath, and things will improve.",
        ],
        'BAD WORD': [
            "WARNING: Please avoid using inappropriate language. 🚫",
            "ALERT: The words you've used are not acceptable. ⚠️",
        ],
        'NEUTRAL': [
            "Things are steady, nothing to worry about. Just keep going. 😊",
            "No strong emotions today, but you're doing just fine. 🌿",
        ],
    }
    return random.choice(responses.get(sentiment, ["Neutral feedback detected."]))

def chat_response(user_message):
    try:
        response = chat.send_message(user_message)
        return response.text
    except Exception as e:
        print(f"Error while connecting to Gemini API: {e}")
        return "Error while connecting to Gemini API."

if __name__ == "__main__":
    app.run(debug=True)
