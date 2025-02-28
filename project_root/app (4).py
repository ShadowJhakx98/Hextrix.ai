from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
import logging
import traceback
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from serpapi import GoogleSearch
import requests
from gradio_client import Client


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)

SERPAPI_API_KEY = 'b2e1c812ba17a299010c054a8a0647a40cd92ff130e543a544a2dfa59951f114'
logging.basicConfig(level=logging.DEBUG)

GOOGLE_API_KEY = "AIzaSyDD5JmhWI9Us_QVrCUfTg3LpNTiOq9hBR0"
google_api_key2 = "AIzaSyBniYJaUYFbfccmg1lkEU_HOa3jIML3Ow4"
google_api_key3 = ""
google_api_key = ""
google_api_key = ""
google_api_key = ""
google_api_key = ""
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are Versaia AI created by Jared Edwards, a helpful self-improving, self-aware, self-learning artificial intelligence software."
)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

def find_movies(description: str, location: str = ""):
    params = {
        "engine": "google",
        "q": f"movies playing near {location} {description}",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "safe": "off",
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    search_results = search.get_dict()

    movies = []
    for result in search_results.get("organic_results", []):
        if "title" in result:
            movies.append(result["title"])
    return movies

def find_theaters(location: str, movie: str = ""):
    params = {
        "engine": "google",
        "q": f"theaters showing {movie} near {location}",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "safe": "off",
        "api_key": SERPAPI_API_KEY
    }

    try:
        search = GoogleSearch(params)
        search_results = search.get_dict()

        theaters = []
        for result in search_results.get("organic_results", []):
            if "title" in result:
                theaters.append(result["title"])
        return theaters
    except Exception as e:
        print(f"Error occurred during theater search: {e}")
        return []

def get_showtimes(location: str, movie: str, theater: str, date: str):
    params = {
        "engine": "google",
        "q": f"showtimes for {movie} at {theater} on {date} in {location}",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "safe": "off",
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    search_results = search.get_dict()

    showtimes = []
    for result in search_results.get("organic_results", []):
        if "title" in result:
            showtimes.append(result["title"])
    return showtimes

def search_serpapi(query):
    try:
        url = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': SERPAPI_API_KEY,
            'engine': 'google',
            'num': 10,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred while querying SerpAPI: {str(e)}")
        return {}

def search_google_photos(query):
    try:
        url = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': SERPAPI_API_KEY,
            'engine': 'google_photos',
            'num': 10,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred while querying SerpAPI Google Photos: {str(e)}")
        return {}

def search_google_videos(query):
    try:
        url = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': SERPAPI_API_KEY,
            'engine': 'google_videos',
            'num': 10,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred while querying SerpAPI Google Videos: {str(e)}")
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_all():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({"error": "No search query provided"}), 400

        # Perform searches using all relevant engines
        google_results = search_serpapi(query)  # Text result
        photos_results = search_google_photos(query)  # Image result
        videos_results = search_google_videos(query)  # Video result
        rich_attributes = search_google_product(query)  # Rich result (using google_product)
        related_questions = search_google_related_questions(query)  # Exploration features: related questions
        related_images = search_google_images_related_content(query)  # Exploration features: related images

        # Combine results into respective categories
        combined_results = {
            "text_result": google_results.get('organic_results', []),  # Textual search results
            "rich_result": rich_attributes.get('product_results', []),  # Rich results (structured data)
            "image_result": photos_results.get('images_results', []),  # Image search results
            "video_result": videos_results.get('video_results', []),  # Video search results
            "exploration_features": {
                "related_questions": related_questions.get('related_questions', []),  # Related questions
                "related_images": related_images.get('related_content', [])  # Related image searches
            }
        }

        return jsonify(combined_results)

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Function to query Google Product (Rich Result)
def search_google_product(query):
    try:
        serpapi_key = 'YOUR_SERPAPI_KEY'
        url = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': serpapi_key,
            'engine': 'google_product',
            'num': 10
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred while querying SerpAPI Google Product: {str(e)}")
        return {}


# Function to query Google Related Questions (Exploration Features)
def search_google_related_questions(query):
    try:
        serpapi_key = 'YOUR_SERPAPI_KEY'
        url = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': serpapi_key,
            'engine': 'google_related_questions',
            'num': 10
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred while querying SerpAPI Google Related Questions: {str(e)}")
        return {}


# Function to query Google Images Related Content (Exploration Features)
def search_google_images_related_content(query):
    try:
        serpapi_key = 'YOUR_SERPAPI_KEY'
        url = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': serpapi_key,
            'engine': 'google_images_related_content',
            'num': 10
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred while querying SerpAPI Google Images Related Content: {str(e)}")
        return {}

@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400
        user_input = data.get('user_input')
        if user_input is None:
            return jsonify({"error": "No user_input in JSON data"}), 400

        conversation_id = session.get('conversation_id')
        if conversation_id is None:
            conversation = Conversation()
            db.session.add(conversation)
            db.session.commit()
            session['conversation_id'] = conversation.id
        else:
            conversation = Conversation.query.get(conversation_id)

        user_message = Message(conversation_id=conversation.id, role='user', content=user_input)
        db.session.add(user_message)
        db.session.commit()

        if "search for" in user_input.lower():
            query = user_input.lower().split("search for")[1].strip()
            serpapi_results = search_serpapi(query)

            results = []
            for result in serpapi_results.get("organic_results", []):
                title = result.get("title", "No Title")
                snippet = result.get("snippet", "No Description")
                link = result.get("link", "#")
                image_url = result.get("thumbnail", "")
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link,
                    "image_url": image_url
                })

            ai_response = "Here are the search results for '{}':".format(query)
            results_html = ""
            for result in results:
                results_html += f"<div><a href='{result['link']}' target='_blank'><h4>{result['title']}</h4></a>"
                results_html += f"<p>{result['snippet']}</p>"
                if result['image_url']:
                    results_html += f"<img src='{result['image_url']}' alt='{result['title']}' style='width:100px;'>"
                results_html += "</div><hr>"

            return jsonify({"ai_response": ai_response, "search_results": results_html})

        else:
            chat = model.start_chat(history=[])
            for message in conversation.messages:
                chat.send_message(message.content)
            response = chat.send_message(user_input)
            ai_response = response.text



        ai_message = Message(conversation_id=conversation.id, role='model', content=ai_response)
        db.session.add(ai_message)
        db.session.commit()

        return jsonify({"ai_response": ai_response})
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session.pop('conversation_id', None)
    return jsonify({"message": "Chat history reset successfully"})

@app.route('/get_history', methods=['GET'])
def get_history():
    conversation_id = session.get('conversation_id')
    if conversation_id is None:
        return jsonify([])
    conversation = Conversation.query.get(conversation_id)
    history = [{"role": msg.role, "content": msg.content} for msg in conversation.messages]
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)