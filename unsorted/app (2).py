from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy
import spacy

nlp = spacy.load('en_core_web_sm')


app = Flask(__name__)
# Configure CORS
# Replace with your actual frontend URL
CORS(app, resources={r"/process": {"origins": "https://shadowjhakx98.github.io/Versaia/livemode.html"}})


# Configure API keys and endpoints
OPENWEATHER_API_KEY = '03c0e16b1429800ac660ddde54706c28'
PUSHBULLET_API_TOKEN = 'o.YR5tFQ9hjRfVV91s9RsYY1qXKZqDcOo6'
GOOGLE_API_KEY = "AIzaSyBniYJaUYFbfccmg1lkEU_HOa3jIML3Ow4"

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)



@app.route('/process', methods=['POST'])
def process_input():
    data = request.get_json()
    user_input = data['user_input']

    # Send the user input to the webhook
    response = requests.post(
        'https://webhook.botpress.cloud/f3ec0507-ee4f-4113-a775-7cee1110ee2e',
        json={"text": user_input}
    )

    if response.status_code == 200:
        ai_response = response.json().get('text', 'No response from AI.')
    else:
        ai_response = 'Error communicating with AI.'

    # Optionally save conversation history here
    # save_conversation(user_input, ai_response)

    return jsonify({"ai_response": ai_response})

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://shadowjhakx98.github.io/Versaia/livemode.html')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/smarthub')
def smarthub():
    return render_template('smarthub.html')

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    data = request.get_json()
    task = data.get('task', '')
    if not task:
        return jsonify({'error': 'No task provided'}), 400
    suggestions = get_ai_suggestions(task)
    return jsonify({'suggestions': suggestions})

@app.route('/weather')
def get_weather():
    city = request.args.get('city', 'Saluda,SC,US')
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=imperial'
    response = requests.get(url)
    data = response.json()
    return jsonify({
        'description': data['weather'][0]['description'],
        'temperature': data['main']['temp']
    })



@app.route('/pushbullet')
def get_pushbullet():
    url = 'https://api.pushbullet.com/v2/pushes'
    headers = {'Access-Token': PUSHBULLET_API_TOKEN}
    response = requests.get(url, headers=headers)
    data = response.json()
    return jsonify(data['pushes'])




def get_ai_suggestions(task):
    palm_suggestion = get_palm_suggestion(task)
    gemini_suggestion = get_gemini_suggestion(task)
    return [palm_suggestion, gemini_suggestion]

def get_palm_suggestion(task):
    model = genai.GenerativeModel('text-bison-001')
    response = model.generate_content(f"Suggest a way to accomplish this task: {task}")
    return response.text

def get_gemini_suggestion(task):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(f"Suggest a way to accomplish this task: {task}")
    return response.text


if __name__ == '__main__':
    app.run(port=9999)