from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Configure API keys and endpoints
OPENWEATHER_API_KEY = '03c0e16b1429800ac660ddde54706c28'
PUSHBULLET_API_TOKEN = 'o.YR5tFQ9hjRfVV91s9RsYY1qXKZqDcOo6'
GOOGLE_API_KEY = "AIzaSyBniYJaUYFbfccmg1lkEU_HOa3jIML3Ow4"

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

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
@app.route('/process_speech', methods=['POST'])
def process_speech(speech_text):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"Process this speech input and provide a helpful response: '{speech_text}'"
    response = model.generate_content(prompt)
    return response.text

@app.route('/process_speech', methods=['POST'])
def process_speech_route():
    data = request.get_json()
    speech = data.get('speech')

    if not speech:
        return jsonify({'error': 'No speech input provided'}), 400

    # Process the speech text and get the AI's response
    ai_response = process_speech(speech)

    return jsonify({'response': ai_response})
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