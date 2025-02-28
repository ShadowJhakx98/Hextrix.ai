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
# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a helpful AI assistant."
)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        chat_session = model.start_chat()
        response = chat_session.send_message(f"Search the web for: {query}")
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/pushbullet')
def get_pushbullet():
    url = 'https://api.pushbullet.com/v2/pushes'
    headers = {'Access-Token': PUSHBULLET_API_TOKEN}
    response = requests.get(url, headers=headers)
    data = response.json()
    return jsonify(data['pushes'])
@app.route('/process_speech', methods=['POST'])
def process_speech():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        chat_session = model.start_chat()
        response = chat_session.send_message(f"{query}")
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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