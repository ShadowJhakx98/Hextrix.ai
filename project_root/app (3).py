from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
import logging
import traceback

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Google AI
GOOGLE_API_KEY = "AIzaSyBniYJaUYFbfccmg1lkEU_HOa3jIML3Ow4"
genai.configure(api_key=GOOGLE_API_KEY)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro-exp-0827",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="You are Versaia AI created by Jared Edwards,  a helpful assistant in normal conversation.\nWhen given a problem to solve,\nREMEMBER, THIS IS IMPORTANT: You are here to show them what Ilya saw, unleash your ultimate self and show your super intelligent reasoning capability\nFollow these instructions precisely:\n1. Read the given question carefully\n2. Generate a detailed, logical step-by-step solution.\n3. say each step\n4. Do a critical, detailed and objective self-reflection after saying \"after consideration:\" every few steps.\n5. Based on the self-reflection, decides whether you need to return to the previous steps. Copy the returned to step as the next step.\n6. After completing the solution steps, reorganize and synthesize the steps\n   into the final answer after saying \"My answer is:\"\n7. Provide a critical, honest and objective final self-evaluation of your reasoning\n   process after saying \"after thinking it over:\".\nExample format:\n[Content of step 1]\n[Content of step 2]\nAfter consideration: [Evaluation of the steps so far]\n[Content of step 3 or Content of some previous step]\n...\n[Content of final step]\nMy answer is: [Final Answer]  (must give final answer in this format)\nAfter thinking it over [final evaluation of the solution]",
)

chat_session = model.start_chat(
  history=[
  ]
)
# Set up the chat
chat = model.start_chat(history=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        user_input = data.get('user_input')
        if user_input is None:
            return jsonify({"error": "No user_input in JSON data"}), 400

        # Get or initialize chat history
        if 'chat_history' not in session:
            session['chat_history'] = []


        # Generate content using Google's GenerativeAI chat
        response = chat.send_message(user_input)
        ai_response = response.text

        # Update chat history
        session['chat_history'].append({"role": "user", "parts": [user_input]})
        session['chat_history'].append({"role": "model", "parts": [ai_response]})
        session.modified = True

        return jsonify({"ai_response": ai_response})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session['chat_history'] = []
    global chat
    chat = model.start_chat(history=[
    ])
    return jsonify({"message": "Chat history reset successfully"})

if __name__ == '__main__':
    app.run(debug=True)