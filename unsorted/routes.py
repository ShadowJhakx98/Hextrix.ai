from flask import Blueprint, request, jsonify
import requests

main = Blueprint('main', __name__)

@main.route('/process', methods=['POST'])
def process_input():
    data = request.get_json()
    user_input = data.get('user_input', '')

    response = requests.post(
        'https://webhook.botpress.cloud/f3ec0507-ee4f-4113-a775-7cee1110ee2e',
        json={"text": user_input}
    )

    if response.status_code == 200:
        ai_response = response.json().get('text', 'No response from AI.')
    else:
        ai_response = 'Error communicating with AI.'

    return jsonify({"ai_response": ai_response})
