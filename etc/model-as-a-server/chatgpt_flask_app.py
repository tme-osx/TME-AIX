# Author: Fatih E. NAR
# Sample Web App for Testing Model as a Server
# Get your API key from https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
#
from flask import Flask, render_template_string, request, jsonify
import requests
import os

app = Flask(__name__)

# Read the API key from the environment variable
API_URL = 'https://mistral-7b-instruct-v0-3-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1/chat/completions'
API_KEY = os.getenv('API_KEY')
MAX_CONTEXT_LENGTH = 6000

# HTML content as a string
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatGPT UI Clone</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .chat-box {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .user-message, .ai-message {
            margin-bottom: 15px;
        }
        .user-message {
            font-weight: bold;
        }
        .ai-message {
            color: #555;
        }
        .input-area {
            display: flex;
        }
        .input-area input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .input-area button {
            padding: 10px 20px;
            margin-left: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>ChatGPT UI Clone</h1>
    <div class="chat-box" id="chat-box">
        <!-- Chat messages will appear here -->
    </div>
    <div class="input-area">
        <input type="text" id="user-input" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        function sendMessage() {
            const userInput = document.getElementById('user-input').value;
            const chatBox = document.getElementById('chat-box');

            if (!userInput.trim()) return;

            const userMessageDiv = document.createElement('div');
            userMessageDiv.className = 'user-message';
            userMessageDiv.textContent = userInput;
            chatBox.appendChild(userMessageDiv);

            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userInput })
            })
            .then(response => response.json())
            .then(data => {
                const aiMessageDiv = document.createElement('div');
                aiMessageDiv.className = 'ai-message';
                aiMessageDiv.textContent = data.response;
                chatBox.appendChild(aiMessageDiv);
                document.getElementById('user-input').value = '';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_CONTENT)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    # Calculate the length of the user message
    message_length = len(user_message.split())

    # Calculate the max_tokens based on the remaining context length
    max_tokens = max(0, (MAX_CONTEXT_LENGTH - message_length - 10))
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': API_KEY,
    }

    data = {
        "messages": [
            {
                "content": "This is a system message",
                "role": "system",
                "name": "system"
            },
            {
                "content": user_message,
                "role": "user",
                "name": "user"
            }
        ],
        "model": "mistral-7b-instruct",
        "max_tokens": max_tokens, 
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "stop": None,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "response_format": {
            "type": "text"
        }
    }

    response = requests.post(API_URL, headers=headers, json=data)

    # Print the raw response for debugging
    print("Raw response:", response.text)
    print("Status code:", response.status_code)

    try:
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content']
    except requests.exceptions.JSONDecodeError:
        return jsonify({"error": "Failed to parse the response from the API."}), 500

    return jsonify({"response": ai_response})

if __name__ == '__main__':
    # Bind to all available interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
