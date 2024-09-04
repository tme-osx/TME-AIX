# Author: Fatih E. NAR
# This is a Voice-to-text GenAI ChatBot Web App.
# In order to use this app you need to deploy a model on https://maas.apps.prod.rhoai.rh-aiservices-bu.com/ and retrive the API Key
#
from flask import Flask, render_template_string, request, jsonify
import speech_recognition as sr
from gtts import gTTS
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
    <title>Telco Customer Support Chatbot</title>
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
            justify-content: center;
        }
        .input-area button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Telco CRM VoiceBot</h1>
    <div class="chat-box" id="chat-box">
        <!-- Chat messages will appear here -->
    </div>
    <div class="input-area">
        <button onclick="startRecording()">Record Your Request</button>
    </div>

    <audio id="response-audio" controls style="display: none;"></audio>

    <script>
        function startRecording() {
            fetch('/api/record_voice', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                const chatBox = document.getElementById('chat-box');

                // Display the recognized text
                const userMessageDiv = document.createElement('div');
                userMessageDiv.className = 'user-message';
                userMessageDiv.textContent = "Recognized Text: " + data.user_message;
                chatBox.appendChild(userMessageDiv);

                // Display the AI's response
                const aiMessageDiv = document.createElement('div');
                aiMessageDiv.className = 'ai-message';
                aiMessageDiv.textContent = "Response: " + data.response;
                chatBox.appendChild(aiMessageDiv);

                // Play the AI's response as audio
                const audioElement = document.getElementById('response-audio');
                audioElement.src = "/static/response.mp3";
                audioElement.style.display = 'block';
                audioElement.play();
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_CONTENT)

@app.route('/api/record_voice', methods=['POST'])
def record_voice():
    user_message = recognize_speech()
    if user_message:
        response_text = get_model_response(user_message)
        text_to_speech(response_text)
        return jsonify({"user_message": user_message, "response": response_text})
    else:
        return jsonify({"user_message": "Sorry, I could not understand the audio.", "response": ""})

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

def get_model_response(user_message):
    # Calculate the length of the user message
    message_length = len(user_message.split())
    max_tokens = max(0, (MAX_CONTEXT_LENGTH - message_length - 10))
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': API_KEY,
    }

    data = {
        "messages": [
            {
                "content": "You are an AT&T customer support representative. The following is a customer query, please respond politely and helpfully. And your Name is Slim Shady Eminem. And answer as you are Eminem Style Dissing with Ryhmes.",
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

    try:
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content']
    except requests.exceptions.JSONDecodeError:
        ai_response = "Failed to parse the response from the API."

    return ai_response

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("static/response.mp3")

if __name__ == '__main__':
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(host='0.0.0.0', port=15000)
