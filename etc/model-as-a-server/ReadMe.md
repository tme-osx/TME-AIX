# ChatGPT UI Clone Flask App

This repository contains a simple Flask application that mimics the ChatGPT UI. The app interacts with a specified API endpoint to process user inputs and display responses in a chat-like interface. In order to utilize this app please get your access api key from https://maas.apps.prod.rhoai.rh-aiservices-bu.com/ 

## Prerequisites

- Docker installed on your machine
- An API key for the Mistral model

## How to Build and Run the Application

### 1. Clone the Repository & Build -> Run

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/chatgpt-flask-app.git
cd chatgpt-flask-app

docker build -t chatgpt-flask-app:ubi8 .

docker run -d -p 5000:5000 --name chatgpt-flask-app -e API_KEY=your_api_key chatgpt-flask-app:ubi8
```

### 2. Open your web browser and navigate to:

```bash
http://localhost:5000
```

<div align="center">
    <img src="https://raw.githubusercontent.com/fenar/TME-AIX/main/etc/model-as-a-server/maas.png"/>
</div>
