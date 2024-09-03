# Telco Customer VoiceBot

The app interacts with a specified API endpoint to process user inputs and display responses in a chat-like interface. <br>
In order to utilize this app please get your access api key from https://maas.apps.prod.rhoai.rh-aiservices-bu.com/ <br> <br>

**⚠️Recipe for Model As a Service with RHOAI, 3Scale and SSO use: [Link](https://github.com/rh-aiservices-bu/models-aas)** <br>

## Prerequisites
- An API key for the Desired Hosted GenAI Model from Model as a Server Backend

## Open your web browser and navigate to:

```bash
http://localhost:15000
```

<div align="center">
    <img src="https://github.com/tme-osx/TME-AIX/blob/main/crm/maas-vb4.png"/>
</div>

## OCP Deployment
- Build the container image (see Dockerfile here) and push to your image repo <br>
- Edit Deployment.yaml (included here) to have proper image urls and API_Key inside -> just simply;
  
```
oc deploy -f Deployment.yaml
```
