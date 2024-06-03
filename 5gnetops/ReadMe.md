# 5G RAN Fault Prediction with AI Assistance

This project contains a machine learning model for predicting fault occurrence rates in a 5G radio network based on various network KPIs. The model is fine-tuned on a dataset containing network KPIs and fault occurrence rates. The server is built using Flask to provide predictions through a REST API.

## Project Structure

- `5gran-predictions.ipynb`: Jupyter notebook used for training the model.
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/5gdatasetsnapshot.png)<br>
- `models/5g_oss_model/`: Directory where the trained model and tokenizer are saved.
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/trainingresults200K.png)<br>
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/evalresults200K.png)<br>
- `5gran-modelserver.py`: Flask application to serve the model predictions.
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/testresults.png)<br>

