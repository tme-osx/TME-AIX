# Revenue Assurance and Fraud Management (RAFM) with AI Assistance

## Project Overview

This project aims to deliver a RAFM prediction (if that particular telco transaction is fraudulent or not) with a AI model assistance with various options; (1) Balanced Random Forest, (2) a Transformer neural network and (3) fine-tuned bert. The models are trained on synthetic telecom data to predict fraud cases  and identify potential anomalies. The goal is to provide proactive revenue management and enhance revenue worklows.

## Options: 
(1) >> Revenue Assurance is a domain where traditional machine learning models, such as tree-based methods, often perform very well. Specifically, models like the Balanced Random Forest  (01-xxx worx) can be advantageous due to their inherent ability to handle class imbalance and their interpretability.<br>[ Test Accuracy: 0.9999833] <br>

(2) >> We gave a chance for a Transformer-NN (02-xxx worx) model with same dataset but it has not been a great experience, as such models needs much more data to capturing complex patterns and interactions and also requires significantly more computational power and time for training. <br> [ Test Accuracy: 0.999325] <br>

(3) >> We have fine-tuned bert with parameter efficient fine tuning using LoRA approach, (03-xxx worx) we managed to optimize based on cuda/mps/cpu use with small batch data usages, model build and served via a model server we implemented, however we are not much satisfied with the test results (see; 03-BertTest.yaml).<br> [ Test Accuracy: Pretty Prety Bad Rabbit Poop] <br>

Winner (So Far) is: [Balanced Random Forest] with the data we have for training/fine-tuning.

## Data
![Revenue Assurance Data Structure](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/rev_ass_data.png)<br>

## Results:
![Revenue Assurance Accuracy](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/rev_ass_models_accuracy.png)<br>

## [Steps for end to end MLOps setup](mlops.md)

## Steps to Run
#(1) (Obviously) Clone the Repo :-)  <br>
#(2) Open revenueassurance/01-telco-revass-ensemble.ipynb and Run-All. This would install all required libs, extract data from data/telecom_revass_data.csv.xz and load it for two models training {model1=RandomForestClassifier, model2=BalancedRandomForestClassifier }, evaluate them and save them under models/ directory. <br>
#(3) Run 01-model_server.py which loads the model2=BalancedRandomForestClassifier (better model for fraud classification) and serves on http://localhost:5000/predict url. <br>
![Model-Server](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/modelserver.png)<br>
#(4) From a cli , do a curl query: <br>
![Revenue Assurance Test Result](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/testresult.png)<br>

## Steps to Test
(A) Potential Fraud Test: <br>
```
curl -X POST -H "Content-Type: application/json" -d '{
    "Call_Duration": 300,
    "Data_Usage": 10000,
    "Sms_Count": 50,
    "Roaming_Indicator": 1,
    "MobileWallet_Use": 1,
    "Plan_Type_prepaid": 1,
    "Plan_Type_postpaid": 0,
    "Cost": 500,
    "Cellular_Location_Distance": 100,
    "Personal_Pin_Used": 0, 
    "Avg_Call_Duration": 50,
    "Avg_Data_Usage": 8000
}' http://localhost:5000/predict
```
(B) Potential Non-Fraud Test: <br>
```
curl -X POST -H "Content-Type: application/json" -d '{
    "Call_Duration": 10,
    "Data_Usage": 300,
    "Sms_Count": 5,
    "Roaming_Indicator": 0,
    "MobileWallet_Use": 1,
    "Plan_Type_prepaid": 1,
    "Plan_Type_postpaid": 0,
    "Cost": 50,
    "Cellular_Location_Distance": 3,
    "Personal_Pin_Used": 1,
    "Avg_Call_Duration": 12,
    "Avg_Data_Usage": 350
}' http://localhost:5000/predict
```
## Deploying as k8s pod: 
Add model2.pkl to Docker folder -> Then you would have all the required files to build a image for model server. <br>
![Docker-Build](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/docker.png)<br>
