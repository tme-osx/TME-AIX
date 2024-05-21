## Revenue Assurance with AI for Telco 
Fraud is mainly a classification problem, hence we picked RandomForestClassifier & BalancedRandomForestClassifier which are  appropriate due to their ability to handle complex, non-linear relationships and provide robust performance even with imbalanced data. <br>

#Data Characteristics:<br>
![Revenue Assurance Data Structure](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/rev_ass_data.png)<br>

#Model(s) Accuracy:<br>
![Revenue Assurance Accuracy](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/rev_ass_models_accuracy.png)<br>

Note: The trained models' accuracy is low due to used data is not necessarily holds VVVV (Volume, Velocity, Variety and Veracity) characteristics in required levels, however not that bad either :-). 

## Steps to Run:<br>
(1) (Obviously) Clone the Repo :-)  <br>
(2) Open revenueassurance/telco-revass.ipynb and Run-All. This would install all required libs, extract data from data/telecom_revass_data.csv.xz and load it for two models training {model1=RandomForestClassifier, model2=BalancedRandomForestClassifier }, evaluate them and save them under models/ directory. <br>
(3) Run model_test.py which loads the model2=BalancedRandomForestClassifier (better model for fraud classification) and serves on http://localhost:5000/predict url. <br>
(4) From a cli , do a curl query: <br>
![Revenue Assurance Test Result](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/revenueassurance/data/testresult.png)<br>
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
    "Last_Time_Pin_Used": 1, 
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
    "Last_Time_Pin_Used": 20,
    "Avg_Call_Duration": 12,
    "Avg_Data_Usage": 350
}' http://localhost:5000/predict
```

