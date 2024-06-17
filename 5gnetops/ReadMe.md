# 5G RAN Fault Prediction with AI Assistance

This project contains a machine learning model for predicting fault occurrence rates in a 5G radio network based on various network KPIs. The model is fine-tuned on a dataset containing network KPIs and fault occurrence rates. The server is built using Flask to provide predictions through a REST API. <br><br>
Model-Card: https://huggingface.co/google-t5/t5-small <br><br>

Data Sources:This data set is correlated child of following two real world datasets and feature engineered for the remaining ones: <br>
[>>FCC Customer Complaints Data Set](https://opendata.fcc.gov/Consumer/CGB-Consumer-Complaints-Data/3xyp-aqkj/about_data)<br>
[>>OpenWeather Data Set](https://openweathermap.org/)<br>

## Project Structure
- Training with 5g Operation KPI Data Structures:
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/5gdatasetsnapshot2.png)<br>
- Training Args & Perf Results:
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/trainingargs.png)<br>
- Model Server Experience (Perfect Day vs Rainy-Miserable Day in New York):
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/modelserver.png)<br>
![](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/testresults.png)<br>

