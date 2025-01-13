
# 5G Operations (OSS) Root Cause Analysis (RCA) with GenAI and RAG Use<br>

This project delivers multi-source data analysis with Model Chaining (Classic-AI -> GenAI) leveraging Vector Store for Log Data Association. 

## Metric file processing
It starts with a processing a telecom metric files for AMF, SMF and UPF<br>


## Anomaly detection
It uses a machine learning model for anomaly detection by using simple isolation forest algorithm.

## Root Cause Analysis 
After detection of the anomalies -> builds a VectorDB with Logs and finds assocated data pieces inside -> Passes to GenAI model that provides and RCA accrodingly<br>

## Example Test Output
Root Cause Analysis:<br>



