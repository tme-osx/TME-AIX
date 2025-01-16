
# 5GCore 
This project delivers multi-source data analysis with Model Chaining (Classic-AI -> GenAI) leveraging Vector Store for Log Data Association. 

## DataSet
👉 https://huggingface.co/datasets/fenar/5gcore-prod
<div align="center">
    <img src="https://raw.githubusercontent.com/tme-osx/TME-AIX/refs/heads/main/5gprod/data/data.png"/>
</div>

## Metric file processing
It starts with a processing a telecom metric files for AMF, SMF and UPF<br>

## Anomaly detection
It uses a machine learning model for anomaly detection by using simple isolation forest algorithm.

```
Based on the provided logs, the anomaly detected is related to the latency_ms metric, which is a measure of the delay in data transmission. This metric is flagged as True in the anomaly column, indicating that there is an issue with the latency in the system.

Looking at the relevant UPF metrics information, we can see that the latency_ms value has increased from 13.37484865015994 at 09:46:55.775630 to 13.86797014781209 at 11:17:55.775630. This increase in latency could be due to several reasons such as network congestion, hardware issues, or software issues.

The CPU utilization and memory utilization have also increased during this time period. The CPU utilization has increased from 45.181238864488 to 50.542563686731 and the memory utilization has increased from 58.74739464810597 to 65.30386944707406. This increase in resource utilization could be contributing to the increased latency as the system may be struggling to process data efficiently due to the high resource usage.

The packet_processing_rate has decreased from 13404.863581966032 to 12357.272237015131, which could also be contributing to the increased latency. If the system is processing packets at a slower rate, this could lead to delays in data transmission.

The buffer_utilization has also increased from 37.98646570464442 to 40.893310839631, indicating that the system is storing more data in the buffer. This could be a result of the system struggling to process data efficiently, leading to an increase in latency.

In conclusion, the root cause of the anomaly in the latency_ms metric could be due to high CPU and memory utilization, a decrease in packet processing rate, and an increase in buffer utilization. Further investigation would be needed to determine the exact cause and to implement appropriate solutions.
```

## Root Cause Analysis -> Issue Resolution
After detection of the anomalies -> builds a VectorDB with Logs and finds assocated data pieces inside -> Passes to GenAI model that provides and RCA accrodingly<br>

```
Based on the provided system logs and alerts, the anomalies in the metrics can be attributed to several issues affecting different components of the system. 

1. Registration Storm: Both the Access and Mobility Management Function (AMF) and Session Management Function (SMF) were affected by a registration storm. This is a situation where a large number of devices attempt to register with the network simultaneously, causing a surge in registration rate. This can lead to increased CPU and memory utilization as the system tries to process the high volume of registration requests. 

2. Resource Exhaustion: Both the User Plane Function (UPF) and SMF experienced resource exhaustion. This indicates that these components were running out of resources, likely due to high demand. This can be seen in the increased CPU and memory utilization metrics for these components. 

3. Session Management Failure: The AMF, SMF, and UPF all experienced session management failures. This suggests that there were issues with establishing, maintaining, or terminating sessions. This could be due to a variety of reasons, including network congestion, software bugs, or hardware failures. 

4. Latency: The UPF experienced increased latency, as indicated by the 'True' value in the 'latency_ms' column of the anomalies table. This could be a result of the aforementioned issues, as increased CPU and memory utilization, as well as session management failures, can all contribute to increased latency. 

In conclusion, the root cause of the anomalies appears to be a combination of a registration storm, resource exhaustion, and session management failures, all of which led to increased CPU and memory utilization and increased latency. Further investigation would be needed to determine the exact cause of these issues and to develop appropriate solutions.
```


