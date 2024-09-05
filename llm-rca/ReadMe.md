
# AI Model Chaining Driven Root Cause Analysis (RCA) with Data-Correlation <br>

This project delivers multi-source data analysis with Model Chaining to detect and resolve anomalies, break down :<br>
     - Process a csv files containing time-series telecom metrics<br>
     - Find anomalies within the data<br>
     - Employs RAG for the systemd log files to find the correlation between the logs and anomalies<br>
     - Provides an root cause analysis and the end of the execution<br>

## Metric file processing
It starts with a processing a telecom metric file. The follwoing is an example of the metrics:<br>

|time                | call_attempt | call_success | call_failure | total_registered_subs |  call_success_rate |
---------------------|--------------|--------------|--------------|-----------------------|--------------------|
0 2024-09-04 00:00:00|           114|           110|             0|                   9031|               96.40|
1 2024-09-04 00:01:00|           113|           110|             0|                   9084|               97.34|
2 2024-09-04 00:02:00|           114|           111|             0|                   9089|               97.36|
3 2024-09-04 00:03:00|           113|           111|             1|                   9035|               98.23|
4 2024-09-04 00:04:00|           112|           111|             1|                   9092|               99.10|


It contains a machine learning model for anomaly detection by using simple isolation forest algorithm.<br>


## Anomaly detection
Anomalies found:<br>

|time                  | call_attempt | call_success | call_failure | total_registered_subs |  call_success_rate |is_anomaly|
-----------------------|--------------|--------------|--------------|-----------------------|--------------------|----------|
683 2024-09-04 11:23:00|           114|            27|             0|                   9031|               23.49|        -1|
684 2024-09-04 11:24:00|           113|            32|             0|                   9084|               28.36|        -1|
685 2024-09-04 11:25:00|           112|            40|             0|                   9089|               35.73|        -1|
686 2024-09-04 11:26:00|           114|            70|             2|                   9035|               61.49|        -1|


## LLM with RAG
After processing the log file via RAG, it provides and RCA accrodingly:<br>


## RCA (Root Cause Analysis)
Root Cause Analysis:<br>
Based on the provided logs and metrics, the anomalies in the metrics seem to be related to the OpenStack services, specifically the Open vSwitch service and the Nova Compute service.<br>

```
At 11:22:13, there is a log entry indicating an assertion failure in the Open vSwitch service, which leads to the service being killed and restarted. This could potentially disrupt network connectivity for the OpenStack services, affecting call attempts and successes.<br>

The Nova Compute service logs show several instances being migrated, rebooted, created, and shut down around the same time. This could potentially cause disruptions in the service, affecting the call success rate. Specifically, at 11:23:01, there is a log entry indicating the start of a migration for an instance, which could potentially disrupt the service.<br>

In addition, the total number of registered subscribers increases from 9033 to 9157 between 11:25:00 and 11:26:00. This sudden increase could potentially overload the system, leading to a decrease in the call success rate.<br>

In conclusion, the anomalies in the metrics could be caused by disruptions in the OpenStack services due to the Open vSwitch service failure and the Nova Compute service operations, as well as a sudden increase in the number of registered subscribers.<br>
```



