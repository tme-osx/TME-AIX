# Telco Media Entertainment AI eXperiements
This is a collaborative workspace repo for various tme use-cases to be build around of AI capabilities & Open Data.<br>

(1) Revenue Assurance Fraud Detection: [Link](https://github.com/fenar/tme-aix/tree/main/revenueassurance)<br>
Data Structure: <br>

```
    revass_data = pd.DataFrame({
        'Call_Duration': call_duration,
        'Data_Usage': data_usage,
        'Sms_Count': sms_count,
        'Roaming_Indicator': roaming_indicator,
        'MobileWallet_Use': mobilewallet_use,
        'Plan_Type': plan_type,
        'Cost': cost,
        'Cellular_Location_Distance': cellular_location_distance,
        'Personal_Pin_Used': personal_pin_used,
        'Avg_Call_Duration': avg_call_duration,
        'Avg_Data_Usage': avg_data_usage,
        'Avg_Cost': avg_cost,
        'Fraud': fraud.astype(int)
    })
```
