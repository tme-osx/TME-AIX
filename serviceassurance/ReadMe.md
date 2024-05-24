
# Service Assurance Insights using Transformer Neural Network

## Project Overview

This project aims to deliver a service assurance insights model using a Transformer neural network. The model is trained on synthetic telecom data to predict network performance metrics and identify potential anomalies. The goal is to provide proactive network management and enhance customer experience.

## Data

The synthetic telecom data includes the following key metrics:
- **Timestamp**: Date and time of the record.
- **Latency**: Network latency in milliseconds.
- **Jitter**: Variation in packet delay.
- **Packet Loss**: Percentage of packets lost.
- **Throughput**: Data transfer rate.
- **MTU**: Maximum Transmission Unit size.
- **CPU Usage**: CPU usage percentage.
- **Memory Usage**: Memory usage percentage.
- **Port Speed**: Network port speed in Gbps.
- **Alarm**: Boolean indicating if an alarm was triggered.
- **Traffic Volume**: Volume of network traffic.
- **Data Speed**: Data transfer speed.

The dataset is saved in a compressed CSV file: `data/telecom_revass_data_v6.csv.xz`.

## Model

The model is based on a Transformer neural network, which is well-suited for sequential data. The architecture includes:
- Embedding layer to convert input sequences.
- Transformer blocks with multi-head attention.
- Fully connected layers for regression output.

### Transformer Block

A custom Transformer block includes:
- Multi-head attention mechanism.
- Feed-forward network.
- Layer normalization and dropout for regularization.

## Training and Evaluation

### Training

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. Early stopping is employed to prevent overfitting. The dataset is split into training and validation sets with an 80-20 split.

### Evaluation

The model's performance is evaluated using the Mean Absolute Percentage Error (MAPE), which provides an accuracy-like metric for regression problems. The evaluation metric gives insights into the model's prediction error as a percentage.

## Results

The model predictions are compared against actual values for key metrics like latency. The MAPE value is calculated and printed, providing a quantitative measure of the model's performance.

## Visualization

The project includes visualizations to compare the actual and predicted values for network latency. The plots provide a clear view of the model's prediction accuracy.

## Summary

This project demonstrates the application of Transformer neural networks for predicting network performance metrics in a telecom environment. The use of synthetic data and the evaluation of the model using MAPE provide a comprehensive approach to service assurance insights.
