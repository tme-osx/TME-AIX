
# Service Assurance Insights using Transformer Neural Network

## Project Overview

This project aims to deliver a service assurance insights (net promoter score -NPS- for business growth predicted based on performance of the telco service architecture) model using a Transformer Neural Network. The model is trained on synthetic telecom  network stack data to predict network performance metrics and predict NPS. The goal is to provide business vs network performance correlation for planned growth.

## Data
![Service Assurance Data Structure](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/serviceassurance/data/sevass-data.png)<br>

The dataset is saved in a compressed CSV file: `data/telecom_sevass_data.csv.xz`.

## Model

The model is based on a Transformer neural network (175K parms), which is well-suited for sequential data. The architecture includes:
- Embedding layer to convert input sequences.
- Transformer blocks with multi-head attention.
- Fully connected layers for regression output.

### Transformer Block
![Service Assurance Data Structure](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/serviceassurance/data/sevass-nn.png)<br>

## Training and Evaluation

### Training

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. Early stopping is employed to prevent overfitting. The dataset is split into training and validation sets with an 80-20 split.

### Evaluation

The model's performance is evaluated using the Mean Absolute Percentage Error (MAPE), which provides an accuracy-like metric for regression problems. The evaluation metric gives insights into the model's prediction error as a percentage.

## Results

The model predictions are compared against actual values for key metrics like latency. The MAPE value is calculated and printed, providing a quantitative measure of the model's performance.
![Service Assurance Result](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/serviceassurance/data/sevass-nps.png)<br>

## Summary

This project demonstrates the application of Transformer neural networks for predicting network performance metrics in a telecom environment. The use of synthetic data and the evaluation of the model using MAPE provide a comprehensive approach to service assurance insights.
