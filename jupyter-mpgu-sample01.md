# Sample Code for Model Design with TF Disribution Strategy

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Build the model within the scope of the distribution strategy
with strategy.scope():
    modelx = Sequential()
    modelx.add(Dense(32, activation='relu', input_dim=len(X.columns)))
    modelx.add(Dropout(0.2))
    modelx.add(Dense(32))
    modelx.add(BatchNormalization())
    modelx.add(Activation('relu'))
    modelx.add(Dropout(0.2))
    modelx.add(Dense(32))
    modelx.add(BatchNormalization())
    modelx.add(Activation('relu'))
    modelx.add(Dropout(0.2))
    modelx.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    modelx.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
modelx.summary()
```
<br>

Sample Output
![alt text](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/images/jupyter-mgpu-modelsample-01.png)<br>

# Sample Code for Model Training with TF Disribution Strategy
```
epochs = 3
history = modelx.fit(X_train, y_train, epochs=epochs, \
                    validation_data=(scaler.transform(X_val.values),y_val), \
                    verbose = True, class_weight = class_weights)
print("Training of model is complete")
```
<br>

Sample Output
![alt text](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/images/jupyter-mgpu-modelsample-02.png)<br>
