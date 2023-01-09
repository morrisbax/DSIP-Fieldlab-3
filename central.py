#from google.colab import drive
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split, ParameterGrid
from tensorflow.keras.utils import to_categorical

def build_cnn(activation_function, dropout_rate1, dropout_rate2,learning_rate):
    input_shape = (50,50,3)
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation=activation_function, input_shape=(50,50,3)))
    model.add(Conv2D(64,kernel_size=(3,3),activation=activation_function))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout_rate1))
    model.add(Flatten())
    model.add(Dense(128,activation=activation_function))
    model.add(Dropout(dropout_rate2))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    return model

def initiate_central_model(reset = False):
    
    if os.path.exists("central_weights.npy") and not reset:
        print("Model already initialized")
        return
    
    batch_size = 64
    epochs = 3
    activation_function = "relu"
    dropout_rate1 = 0.01
    dropout_rate2 = 0.01
    learning_rate = 0.01
    
    model = build_cnn(activation_function, dropout_rate1, dropout_rate2,learning_rate)
    
    weights = model.get_weights()
    
    with open("central_weights.npy", "wb") as f:
        np.save(f, weights)
    
    return

def update_weights():
    
    weights = []
    if not os.path.exists("new_local_weights") or len(os.listdir("new_local_weights")) == 0:
        print("no new weights yet...")
        return
    
    for file in os.listdir("new_local_weights"):
        weights.append(np.load(f"new_local_weights/{file}", allow_pickle = True))
    
    sum_weights = weights[0]
    for i,x in enumerate(weights[1:]):
        sum_weights = np.add(sum_weights,x)
    
    avg_weights = sum_weights / len(weights)
    
    with open("central_weights.npy", "wb") as f:
        np.save(f, weights)
    
    return

def run(reset):

    print("Initializing model...")
    # To initialize the model (this only needs to be done once at the beginning):
    initiate_central_model(reset)
    
    print("Gathering new locally updated weights and updating central weigths...")
    # To update the model using updated weights from clients (which is done every 24h)
    update_weights()
    
    print("Done!")
    
    return

#Just run this to update weights. model is automatically initialized if it doesn't exsit yet
#To overwrite existing model, set reset to True
run(reset = False)
