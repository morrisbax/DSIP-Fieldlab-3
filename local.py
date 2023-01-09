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

def load_data(folder):

    #Creating empty dictionaries with empty lists
    images = []
    labels = []
    
    for patient in os.listdir(folder):
        patient_folder = os.path.join(folder, patient)
        for classif in os.listdir(patient_folder):
            img_folder = os.path.join(patient_folder, classif)
            if os.path.isdir(img_folder):
                for img_name in os.listdir(img_folder):
                    img = Image.open(os.path.join(img_folder, img_name))
                    np_img = np.asarray(img)
                    if(np_img.shape == (50,50,3)):
                        images.append(np_img)
                        labels.append(int(classif))
    
    return images, labels

def get_weights():
    
    weights = np.load('weights.npy', allow_pickle = True)
    batch_size = 64
    epochs = 3
    activation_function = "relu"
    dropout_rate1 = 0.01
    dropout_rate2 = 0.01
    learning_rate = 0.01
    
    return weights, batch_size, epochs, activation_function, dropout_rate1, dropout_rate2,learning_rate
    
    
      
def client_model(weights, batch_size, epochs, activation_function, dropout_rate1, dropout_rate2,learning_rate,images,labels):

    #Getting client images
    images = np.array(images)/255.0
    
    #Getting client labels
    labels = to_categorical(labels,num_classes=2)
    
    #Building client CNN
    cnn = build_cnn(activation_function, dropout_rate1, dropout_rate2, learning_rate)
    
    #Copying central weights to client CNN
    cnn.set_weights(weights)
    
    #Updating weights
    cnn.fit(images, labels, batch_size = batch_size, epochs=epochs)
    
    #Sending new weights
    new_weights = cnn.get_weights()
    
    return new_weights

def save_weights(weights):
    
    folder = "new_local_weights"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    n_files = len([f for f in os.listdir(folder)if os.path.isfile(os.path.join(folder, f))])
    
    with open(f"{folder}/new_local_weights{n_files+1}.npy", "wb") as f:
        np.save(f, weights)
    
    return        
    

def run(folder):
    
    print("loading weights from central server...")
    
    weights, batch_size, epochs, activation_function, dropout_rate1, dropout_rate2,learning_rate = get_weights()
    
    print("Accessing local data...")
    
    images, labels = load_data(folder)
    
    print("Training model using local data...")
    
    new_weights = client_model(weights, batch_size, epochs, activation_function, dropout_rate1, dropout_rate2,learning_rate,images,labels)

    print("Saving updated weights to central server...")
    
    save_weights(new_weights)
    
    print("Done!")
    
    return

# To get weights from central model, update those weights using local data and save those weights
# The parameter is the location of the folder containing data of one or more patients
run("/Users/morrisbax/Documents/DSIP/Patients_test1")

