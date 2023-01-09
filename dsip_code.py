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

def central_model(training_batch_size, epochs, activation_function, dropout_rate1, dropout_rate2, learning_rate, clients_batch_size, n_client_batches):
  
  #Building central CNN
  cnn = build_cnn(activation_function, dropout_rate1, dropout_rate2, learning_rate)

  #Loop over batches
  for batch in range(n_client_batches):
    
    #Loop over clients in batch
    batch_weights =[]
    for j in range(1+(batch*clients_batch_size),clients_batch_size+1+(batch*clients_batch_size)):

      #Retrieving central weights
      weights = cnn.get_weights()

      #Sending central weights to client to be updated
      client_weights = client_model(j, weights, training_batch_size, epochs, activation_function, dropout_rate1, dropout_rate2,learning_rate,)

      #Saving updated weights
      batch_weights.append(client_weights)
    
    #Getting average of updated weights
    sum_batch_weights = batch_weights[0]
    for i,x in enumerate(batch_weights[1:]):
      sum_batch_weights = np.add(sum_batch_weights,x)
    avg_batch_weights = sum_batch_weights / len(batch_weights)
    #Setting weights to updated weights
    cnn.set_weights(avg_batch_weights)

  return cnn

def client_model(client_number, weights, batch_size, epochs, activation_function, dropout_rate1, dropout_rate2,learning_rate):

  print(f"training at client number {client_number} ...")

  #Getting client images
  client_images = np.array(images[f"client{client_number}"])/255.0

  #Getting client labels
  client_labels = to_categorical(labels[f"client{client_number}"],num_classes=2)

  #Building client CNN
  cnn = build_cnn(activation_function, dropout_rate1, dropout_rate2, learning_rate)

  #Copying central weights to client CNN
  cnn.set_weights(weights)

  #Updating weights
  cnn.fit(client_images, client_labels, batch_size = batch_size, epochs=epochs)

  #Sending new weights
  new_weights = cnn.get_weights()
  return new_weights

def test_cnn(test_model,images,labels):
  
  #Getting test images
  test_images = np.array(images["client0"])/255.0

  #Getting test labels
  test_labels = to_categorical(labels["client0"],num_classes=2)

  #Scoring
  test_scores = test_model.evaluate(test_images,test_labels)

  return test_scores

def load_data(folder,n_clients,patients_per_client):

  """
  # This will prompt for authorization.
  drive.mount('/content/drive')
  """

  n_clients = n_clients + 1 #We need one for testing
  
  #Creating empty dictionaries with empty lists
  images = {}
  labels = {}
  for n in range(n_clients):
    key = "client{}".format(n)
    images[key] = []
    labels[key] = []

  #Loading data into dictionaries
  i = 0
  j = 0
  print(f"loading client {j}")
  key = "client0"
  count = 0
  for patient in os.listdir(folder):
    patient_folder = os.path.join(folder, patient)
    if os.path.isdir(patient_folder):
        for classif in os.listdir(patient_folder):
          img_folder = os.path.join(patient_folder, classif)
          if os.path.isdir(img_folder):
              for img_name in os.listdir(img_folder):
                img = Image.open(os.path.join(img_folder, img_name))
                np_img = np.asarray(img)
                if(np_img.shape == (50,50,3)):
                  images[key].append(np_img)
                  labels[key].append(int(classif))
    count += 1
    
    #Stop if enough clients have enough patients loaded
    if i == n_clients*patients_per_client-1:
      break
    i+=1

    #Go to next client if currect client is done
    if i % patients_per_client == 0:
      j += 1
      key = "client{}".format(j)
      print(f"loading client {j} ...")
      count = 0

  return images, labels

def regular_cnn(training_batch_size, epochs, activation_function, dropout_rate1, dropout_rate2, learning_rate,images,labels):
  
  #Convert to regular data set without clients
  train_images = np.concatenate([images[x] for x in list(images.keys())[1:]], 0)
  train_labels = np.concatenate([labels[x] for x in list(labels.keys())[1:]], 0)

  #Preprocess data
  train_images = np.array(train_images)/255.0
  train_labels = to_categorical(train_labels,num_classes=2)

  #Build and fit model
  cnn = build_cnn(activation_function, dropout_rate1, dropout_rate2, learning_rate)
  cnn.fit(train_images, train_labels, batch_size = training_batch_size, epochs=epochs)

  return cnn

def predict(model, images):
  
  prediction = model.predict(images)
  
  return prediction

def check_data(data):
  
  if type(images) is not dict:
    print(False)
  for key, value in images.items():
    shape = np.shape(images[key])
    if shape[1] != 50 or shape[2] != 50 or shape[3] != 3:
      print(False)
    else:
      print(True)
 
  return


images,labels = load_data(folder = "/Users/morrisbax/Documents/DSIP/Patients",n_clients = 9,patients_per_client = 15)



#%%

print(len(images))

#%%

federated_cnn = central_model(64, 3, "relu", 0.01, 0.01, 0.01,clients_batch_size = 3,n_client_batches = 3)
fed_score = test_cnn(federated_cnn,images,labels)
print(fed_score)

reg_cnn = regular_cnn(64, 3, "relu", 0.01, 0.01, 0.01,images,labels)
reg_score = test_cnn(reg_cnn,images,labels)
print(reg_score)

