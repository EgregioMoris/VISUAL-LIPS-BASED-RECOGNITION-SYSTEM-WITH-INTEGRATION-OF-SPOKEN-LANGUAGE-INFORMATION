#Implementazione modello neurale ConvLSTM per il Riconoscimento della lingua parlata "Soggetto Visto"
import dill
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import ConvLSTM2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization, AveragePooling3D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

def create_samples(x, y): 
  samples = []
  for i in range(0, len(x)): 
    samples.append((x[i], y[i]))
  
  return samples

def generator_(samples, batch_size):

  num_samples = len(samples)
  while True:
    for offset in range(0, num_samples, batch_size): 

      batch_samples = samples[offset:offset+batch_size]
      batch_target = samples[offset:offset+batch_size]

      x_train = []
      y_train = []

      for i in range(0, len(batch_samples)): 
        x_train.append(batch_samples[i][0])
        y_train.append(batch_target[i][1])

      x_train = np.array(x_train)
      y_train = np.array(y_train)
      
      yield x_train, y_train

x_test_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/[New]Esperimenti/S1_old_dataset/data_binaries/x_test_0", "rb"))
y_test_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/[New]Esperimenti/S1_old_dataset/data_binaries/y_test", "rb"))

x_train_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/[New]Esperimenti/S1_old_dataset/data_binaries/x_train_0", "rb"))
y_train_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/[New]Esperimenti/S1_old_dataset/data_binaries/y_train", "rb"))

batch_size = 6


samples_train = create_samples(x_train_conv, y_train_conv)
samples_test = create_samples(x_test_conv, y_test_conv)




train_generator = generator_(samples_train, batch_size)
test_generator = generator_(samples_test, batch_size)



#Risultati
path = "C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/[New]Esperimenti/S1_old_dataset/result/6_batch"
callback_check = ModelCheckpoint(path, monitor="val_accuracy", 
                                 save_best_only=True)

callbacks = [callback_check]

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_path,
  save_weights_only=True,
  verbose=1,
  save_freq=5*batch_size)




model = Sequential()

model.add(ConvLSTM2D(64, kernel_size=(3, 3), return_sequences=True, data_format="channels_last", input_shape=x_train_conv.shape[1:]))

model.add(Dropout(0.5))

model.add(ConvLSTM2D(96, kernel_size=(3, 3), return_sequences=True))
model.add(Dropout(0.5))




model.add(Flatten())
model.add(Dense(16))
model.add(Dense(8, activation="softmax"))


opt = tf.keras.optimizers.Adamax(0.0001)#2
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, 
              metrics="accuracy")

#model.load_weights(checkpoint_path)

epochs = 200

history = model.fit(train_generator, shuffle=True, epochs=epochs, steps_per_epoch=32, 
          validation_data=(x_test_conv, y_test_conv), callbacks=callbacks)
          #50, 21

history2 = model.evaluate(x_test_conv, y_test_conv)



accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]


loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(range(epochs), accuracy, label='Training Accuracy')
plt.plot(range(epochs), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(range(epochs), loss, label='Training Loss')
plt.plot(range(epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.title('Training and Validation Loss')
plt.show()



