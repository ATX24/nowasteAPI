from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU

from keras.optimizers import Adam as Adam

def rohanNet():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(227,227,3),padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(MaxPooling2D((2, 2),padding='same'))
  model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
  model.add(LeakyReLU(alpha=0.1))                  
  model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(Flatten())
  model.add(Dense(128, activation='linear'))
  model.add(LeakyReLU(alpha=0.1))                  
  model.add(Dense(2, activation='softmax'))

  model.summary()

  # Compile the model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model