import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np


import pickle

# Separating file to be trained
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# Separating file to be train and test
size_data = int(0.9 * len(X))
size_test = int(len(X) - size_data)

train_data = X[:size_data]
train_test = X[-size_test:]

train_label = y[:size_data]
test_label = y[-size_test:]

X = X / float(255)



model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# this converts our 3D feature maps to 1D feature vectors
model.add(Flatten(input_shape=(50, 50)))

# Type of normalization where 20% of the model is sleeping
model.add(Dropout(0.2))

# Will transform all the negative numbers into 0
# Dense Layer will connect all the neurons
model.add(Dense(64, activation=tf.nn.relu))

# Softmax work with the probability of images which is gonna result between 1 and 0
model.add(Dense(2, activation=tf.nn.softmax))

# Compiling and training
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


hist = model.fit(X, y, batch_size=32, shuffle=True, epochs=30, validation_split=0.3)



model_saved = model.save('catndog.h5')

## Testing the accuracy of the model
teste = model.predict(train_label)

for t in range(20):
    print("Result of train:", np.argmax(teste[t]))
    print("Result of label:", test_label[t])




