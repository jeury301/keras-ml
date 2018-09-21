# using keras to define a network that recognizes MNIST handwritten digits.
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
np.random.seed(1671) # for reproductibility

# network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = RMSprop() # SGD optimizer
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

# data: shuffled and split between train and test sets
# X_train is 60000 rows of 28x28 values --> reshaped 60000x784
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshaping the X matrices
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED) # 60k x 784
X_train = X_train.astype('float32') # optimized for gpu computation
X_test = X_test.reshape(10000,RESHAPED) # 10k x 784
X_test = X_test.astype('float32') # optimized for gpu computation

print(X_train.shape, X_test.shape)

# normalizing the data
X_train /= 255 # values -> [0,1]
X_test /= 255 # values -> [0,1]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# converting class vectors to binary class matrices AKA ONE-HOT encoding
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

print(Y_train.shape, Y_test.shape)

# 10 outputs
# using softmax - which is a generalization of the sigmoid function
# Softmax squashes a k-dimensional vector of arbitrary real values into
# a k-dimensional vector of real values in the range(0,1)
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,))) # adding fully connected layer
model.add(Activation('relu')) # added after 92% accuracy
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN)) # added after 92% accuracy
model.add(Activation('relu')) # added after 92% accuracy
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES)) # added after 92% accuracy
model.add(Activation('softmax'))
model.summary()

# compiling the model
# the categorical cross-entropy is the default loss function for softmax
# this loss function is suitable for multiclass label prediction
# we are using the metric: accuracy, which is the proportion of correct predictions
# with respect to the targets
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,metrics=['accuracy'])

# training the model
history = model.fit(X_train, Y_train,
    batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
    validation_split = VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
