# the initial building block of Keras is a model,
# and the simplest model is called sequential
# sequential is a linear pipeline (a stack) of neural networks layers
from keras.models import Sequential
from keras.layers import Dense

# defining our sequential model
model = Sequential()

# adding a fully conected layer with 12 artifical neurons, and 8 features
# initializing weights with random values in range [-.05,.05]
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))
model.summary()
# keras provides a few initalization choices:
#   - random_uniform: random initialization in range [-.05,.05]
#   - random_normal: random initalization according to Gaussian,
#   with 0 mean and small std of 0.05 aka bell curve
#   - zero: all weights are zero
