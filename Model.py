# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
def Model(X,y,inputSize):
    X = K.cast_to_floatx(X)
    y = K.cast_to_floatx(y)
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=inputSize, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=1, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)

    print('Accuracy: %.2f' % (accuracy * 100))
    return model
