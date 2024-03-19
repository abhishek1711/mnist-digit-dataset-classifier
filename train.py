# import tools
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist
import numpy as np
import os


MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


def get_model():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_class, activation="softmax"))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    return model

input_shape = (28,28,1)
output_class = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# normalize the images
X_train = X_train/255.0
X_test = X_test/255.0


print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

#instanitiating the model
model = get_model()

 
# training the model 
BATCH_SIZE = 512
epochs = 11
model.fit(x=X_train, y=y_train, batch_size = BATCH_SIZE, epochs = epochs )

model.save(MODEL_PATH)