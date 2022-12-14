import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

plt.imshow(x_train[0],cmap=plt.cm.binary)

x_train = x_train/255.0
x_test = x_test/255.0

y_train, y_test = y_train.flatten(), y_test.flatten()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=x_train[0].shape),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=50)

test_loss, test_acc = model.evaluate(x_test,y_test)

class_names=['airplane','automobile','bird','	cat','deer','dog','frog','horse','ship','truck']

prediction = model.predict(x_test)

for i in range(5):
  plt.grid(False)
  plt.imshow(x_test[i],cmap=plt.cm.binary)
  plt.xlabel('Actual:'+ class_names[y_test[i]])
  plt.title('Prediction :'+ class_names[np.argmax(prediction[i])]) 
  plt.show()

