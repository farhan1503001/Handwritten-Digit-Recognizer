# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:53:25 2018

@author: LENOVO
"""

import keras as k
import numpy as np
import matplotlib.pyplot as plt
minst=k.datasets.mnist
(x_train,y_train),(x_test,y_test)=minst.load_data()
#Normalizing the datasets
x_train=k.utils.normalize(x_train)
x_test=k.utils.normalize(x_test)
#Showing the training image
plt.imshow(x_train[0])
plt.show()
#Showing first test image
plt.imshow(x_test[1])
plt.show()
#Building the model
model=k.models.Sequential()
model.add(k.layers.Flatten())
model.add(k.layers.Dense(168,activation='relu'))
model.add(k.layers.Dense(128,activation='relu'))
model.add(k.layers.Dense(56,activation='tanh'))
model.add(k.layers.Dense(10,activation='softmax'))
#Compiling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#Fitting the model
model.fit(x_train,y_train,epochs=5)
val_loss,val_accuracy=model.evaluate(x_test,y_test)
print(val_loss,val_accuracy)
model.save('recognizer.h5')
newmodel=k.models.load_model('recognizer.h5')
predictions=newmodel.predict(x_test)
#resulting
result=np.argmax(predictions[0])
print(result," and real value",y_test[0])
plt.imshow(x_test[0])
plt.show()
model.summary()