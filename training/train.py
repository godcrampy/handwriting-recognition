
    
# creating dictionary for set of objects
import os
object_list = next(os.walk('data/train_dataset'))[1]
obj = {}
for n, category in enumerate(object_list):
    obj[n] = category

from keras.models import  Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


#CREATING A NEURAL NETWORK TO WORK ON
classifier = Sequential()
#Creating  input layer

#Layer 1
classifier.add(Conv2D(filters=32 ,kernel_size = (3,3),activation = "relu",input_shape=(128,128,3), padding = 'same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Layer 2
classifier.add(Conv2D(filters=32 ,kernel_size = (3,3),activation = "relu", padding = 'same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Layer 3
classifier.add(Conv2D(filters=64 ,kernel_size = (3,3),activation = "relu", padding = 'same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Layer 4
classifier.add(Conv2D(filters=64 ,kernel_size = (3,3),activation = "relu", padding = 'same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#flatten
classifier.add(Flatten())


classifier.add(Dense(units=2048,activation="relu"))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=2048,activation="relu"))
classifier.add(Dense(units=2048,activation="relu"))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=2048,activation="relu"))
classifier.add(Dense(units=2048,activation="relu"))
classifier.add(Dense(units=2048,activation="relu"))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=2048,activation="relu"))



#Creating the output fully connected layer
classifier.add(Dense(units=56,activation="softmax"))

#Compiling and optimizing the entire cnn ; we are going to use stochaistic GD
classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])


#TRAINING OUR TRAIN AND CROSS VAL AND TEST SET
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train_dataset',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data/test_dataset',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')
#Fitting the images to our CNN network created

history = classifier.fit_generator(training_set ,
                         steps_per_epoch=250 ,
                         epochs= 80,
                         validation_data = test_set ,
                         validation_steps=63)
#classifier.save('fourth.h5')


from tensorflow.keras.models import load_model

#classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# deletes the existing model
#del classifier
 
#classifier = load_model('my_model.h5')
#Training our model

import numpy as np
from keras_preprocessing import image
test_image = image.load_img("laptop.jpg",
                            target_size=(128,128))
test_image = image.img_to_array(test_image) #64x64
test_image= np.expand_dims(test_image,axis=0) #64x64x1
result = classifier.predict_classes(test_image)

for name, number in training_set.class_indices.items():
    if number == int(result):
        print(name)

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()