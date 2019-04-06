import os 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Model 

#Importing our model of known weights that we will use for our transfer learning with CNN

base_model = InceptionV3 ( 
        weights = "imagenet", 
        include_top = False ,  #We dont include the last layer of the  model
        input_shape=(128,128,3) )
#Steps for making our entire dense layer and output softmax activation function

x=base_model.output 
x=MaxPooling2D()(x) 
x=Dense(units=256,activation="relu")(x)
x=Dropout(0.2)(x)
x=Dense(units=512,activation="relu")(x)
x=Dense(units=512,activation="relu")(x)
x=Dense(units=512,activation="relu")(x),,
x=Dense(units=256,activation="relu")(x)
x=Dropout(0.4)(x)
x=Flatten()(x)
num_classes=len(os.listdir("C:/ml_project/ObjectCategories_sets/TrainingSet"))
prediction = Dense(units=num_classes,activation="softmax")(x)
model=Model(inputs = base_model.input , outputs = prediction)

for layer in base_model.layers : 
    layer.trainable=False
        
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
test_datagen  =  ImageDataGenerator(rescale= 1./255)

#Creating the training and test set
training_set = train_datagen.flow_from_directory("C:/ml_project/ObjectCategories_sets/TrainingSet",
                                                 target_size=(128,128),
                                                 color_mode='rgb',
                                                 batch_size= 32 ,
                                                 class_mode="categorical")
test_set = test_datagen.flow_from_directory("C:/ml_project/ObjectCategories_sets/TestSet",
                                            target_size=(128,128),
                                            color_mode='rgb',
                                            batch_size=32 ,
                                            class_mode="categorical")
#Fitting the images to our CNN network created
history=model.fit_generator(training_set,
                   steps_per_epoch=250,
                   epochs=100,
                   validation_data=test_set ,
                   validation_steps=63)

model.save('ImageRec2.h5')


 
import matplotlib.pyplot as plt 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


import numpy as np
from keras.preprocessing import image 
test_image = image.load_img("C:/ml_project/ObjectCategories_sets/TrainingSet/yin_yang/image_0045.jpg",
                            target_size=(256,256))

test_image = image.img_to_array(test_image) #64x64
test_image = np.expand_dims(test_image,axis=0) #64x64x1
result = model.predict(test_image)
prediction=np.argmax(result)

for name, number in training_set.class_indices.items():
    if number == int(prediction):
        print(name)
