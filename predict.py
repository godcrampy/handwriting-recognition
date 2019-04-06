from PIL import Image
import sys
from keras.models import load_model
import numpy as np

url = 'images/' + sys.argv[1]
image = Image.open(url)
image.show()

classifier = load_model('weights.h5')

classes = np.load('classes.npy').item()


from keras_preprocessing import image
test_image = image.load_img(url,
                            target_size=(128,128))
test_image = image.img_to_array(test_image) #64x64
test_image= np.expand_dims(test_image,axis=0) #64x64x1
result = classifier.predict_classes(test_image)

for name, number in classes.items():
    if number == int(result):
        print("\n\n\n=====================")
        print(name)
        print("=====================")

