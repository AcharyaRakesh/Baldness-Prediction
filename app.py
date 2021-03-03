
from main import load_model

model = load_model('model.h5')

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('',target_size=(64,64) )
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image=np.expand_dims(test_image,axis=0)
result = model.predict(test_image)

if result[0]<=0.5:
    print('image classified as Bald')
else:
    print('image is not')