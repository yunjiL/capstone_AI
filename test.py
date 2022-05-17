
from tensorflow.python.keras.initializers import glorot_uniform
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import  img_to_array
from dataclasses import dataclass
#import sewer_defect


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# image folder
folder_path = 'dataset/test/'

# path to model
model_path = 'model/sewer_weight.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(model_path)

# dimensions of images
img_width, img_height = 224, 224

i = 0
images = []
for img in os.listdir(folder_path):
   img1 = image.load_img(os.path.join(folder_path, img), target_size=(img_width, img_height))
   img2 = img_to_array(img1)
   img2 = np.expand_dims(img2, axis=0)
   classes = model.predict(img2)[0]
   idxs = np.argsort(classes)[::-1][:2]

   classname = ['BK', 'CC', 'CL', 'DS', 'ETC', 'JD', 'JF', 'LP',
                'SD', 'UP_IN', 'UP_PJ']

   out = cv2.imread(os.path.join(folder_path, img))

   for (i, j) in enumerate(idxs):
        if classes[idxs[i]] * 100 <= 5:
           continue;
        label = "{}:{:.2f}%".format(classname[idxs[i]], classes[idxs[i]] * 100)
        cv2.putText(out, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #numclass로 class 분류하여 저장
    
   cv2.imwrite("visualization/sd/%s"%img,out)
   