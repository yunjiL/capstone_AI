from tensorflow.python.keras.initializers import glorot_uniform
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import  img_to_array
from dataclasses import dataclass
import glob
import os

class NUM_class:
    class_A: int = 0;
    class_B: int = 0;
    class_C: int = 0;

classA = ['JD', 'BK'];
classB = ['SD'];
classC = ['CC', 'CL', 'LP','JF','DS'];

class Defect_info:
    Img_loc: str;
    defect: str;
    Class: chr;

class pipe_info:
    type: str;
    length: float;
    diameter: float;

pipe_cost = 0;


def count_gauge(NUM_total_img, NUM_complete_pipe):
    calc_progress = NUM_complete_pipe / NUM_total_img;
    
    return calc_progress;

def calculate_risk(NUM_class):
    Defect_sum = 0;
    
    if NUM_class.class_A != 0:
        return 1;
    Defect_sum += NUM_class.class_B * 20;
    Defect_sum += NUM_class.class_C * 5;
    
    if Defect_sum >= 100:
        return 1;
    else:
        return 0;


html_string_start = """
{% extends "layout.html" %}
{% block content %}
    <form class="form-signin" method=post enctype=multipart/form-data>
		<h1 class="h2 mb-3">Sewer Defect Segmentation</h1>
"""

html_string_end = """
    </form>
{% endblock %}
"""

def get_picture_html(out, Imgclass, NUM_complete_pipe):
    image_html = """
    <tr>
        <th scope="row">{num}</th>
        <td>{Imgclass_}</td>
        <td></td>
        <td><img src="../visualization/sd/{out_name}" width=200></td>
    </tr>
    """
    return image_html.format(out_name=out, Imgclass_=Imgclass, num = NUM_complete_pipe)


def get_count_html(category, count):
    count_html = """<li> {category_name} : {count_} </li>"""
    return count_html.format(category_name=category, count_=count)


def get_value_count(image_class_dict):
    count_dic = {}
    for category in image_class_dict.values():
        if category in count_dic.keys():
            count_dic[category] = count_dic[category] + 1
        else:
            count_dic[category] = 1
    return count_dic


def generate_html(out=None, Imgclass=None, NUM_complete_pipe=None):
    picture_html = ""

    if out is not None:
        if out.split('.')[1] == 'jpg' or out.split('.')[1] == 'png':
            picture_html += get_picture_html(out, Imgclass, NUM_complete_pipe)

    file_content = picture_html

    with open('templates/generate.html', 'a') as f:
        f.write(file_content)
        
def detect_defect(img, model, folder_path):
    # dimensions of images
    img_width, img_height = 224, 224
    i = 0
    images = []
    img1 = image.load_img(os.path.join(folder_path, img), target_size=(img_width, img_height))
    img2 = img_to_array(img1)
    img2 = np.expand_dims(img2, axis=0)
    img2.shape
    classes = model.predict(img2)[0]
    idxs = np.argsort(classes)[::-1][:2]

    classname = ['BK', 'CC', 'CL', 'DS', 'ETC', 'JD', 'JF', 'LP',
                'SD', 'UP_IN', 'UP_PJ']

    out = cv2.imread(os.path.join(folder_path, img))
    img_class = ""
    for (i, j) in enumerate(idxs):
        if classes[idxs[i]] * 100 <= 5:
            continue;
        label = "{}:{:.2f}%".format(classname[idxs[i]], classes[idxs[i]] * 100)
        img_class += classname[idxs[i]] + " ";
        if classname[idxs[i]] in classA:
            NUM_class.class_A += 1;
        elif classname[idxs[i]] in classB:
            NUM_class.class_B += 1;
        elif classname[idxs[i]] in classC:
            NUM_class.class_C += 1;
        
        cv2.putText(out, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    generate_html(img, img_class, NUM_complete_pipe);
    
    cv2.imwrite("visualization/sd/%s"%img,out)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# image folder
folder_path = 'dataset/test/'

# path to model
model_path = 'model/sewer_weight.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(model_path)

#이미지 총 개수
NUM_total_img = len(os.listdir(folder_path));
NUM_complete_pipe = 0;

#이미지 결함 분류
for img in os.listdir(folder_path):
    NUM_complete_pipe += 1;
    detect_defect(img, model, folder_path);
    calc_progress = count_gauge(NUM_total_img, NUM_complete_pipe);

threshold = calculate_risk(NUM_class);