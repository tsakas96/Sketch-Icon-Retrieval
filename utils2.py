import cv2
import os
import numpy as np
import copy

dataset_path = "Sketch-Icon-Dataset/"
icon_path = os.path.join(dataset_path, 'icon/')
sketch_path = os.path.join(dataset_path, 'sketch/')


def load_img(path):
    img = cv2.imread(path)/255
    return cv2.resize(img, (100,100))

def get_dict_icon_sketches():
    
    icon_sketches_dictionary = {}

    for category in os.listdir(icon_path):
        category_path_icon = os.path.join(icon_path, category)
        category_path_sketch = os.path.join(sketch_path, category)
        for icon in os.listdir(category_path_icon):
            sketch_list = []
            for sketch in os.listdir(category_path_sketch):
                if icon.replace(".jpg","") == sketch.split("_")[0]:
                    sketch_list.append(sketch)
            icon_sketches_dictionary[icon] = (category, sketch_list)
    
    return icon_sketches_dictionary


def get_batch(icon_sketches_dictionary, BATCH_SIZE):
    l = []
    i_ = []
    s_ = []
    dic = copy.deepcopy(icon_sketches_dictionary)

    for _ in range(BATCH_SIZE):
        # positive pair
        if np.random.uniform() >= 0.5:
            icon = np.random.choice(list(dic.keys()))
            category = dic[icon][0]
            i = icon_path + category + '/' + icon

            sketch = np.random.choice(dic[icon][1])
            # remove the sketch from the dictionary so it will not choose the same sketch for an icon
            #dic[icon][1].remove(sketch)
            dic.pop(icon)
            s = sketch_path + category + '/' + sketch
            label = 1
        # negative pair
        else:
            icon = np.random.choice(list(dic.keys()))
            category = dic[icon][0]
            dic.pop(icon)
            i = icon_path + category + '/' + icon

            random_icon = np.random.choice(list(dic.keys()))
            while random_icon == icon:
                random_icon = np.random.choice(list(dic.keys()))

            sketch_category = dic[random_icon][0]
            sketch = np.random.choice(dic[random_icon][1])
            s = sketch_path + sketch_category + '/' + sketch
            label = 0
        i_.append(i)
        s_.append(s)
        l.append(label)
    
    
    icons = np.array([load_img(i) for i in i_])
    sketches = np.array([load_img(i) for i in s_])
    labels = np.array(l)

    return icons, sketches, labels