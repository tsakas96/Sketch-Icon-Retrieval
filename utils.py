import cv2
import os
import numpy as np

dataset_path = "Sketch-Icon-Dataset/"
icon_path = os.path.join(dataset_path, 'icon/')
sketch_path = os.path.join(dataset_path, 'sketch/')


def load_img(path):
    img = cv2.imread(path)/255
    return cv2.resize(img, (100,100))


def get_dict_categories():
    
    icon_dictionary = {}

    for category in os.listdir(icon_path):
        category_path = os.path.join(icon_path, category)

        icon_dictionary[category] = os.listdir(category_path)

    sketch_dictionary = {}

    for category in os.listdir(sketch_path):
        category_path = os.path.join(sketch_path, category)

        sketch_dictionary[category] = os.listdir(category_path) 
    
    return icon_dictionary, sketch_dictionary

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


def get_batch(icon_dictionary, sketch_dictionary):
    l = []
    p_ = []
    s_ = []

    for _ in range(128):
        if np.random.uniform() >= 0.5:
            icon_class = np.random.choice(list(icon_dictionary))
            icon = np.random.choice(icon_dictionary[icon_class])
            icon_dictionary[icon_class].remove(icon)
            p = icon_class + '/' + icon

            sketch_class = icon_class
            sketch = np.random.choice(sketch_dictionary[sketch_class])
            sketch_dictionary[sketch_class].remove(sketch)
            s = sketch_class + '/' + sketch
            label = 1

        else:
            x = list(icon_dictionary)
            icon_class = np.random.choice(x)
            icon = np.random.choice(icon_dictionary[icon_class])
            icon_dictionary[icon_class].remove(icon)
            p = icon_class + '/' + icon
            x.remove(icon_class)

            sketch_class = np.random.choice(x)
            sketch = np.random.choice(sketch_dictionary[sketch_class])
            sketch_dictionary[sketch_class].remove(sketch)
            s = sketch_class + '/' + sketch
            label = 0

        p_.append(os.path.join(dataset_path, 'icon/', p))
        s_.append(os.path.join(dataset_path, 'sketch/', s))
        l.append(label)
    
    images = np.array([load_img(i) for i in p_])
    sketches = np.array([load_img(i) for i in s_])
    labels = np.array(l)

    return images, sketches, labels

def get_positive_pairs(icon_sketches_dictionary):
    l = []
    p_ = []
    s_ = []

    for icon, (category, sketch_list) in icon_sketches_dictionary.items():
        p = category + '/' + icon

        for sketch in sketch_list:
            s = category + '/' + sketch
            label = 1
            p_.append(os.path.join(dataset_path, 'icon/', p))
            s_.append(os.path.join(dataset_path, 'sketch/', s))
            l.append(label)
    
    images = np.array([load_img(i) for i in p_])
    sketches = np.array([load_img(i) for i in s_])
    labels = np.array(l)

    return images, sketches, labels

def load_icons(icon_sketches_dictionary):
    p_ = []

    for icon, (category, _) in icon_sketches_dictionary.items():
        p = category + '/' + icon
        p_.append(os.path.join(dataset_path, 'icon/', p))
    
    icons = np.array([load_img(i) for i in p_])
    icons.dump("icons.npy")

    return icons

def load_sketches(icon_sketches_dictionary):
    s_ = []

    for _, (category, sketch_list) in icon_sketches_dictionary.items():

        for sketch in sketch_list:
            s = category + '/' + sketch
            s_.append(os.path.join(dataset_path, 'sketch/', s))
    
    sketches = np.array([load_img(i) for i in s_])
    np.save(open('sketches.npy', 'wb'), sketches, allow_pickle=True)

    return sketches

dic = get_dict_icon_sketches()

#icons = load_icons(dic)
#print(icons.shape)
#sketches = load_sketches(dic)
sketches2 = np.load("sketches.npy", allow_pickle=True)
print(sketches2.shape)

#im, sk, lab = get_positive_pairs(dic)