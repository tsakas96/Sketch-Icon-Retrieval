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

def load_icons(icon_sketches_dictionary):
    filename = 'icons.npy'
    filename2 = 'icons_name_cate.npy'
    if os.path.isfile(filename) and os.path.isfile(filename2):
        icons = np.load(filename, allow_pickle=True)
        name_cat = np.load(filename2, allow_pickle=True)
    else:
        p_ = []
        iconNames = []
        categories = []
        for icon, (category, _) in icon_sketches_dictionary.items():
            p = category + '/' + icon
            p_.append(os.path.join(dataset_path, 'icon/', p))
            iconName = icon.replace(".jpg","")
            iconNames.append(iconName)
            categories.append(category)
        
        name_cat = np.column_stack((iconNames, categories))
        np.save(open(filename2, 'wb'), name_cat, allow_pickle=True)
        icons = np.array([load_img(i) for i in p_])
        np.save(open(filename, 'wb'), icons, allow_pickle=True)

    return icons, name_cat

def load_sketches(icon_sketches_dictionary):
    filename = 'sketches.npy'
    filename2 = "sketch_names.npy"
    if os.path.isfile(filename):
        sketches = np.load(filename, allow_pickle=True)
        sketch_names_array = np.load(filename2, allow_pickle=True)
    else:
        s_ = []
        names = []
        for _, (category, sketch_list) in icon_sketches_dictionary.items():

            for sketch in sketch_list:
                s = category + '/' + sketch
                s_.append(os.path.join(dataset_path, 'sketch/', s))
                # I am storing the sketch name without _(#).png which is the corresponding name of the icon
                name = sketch.split("_")[0]
                names.append(name)
        
        sketch_names_array = np.array([name for name in names])
        np.save(open(filename2, 'wb'), sketch_names_array, allow_pickle=True)
        sketches = np.array([load_img(i) for i in s_])
        np.save(open(filename, 'wb'), sketches, allow_pickle=True)

    return sketches, sketch_names_array

def create_positive_sketch_icon_indices(icons_name_cat, sketch_names_array):
    icon_indices = []
    sketch_indices = []
    label_list = []
    label = 1
    for i in range(0, len(sketch_names_array)):
        for j in range(0, len(icons_name_cat)):
            if sketch_names_array[i] == icons_name_cat[j][0]:
                sketch_indices.append(i)
                icon_indices.append(j)
                label_list.append(label)
                break
    sketch_icon_indices = np.column_stack((sketch_indices, icon_indices))
    labels = np.array([l for l in label_list])
    return sketch_icon_indices, labels


