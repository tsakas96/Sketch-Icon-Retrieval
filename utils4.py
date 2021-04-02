import cv2
import os
import numpy as np
import torch
import copy

dataset_path = "Sketch-Icon-Dataset/"
icon_path = os.path.join(dataset_path, 'icon/')
sketch_path = os.path.join(dataset_path, 'sketch/')


def load_img(path):
    img = cv2.imread(path)/255
    return cv2.resize(img, (100,100))

def load_train_set():
    return np.load(dataset_path + "train_set.npy")

def load_test_set():
    return np.load(dataset_path + "test_set.npy")

def get_icons_and_sketches():
    
    icon_name = []
    icon_category = []
    sketch_name = []
    sketch_category = []

    for category in os.listdir(icon_path):
        category_path_icon = os.path.join(icon_path, category)
        category_path_sketch = os.path.join(sketch_path, category)
        for icon in os.listdir(category_path_icon):
            icon_name.append(icon)
            icon_category.append(category)
        for sketch in os.listdir(category_path_sketch):
            sketch_name.append(sketch)
            sketch_category.append(category)
    
    icon_name_category =  np.column_stack((icon_name, icon_category))
    sketch_name_category =  np.column_stack((sketch_name, sketch_category))
    
    return icon_name_category, sketch_name_category

def load_icons_sketches_dic(icon_name_category, sketch_name_category):
    icon_dictionary = {}
    for icon_name, icon_category in icon_name_category:
        path = icon_path + icon_category + "/" + icon_name
        icon_data = load_img(path)
        icon_dictionary[icon_name] = icon_data
    
    sketch_dictionary = {}
    for sketch_name, sketch_category in sketch_name_category:
        path = sketch_path + sketch_category + "/" + sketch_name
        sketch_data = load_img(path)
        sketch_dictionary[sketch_name] = sketch_data
    return icon_dictionary, sketch_dictionary

def load_icons_sketches_dic2(icon_name_category, sketch_name_category):
    icon_dictionary = {}
    for icon_name, icon_category in icon_name_category:
        path = icon_path + icon_category + "/" + icon_name
        with open(path,'rb') as f:
          icon_data = f.read() 
          icon_dictionary[icon_name] = icon_data
    
    sketch_dictionary = {}
    for sketch_name, sketch_category in sketch_name_category:
        path = sketch_path + sketch_category + "/" + sketch_name
        with open(path,'rb') as f:
          sketch_data = f.read()
          sketch_dictionary[sketch_name] = sketch_data
    return icon_dictionary, sketch_dictionary

# create sketch to image specific positive pairs
def positive_pairs_generator(icon_name_category, sketch_name_category):
    positive_pairs_list = []
    for sketch, sketch_category in sketch_name_category:
        for icon, icon_category in icon_name_category:
            if sketch_category == icon_category and icon.replace(".jpg","") == sketch.split("_")[0]:
                positive_pairs_list.append((sketch, sketch_category, icon, icon_category))
                break

    positive_pairs_array = np.array(positive_pairs_list)
    return positive_pairs_array

def possible_negative_pairs_generator(positive_pairs_array, icon_name_category):
    possible_negative_pairs = {}
    for sketch, sketch_category,_,_ in positive_pairs_array:
        negative_icon_category_list = []
        for icon, category in icon_name_category:
            if category == sketch_category and icon.replace(".jpg","") != sketch.split("_")[0]:
                negative_icon_category_list.append((icon, category))
        possible_negative_pairs[sketch] = negative_icon_category_list
    return possible_negative_pairs

def triplets_generator(positive_pairs_array, icon_name_category, possible_negative_pairs):
    negative_icon_category_list = []
    for sketch,_,_,_ in positive_pairs_array:
        temp_list = possible_negative_pairs[sketch]
        random_index = int(np.random.randint(0, len(temp_list)))
        negative_icon_category = temp_list[random_index]
        #negative_icon_category = find_negative_icon(sketch, sketch_category, icon_name_category)
        negative_icon_category_list.append(negative_icon_category)
    negative_icon_category_array = np.array(negative_icon_category_list)
    triplet_array = np.append(positive_pairs_array, negative_icon_category_array, axis=1)
    return triplet_array

# def triplets_generator(positive_pairs_array, icon_name_category):
#     negative_icon_category_list = []
#     for sketch, sketch_category,_,_ in positive_pairs_array:
#         negative_icon_category = find_negative_icon(sketch, sketch_category, icon_name_category)
#         negative_icon_category_list.append(negative_icon_category)
#     negative_icon_category_array = np.array(negative_icon_category_list)
#     triplet_array = np.append(positive_pairs_array, negative_icon_category_array, axis=1)
#     return triplet_array



# it finds a negative icon from the same category
def find_negative_icon(sketch, sketch_category, icon_name_category):
    negative_icon_category_list = []
    for icon, category in icon_name_category:
        if category == sketch_category and icon.replace(".jpg","") != sketch.split("_")[0]:
            negative_icon_category_list.append((icon, category))

    random_index = int(np.random.randint(0, len(negative_icon_category_list)))
    negative_icon_category = negative_icon_category_list[random_index]
    return negative_icon_category

# This code implements the triplets of sketches with the negative pairs with all icons of the same category
# def triplets_generator(positive_pairs_array, icon_name_category):
#     negative_icon_category_list = []
#     positive_pairs_list = []
#     for sketch, sketch_category, positive_icon_name, positive_icon_category in positive_pairs_array:
#         negative_icon_category = find_negative_icon(sketch, sketch_category, icon_name_category)
#         for row in negative_icon_category:
#             negative_icon_category_list.append(row)
#             positive_pairs_list.append((sketch, sketch_category, positive_icon_name, positive_icon_category))
#     negative_icon_category_array = np.array(negative_icon_category_list)
#     positive_array = np.array(positive_pairs_list)
#     triplet_array = np.append(positive_array, negative_icon_category_array, axis=1)
#     return triplet_array

# # it finds a negative icon from the same category
# def find_negative_icon(sketch, sketch_category, icon_name_category):
#     negative_icon_category_list = []
#     for icon, category in icon_name_category:
#         if category == sketch_category and icon.replace(".jpg","") != sketch.split("_")[0]:
#             negative_icon_category_list.append((icon, category))
#     return negative_icon_category_list

def get_batch(batch_triplet_pairs, icon_dictionary, sketch_dictionary):
    s_ = []
    p_ = []
    n_ = []
    for row in batch_triplet_pairs:
        sketch_name = row[0]
        positive_icon_name = row[2]
        negative_icon_name = row[4]

        s_.append(sketch_dictionary[sketch_name])
        p_.append(icon_dictionary[positive_icon_name])
        n_.append(icon_dictionary[negative_icon_name])
    
    sketches = np.array(s_)
    positive_icons = np.array(p_)
    negative_icons = np.array(n_)

    return sketches, positive_icons, negative_icons

def get_batch_low_ram(batch_triplet_pairs):
    s_ = []
    p_ = []
    n_ = []
    for row in batch_triplet_pairs:
        sketch_name = row[0]
        sketch_category = row[1]
        positive_icon_name = row[2]
        positive_icon_category = row[3]
        negative_icon_name = row[4]
        negative_icon_category = row[5]

        s_.append(sketch_path + sketch_category + "/" + sketch_name)
        p_.append(icon_path + positive_icon_category + "/" + positive_icon_name)
        n_.append(icon_path + negative_icon_category + "/" + negative_icon_name)
    
    sketches = np.array([load_img(i) for i in s_])
    positive_icons = np.array([load_img(i) for i in p_])
    negative_icons = np.array([load_img(i) for i in n_])

    return sketches, positive_icons, negative_icons

def get_batch_triplet_classification(batch_triplet_pairs, icon_dictionary, sketch_dictionary, icon_categories_dic, sketch_categories_dic):
    s_ = []
    p_ = []
    n_ = []
    i_c = []
    s_c = []
    for row in batch_triplet_pairs:
        sketch_name = row[0]
        sketch_category = row[1]
        positive_icon_name = row[2]
        positive_icon_category = row[3]
        negative_icon_name = row[4]

        s_.append(sketch_dictionary[sketch_name])
        p_.append(icon_dictionary[positive_icon_name])
        n_.append(icon_dictionary[negative_icon_name])
        i_c.append(icon_categories_dic[positive_icon_category])
        s_c.append(sketch_categories_dic[sketch_category])
    
    sketches = np.array(s_)
    positive_icons = np.array(p_)
    negative_icons = np.array(n_)
    icon_targets = np.array(i_c)
    sketch_targets = np.array(s_c)

    return sketches, positive_icons, negative_icons, icon_targets, sketch_targets

def get_batch_triplet_classification_low_ram(batch_triplet_pairs, icon_categories_dic, sketch_categories_dic):
    s_ = []
    p_ = []
    n_ = []
    i_c = []
    s_c = []
    for row in batch_triplet_pairs:
        sketch_name = row[0]
        sketch_category = row[1]
        positive_icon_name = row[2]
        positive_icon_category = row[3]
        negative_icon_name = row[4]
        negative_icon_category = row[5]

        s_.append(sketch_path + sketch_category + "/" + sketch_name)
        p_.append(icon_path + positive_icon_category + "/" + positive_icon_name)
        n_.append(icon_path + negative_icon_category + "/" + negative_icon_name)
        i_c.append(icon_categories_dic[positive_icon_category])
        s_c.append(sketch_categories_dic[sketch_category])
    
    sketches = np.array([load_img(i) for i in s_])
    positive_icons = np.array([load_img(i) for i in p_])
    negative_icons = np.array([load_img(i) for i in n_])
    icon_targets = np.array(i_c)
    sketch_targets = np.array(s_c)

    return sketches, positive_icons, negative_icons, icon_targets, sketch_targets

def get_batch_sketches(batch_sketches):
    s_ = []
    for sketch, category in batch_sketches:
        sketch_path_file = sketch_path + category + "/" + sketch
        s_.append(sketch_path_file)
    
    sketches = np.array([load_img(i) for i in s_])
    return sketches

def get_batch_icons(batch_icons):
    i_ = []
    for icon, category in batch_icons:
        icon_path_file = icon_path + category + "/" + icon
        i_.append(icon_path_file)
    
    icons = np.array([load_img(i) for i in i_])
    return icons

def get_batch_icons_targets(icon_name_category_batch, icon_dictionary, icon_categories_dic):
    i_ = []
    t_ = []
    for icon_name, icon_category in icon_name_category_batch:
        i_.append(icon_dictionary[icon_name])
        t_.append(icon_categories_dic[icon_category])
    icons = np.array([i for i in i_])
    targets = np.array([i for i in t_])
    return icons, targets

def get_batch_icons_targets_low_ram(icon_name_category_batch, icon_categories_dic):
    i_ = []
    t_ = []
    for icon_name, icon_category in icon_name_category_batch:
        i_.append(icon_path + icon_category + "/" + icon_name)
        t_.append(icon_categories_dic[icon_category])
    icons = np.array([load_img(i) for i in i_])
    targets = np.array([i for i in t_])
    return icons, targets

def get_batch_sketches_targets(sketch_name_category_batch, sketch_dictionary, sketch_categories_dic):
    s_ = []
    t_ = []
    for sketch_name, sketch_category in sketch_name_category_batch:
        s_.append(sketch_dictionary[sketch_name])
        t_.append(sketch_categories_dic[sketch_category])
    sketches = np.array([i for i in s_])
    targets = np.array([i for i in t_])
    return sketches, targets

def get_batch_sketches_targets_low_ram(sketch_name_category_batch, sketch_categories_dic):
    s_ = []
    t_ = []
    for sketch_name, sketch_category in sketch_name_category_batch:
        s_.append(sketch_path + sketch_category + "/" + sketch_name)
        t_.append(sketch_categories_dic[sketch_category])
    sketches = np.array([load_img(i) for i in s_])
    targets = np.array([i for i in t_])
    return sketches, targets