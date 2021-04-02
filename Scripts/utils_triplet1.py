import cv2
import os
import numpy as np

# this is the folder name of the dataset
dataset_path = "Sketch-Icon-Dataset/"
# this is the folder name of the icons
icon_path = os.path.join(dataset_path, 'icon/')
# this is the folder name of the sketches
sketch_path = os.path.join(dataset_path, 'sketch/')


def load_img(path):
    """
    path: the full path of the picture
    Load the image using cv2, store the pixels devided by 255
    into img and resize the image to (100,100). This input size is the 
    size for mynet model
    """
    img = cv2.imread(path)/255
    return cv2.resize(img, (100,100))

def load_train_set():
    """
    Load a numpy array created by GenerateTrainAndTestSet.ipynb file
    The array has 2 columns (name, category) and each row is a train sketch example
    This arrays are created, so the classification training and the triplet training will have the same
    training and test examples. I use 90% of the sketches as training examples.
    """
    return np.load(dataset_path + "train_set.npy")

def load_test_set():
    """
    Load a numpy array created by GenerateTrainAndTestSet.ipynb file
    The array has 2 columns (name, category) and each row is a test sketch example
    This arrays are created, so the classification training and the triplet training will have the same
    training and test examples. I use 10% of the sketches as test examples.
    """
    return np.load(dataset_path + "test_set.npy")

def get_icons_and_sketches():
    """[summary]
    icon_name: list of the icon names in all the category folders
    icon_category: list of the icon categories. this list is parallel with icon_name
    as a result for each icon I store its category
    sketch_name: list of the sketch names in all the category folders
    sketch_category: list of the sketch categories. this list is parallel with sketch_name
    as a result for each sketch I store its category

    Returns:
    icon_name_category: np array with two columns (icon name, icon category)
    sketch_name_category: np array with two columns (sketch name, sketch category)
    """
    icon_name = []
    icon_category = []
    sketch_name = []
    sketch_category = []

    for category in os.listdir(icon_path):
        category_path_icon = os.path.join(icon_path, category)
        for icon in os.listdir(category_path_icon):
            icon_name.append(icon)
            icon_category.append(category)
    for category in os.listdir(sketch_path):
        category_path_sketch = os.path.join(sketch_path, category)
        for sketch in os.listdir(category_path_sketch):
            sketch_name.append(sketch)
            sketch_category.append(category)
    
    icon_name_category =  np.column_stack((icon_name, icon_category))
    sketch_name_category =  np.column_stack((sketch_name, sketch_category))
    
    return icon_name_category, sketch_name_category

def load_icons_sketches_dic(icon_name_category, sketch_name_category):
    """[summary]

    Args:
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns
        sketch_name_category np.array(sketch_name, sketch_category): Array with sketch name and sketch category as columns

    Returns:
        icon_dictionary dic(icon_name: np.array((100,100,3))): Dictionary with icon name as key and a numpy array of pixels as value
        sketch_dictionary dic(sketch_name: np.array((100,100,3))): Dictionary with sketch name as key and a numpy array of pixels as value
    """
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

# create sketch to image specific positive pairs
def positive_pairs_generator(icon_name_category, sketch_name_category):
    """[summary]

    Args:
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns
        sketch_name_category np.array(sketch_name, sketch_category): Array with sketch name and sketch category as columns

    Returns:
        positive_pairs_array np.array(sketch name, sketch category, icon name, icon category): Numpy array of positive pairs
        (the sketch and the corresponding icon for this sketch)
    """
    positive_pairs_list = []
    for sketch, sketch_category in sketch_name_category:
        for icon, icon_category in icon_name_category:
            if sketch_category == icon_category and icon.replace(".jpg","") == sketch.split("_")[0]:
                positive_pairs_list.append((sketch, sketch_category, icon, icon_category))
                break

    positive_pairs_array = np.array(positive_pairs_list)
    return positive_pairs_array

def possible_negative_pairs_generator(positive_pairs_array, icon_name_category):
    """[summary]

    Args:
        positive_pairs_array  np.array(sketch name, sketch category, icon name, icon category): Numpy array of positive pairs
        (the sketch and the corresponding icon for this sketch)
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns

    Returns:
        possible_negative_pairs dic(sketch name: list((icon name, icon category))): Dictionary of sketch names as keys and the 
        negative pairs on the same category
    
    This function finds all the possible negative pairs (negative icons) of the sketches IN THE SAME CATEGORY!
    """
    possible_negative_pairs = {}
    for sketch, sketch_category,_,_ in positive_pairs_array:
        negative_icon_category_list = []
        for icon, category in icon_name_category:
            if category == sketch_category and icon.replace(".jpg","") != sketch.split("_")[0]:
                negative_icon_category_list.append((icon, category))
        possible_negative_pairs[sketch] = negative_icon_category_list
    return possible_negative_pairs

def triplets_generator(positive_pairs_array, icon_name_category, possible_negative_pairs):
    """[summary]

    Args:
        positive_pairs_array np.array(sketch name, sketch category, icon name, icon category): Numpy array of positive pairs
        (the sketch and the corresponding icon for this sketch)
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns
        possible_negative_pairs dic(sketch name: list((icon name, icon category))): Dictionary of sketch names as keys and the 
        negative pairs on the same category

    Returns:
        triplet_array: np.array(sketch name, sketch category, positive icon name, positive icon category,
         negative icon name, negative icon category): Numpy array of triplets (sketch, positive icon, negative icon)
    """
    negative_icon_category_list = []
    for sketch,_,_,_ in positive_pairs_array:
        temp_list = possible_negative_pairs[sketch]
        random_index = int(np.random.randint(0, len(temp_list)))
        negative_icon_category = temp_list[random_index]
        negative_icon_category_list.append(negative_icon_category)
    negative_icon_category_array = np.array(negative_icon_category_list)
    triplet_array = np.append(positive_pairs_array, negative_icon_category_array, axis=1)
    return triplet_array

# it finds a negative icon from the same category
def find_negative_icon(sketch, sketch_category, icon_name_category):
    """[summary]

    Args:
        sketch (string): the name of sketch
        sketch_category (string): the category of the sketch
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns

    Returns:
        negative_icon_category tuple(icon name, icon category): Tuple for negative icon

        This function returns the name and category of the negative icon, which is selected randomly from the possible negative pairs list 
        of the sketch
    """
    negative_icon_category_list = []
    for icon, category in icon_name_category:
        if category == sketch_category and icon.replace(".jpg","") != sketch.split("_")[0]:
            negative_icon_category_list.append((icon, category))

    random_index = int(np.random.randint(0, len(negative_icon_category_list)))
    negative_icon_category = negative_icon_category_list[random_index]
    return negative_icon_category

def get_batch(batch_triplet_pairs, icon_dictionary, sketch_dictionary):
    """[summary]

    Args:
        batch_triplet_pairs np.array(BATCH_SIZE, 6): Numpy array of a batch of triplets (sketch name, sketch category,
        positive icon name, positive icon category, negative icon name, negative icon category)
        icon_dictionary dic(icon_name: np.array((100,100,3))): Dictionary with icon name as key and a numpy array of pixels as value
        sketch_dictionary dic(sketch_name: np.array((100,100,3))): Dictionary with sketch name as key and a numpy array of pixels as value

    Returns:
        sketches np.array(BATCH_SIZE, 100, 100, 3): a numpy array of the pixels of a batch of sketches
        positive_icons np.array(BATCH_SIZE, 100, 100, 3): a numpy array of the pixels of a batch of positive icons
        negative_icons np.array(BATCH_SIZE, 100, 100, 3): a numpy array of the pixels of a batch of negative icons

        This function create the batches from the icon_dictionary, sketch_dictionary which are dictionaries of the images
        loaded into ram. If these dictionaries dont fit into ram then this function should not be used thus the dictionaries
        should not be created.
    """
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

def get_batch_triplet_classification(batch_triplet_pairs, icon_dictionary, sketch_dictionary, icon_categories_dic, sketch_categories_dic):
    """[summary]

    Args:
        batch_triplet_pairs np.array(BATCH_SIZE, 6): Numpy array of a batch of triplets (sketch name, sketch category,
        positive icon name, positive icon category, negative icon name, negative icon category)
        icon_dictionary dic(icon_name: np.array((100,100,3))): Dictionary with icon name as key and a numpy array of pixels as value
        sketch_dictionary dic(sketch_name: np.array((100,100,3))): Dictionary with sketch name as key and a numpy array of pixels as value
        icon_categories_dic dic(icon name: np.array(CLASS_NUM))): Dictionary of the icon categories as keys and a numpy array of one hot encoded
        representation of the icon categories
        sketch_categories_dic dic(sletch category: np.array(CLASS_NUM))): Dictionary of the sketch categories as keys and a numpy array of one hot encoded
        representation of the sketch categories

    Returns:
        sketches np.array(BATCH_SIZE, 100, 100, 3): a numpy array of the pixels of a batch of sketches
        positive_icons np.array(BATCH_SIZE, 100, 100, 3): a numpy array of the pixels of a batch of positive icons
        negative_icons np.array(BATCH_SIZE, 100, 100, 3): a numpy array of the pixels of a batch of negative icons
        icon_targets np.array(BATCH_SIZE, CLASS_NUM): a numpy array of the targets of the icons
        sketch_targets np.array(BATCH_SIZE, CLASS_NUM): a numpy array of the targets of the sketches

        This function create the batches from the icon_dictionary, sketch_dictionary which are dictionaries of the images
        loaded into ram. If these dictionaries dont fit into ram then this function should not be used thus the dictionaries
        should not be created.
    """
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

def get_batch_sketches(batch_sketches):
    """[summary]

    Args:
        batch_sketches np.array(sketch name, sketch category): numpy array of sketch names and categories as columns

    Returns:
        sketches np.array(BATCH_SIZE, 100, 100, 3): batch of sketches as pixels
    """
    s_ = []
    for sketch, category in batch_sketches:
        sketch_path_file = sketch_path + category + "/" + sketch
        s_.append(sketch_path_file)
    
    sketches = np.array([load_img(i) for i in s_])
    return sketches

def get_batch_icons(batch_icons):
    """[summary]

    Args:
        batch_icons np.array(icon name, icon category): numpy array of icon names and categories as columns

    Returns:
        icons np.array(BATCH_SIZE, 100, 100, 3): batch of icons as pixels
    """
    i_ = []
    for icon, category in batch_icons:
        icon_path_file = icon_path + category + "/" + icon
        i_.append(icon_path_file)
    
    icons = np.array([load_img(i) for i in i_])
    return icons

def get_batch_icons_targets(icon_name_category_batch, icon_dictionary, icon_categories_dic):
    """[summary]

    Args:
        icon_name_category_batch  np.array(icon name, icon category): numpy array of icon names and categories as columns
        icon_dictionary dic(icon_name: np.array((100,100,3))): Dictionary with icon name as key and a numpy array of pixels as value
        icon_categories_dic dic(icon name: np.array(CLASS_NUM))): Dictionary of the icon categories as keys and a numpy array of one hot encoded

    Returns:
        icons np.array(BATCH_SIZE, 100, 100, 3): batch of icons as pixels
        targets np.array(BATCH_SIZE, CLASS_NUM): class targets of the icons

        This function take the images from the dictionary which is stored in ram.
    """
    i_ = []
    t_ = []
    for icon_name, icon_category in icon_name_category_batch:
        i_.append(icon_dictionary[icon_name])
        t_.append(icon_categories_dic[icon_category])
    icons = np.array([i for i in i_])
    targets = np.array([i for i in t_])
    return icons, targets

def get_batch_sketches_targets(sketch_name_category_batch, sketch_dictionary, sketch_categories_dic):
    """[summary]

    Args:
        sketch_name_category_batch  np.array(sketch name, sketch category): numpy array of sketch names and categories as columns
        sketch_dictionary dic(sketch_name: np.array((100,100,3))): Dictionary with sketch name as key and a numpy array of pixels as value
        sketch_categories_dic dic(sletch category: np.array(CLASS_NUM))): Dictionary of the sketch categories as keys and a numpy array of one hot encoded
        representation of the sketch categories

    Returns:
        sketches np.array(BATCH_SIZE, 100, 100, 3): batch of sketches as pixels
        targets np.array(BATCH_SIZE, CLASS_NUM): class targets of the sketches

        This function take the images from the dictionary which is stored in ram.
    """
    s_ = []
    t_ = []
    for sketch_name, sketch_category in sketch_name_category_batch:
        s_.append(sketch_dictionary[sketch_name])
        t_.append(sketch_categories_dic[sketch_category])
    sketches = np.array([i for i in s_])
    targets = np.array([i for i in t_])
    return sketches, targets