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
    into img and resize the image to (224,224). This input size is the 
    size for googlenet model
    """
    img = cv2.imread(path)/255
    return cv2.resize(img, (224,224))

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

# create sketch to image specific positive pairs
def positive_pairs_generator(icon_name_category, sketch_name_category):
    """[summary]

    Args:
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns
        sketch_name_category np.array(sketch_name, sketch_category): Array with sketch name and sketch category as columns

    Returns:
        positive_pairs_array np.array(sketch name, sketch category, icon name, icon category): Numpy array of positive pairs
        (the sketch and the corresponding icon for this sketch)
        positive_labels_array np.array(ones): Numpy array of ones
    """
    label = 1
    positive_labels_list = []
    positive_pairs_list = []
    for sketch, sketch_category in sketch_name_category:
        for icon, icon_category in icon_name_category:
            if sketch_category == icon_category and icon.replace(".jpg","") == sketch.split("_")[0]:
                positive_pairs_list.append((sketch, sketch_category, icon, icon_category))
                positive_labels_list.append(label)
                break

    positive_pairs_array = np.array(positive_pairs_list)
    positive_labels_array = np.array(positive_labels_list)
    return positive_pairs_array, positive_labels_array

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

def negative_pairs_generator(positive_pairs_Train, possible_negative_pairs):
    """[summary]

    Args:
        possible_negative_pairs dic(sketch name: list((icon name, icon category))): Dictionary of sketch names as keys and the 
        negative pairs on the same category

    Returns:
        negative_pair_array: np.array(sketch name, sketch category, icon name, icon category): Numpy array of negative pairs
        negative_labels_array np.array(zeros): Numpy array of zeros
    """
    label = 0
    negative_labels_list = []
    negative_pair_list = []
    for sketch, sketch_category,_,_ in positive_pairs_Train:
        temp_list = possible_negative_pairs[sketch]
        random_index = int(np.random.randint(0, len(temp_list)))
        negative_icon_category = temp_list[random_index]
        negative_pair_list.append((sketch, sketch_category, negative_icon_category[0], negative_icon_category[1]))
        negative_labels_list.append(label)
    negative_pair_array = np.array(negative_pair_list)
    negative_labels_array = np.array(negative_labels_list)
    return negative_pair_array, negative_labels_array

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

def get_batch_low_ram(batch_pairs):
    """[summary]

    Args:
        batch_pairs np.array(BATCH_SIZE, 4): Numpy array of a batch of triplets (sketch name, sketch category,
        icon name, icon category)

    Returns:
        sketches np.array(BATCH_SIZE, 224, 224, 3): a numpy array of the pixels of a batch of sketches
        icons np.array(BATCH_SIZE, 224, 224, 3): a numpy array of the pixels of a batch of icons

        This function loads the images and resize them while it creates the batch. This is better for ram
        since only one batch will be in memory in training and it doesnt need to have the images preloaded into ram.
        This affects the time, because it reads the photos from disk each time, as a result is slower.
    """
    s_ = []
    i_ = []
    for row in batch_pairs:
        sketch_name = row[0]
        sketch_category = row[1]
        icon_name = row[2]
        icon_category = row[3]

        s_.append(sketch_path + sketch_category + "/" + sketch_name)
        i_.append(icon_path + icon_category + "/" + icon_name)
    
    sketches = np.array([load_img(i) for i in s_])
    icons = np.array([load_img(i) for i in i_])

    return sketches, icons

def get_batch_siamese_classification(batch_pairs, icon_categories_dic, sketch_categories_dic):
    """[summary]

    Args:
        batch_pairs np.array(BATCH_SIZE, 6): Numpy array of a batch of triplets (sketch name, sketch category,
        icon name, icon category)
        icon_categories_dic dic(icon name: np.array(CLASS_NUM))): Dictionary of the icon categories as keys and a numpy array of one hot encoded
        representation of the icon categories
        sketch_categories_dic dic(sletch category: np.array(CLASS_NUM))): Dictionary of the sketch categories as keys and a numpy array of one hot encoded
        representation of the sketch categories

    Returns:
        sketches np.array(BATCH_SIZE, 224, 224, 3): a numpy array of the pixels of a batch of sketches
        icons np.array(BATCH_SIZE, 224, 224, 3): a numpy array of the pixels of a batch of positive icons
        icon_targets np.array(BATCH_SIZE, CLASS_NUM): a numpy array of the targets of the icons
        sketch_targets np.array(BATCH_SIZE, CLASS_NUM): a numpy array of the targets of the sketches

        This function create the batches from the icon_dictionary, sketch_dictionary which are dictionaries of the images
        loaded into ram. If these dictionaries dont fit into ram then this function should not be used thus the dictionaries
        should not be created.
    """
    s_ = []
    i_ = []
    i_c = []
    s_c = []
    for row in batch_pairs:
        sketch_name = row[0]
        sketch_category = row[1]
        icon_name = row[2]
        icon_category = row[3]


        s_.append(sketch_path + sketch_category + "/" + sketch_name)
        i_.append(icon_path + icon_category + "/" + icon_name)
        i_c.append(icon_categories_dic[icon_category])
        s_c.append(sketch_categories_dic[sketch_category])
    
    sketches = np.array([load_img(i) for i in s_])
    icons = np.array([load_img(i) for i in i_])
    icon_targets = np.array(i_c)
    sketch_targets = np.array(s_c)

    return sketches, icons, sketch_targets, icon_targets

def get_batch_sketches(batch_sketches):
    """[summary]

    Args:
        batch_sketches np.array(sketch name, sketch category): numpy array of sketch names and categories as columns

    Returns:
        sketches np.array(BATCH_SIZE, 224, 224, 3): batch of sketches as pixels
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
        icons np.array(BATCH_SIZE, 224, 224, 3): batch of icons as pixels
    """
    i_ = []
    for icon, category in batch_icons:
        icon_path_file = icon_path + category + "/" + icon
        i_.append(icon_path_file)
    
    icons = np.array([load_img(i) for i in i_])
    return icons

def get_batch_icons_targets(icon_name_category_batch, icon_categories_dic):
    """[summary]

    Args:
        icon_name_category_batch  np.array(icon name, icon category): numpy array of icon names and categories as columns
        icon_categories_dic dic(icon name: np.array(CLASS_NUM))): Dictionary of the icon categories as keys and a numpy array of one hot encoded

    Returns:
        icons np.array(BATCH_SIZE, 224, 224, 3): batch of icons as pixels
        targets np.array(BATCH_SIZE, CLASS_NUM): class targets of the icons

        
        This function read the images from disk when it creates the bacth. As a result only the batch will be stored in ram
        during training.
    """
    i_ = []
    t_ = []
    for icon_name, icon_category in icon_name_category_batch:
        i_.append(icon_path + icon_category + "/" + icon_name)
        t_.append(icon_categories_dic[icon_category])
    icons = np.array([load_img(i) for i in i_])
    targets = np.array([i for i in t_])
    return icons, targets

def get_batch_sketches_targets(sketch_name_category_batch, sketch_categories_dic):
    """[summary]

    Args:
        sketch_name_category_batch  np.array(sketch name, sketch category): numpy array of sketch names and categories as columns
        sketch_categories_dic dic(sletch category: np.array(CLASS_NUM))): Dictionary of the sketch categories as keys and a numpy array of one hot encoded
        representation of the sketch categories

    Returns:
        sketches np.array(BATCH_SIZE, 224, 224, 3): batch of sketches as pixels
        targets np.array(BATCH_SIZE, CLASS_NUM): class targets of the sketches

        This function read the images from disk when it creates the bacth. As a result only the batch will be stored in ram
        during training.
    """
    s_ = []
    t_ = []
    for sketch_name, sketch_category in sketch_name_category_batch:
        s_.append(sketch_path + sketch_category + "/" + sketch_name)
        t_.append(sketch_categories_dic[sketch_category])
    sketches = np.array([load_img(i) for i in s_])
    targets = np.array([i for i in t_])
    return sketches, targets