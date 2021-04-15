import cv2
import os
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

# this is the folder name of the dataset
dataset_path = "Sketch-Icon-Dataset/"
# this is the folder name of the icons
icon_path = os.path.join(dataset_path, 'icon/')
# this is the folder name of the sketches
sketch_path = os.path.join(dataset_path, 'sketch/')

input_shape = (224, 224)

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

def triplet_path_generator(triplet_array):
    sketches = [str(sketch_path + triplet[1] + "/" + triplet[0]) for triplet in triplet_array]
    positive_icons = [str(icon_path + triplet[3] + "/" + triplet[2]) for triplet in triplet_array]
    negative_icons = [str(icon_path + triplet[5] + "/" + triplet[4]) for triplet in triplet_array]
    return sketches, positive_icons, negative_icons

def path_generator(sketches, icons):
    sketches_paths = [str(sketch_path + sketch[1] + "/" + sketch[0]) for sketch in sketches]
    icons_paths = [str(icon_path + icon[1] + "/" + icon[0]) for icon in icons]
    return sketches_paths, icons_paths

def icon_path_class_generator(icons, icon_categories_dic):
  icons_paths = []
  icons_class = []
  for icon in icons:
    icons_paths.append(str(icon_path + icon[1] + "/" + icon[0]))
    icons_class.append(icon_categories_dic[icon[1]])
  return icons_paths, icons_class

def sketch_path_class_generator(sketches, sketch_categories_dic):
  sketches_paths = []
  sketches_class = []
  for sketch in sketches:
    sketches_paths.append(str(sketch_path + sketch[1] + "/" + sketch[0]))
    sketches_class.append(sketch_categories_dic[sketch[1]])
  return sketches_paths, sketches_class

def preprocess_image(filename):
    """
    Load the specified file, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, input_shape)
    return image

def preprocess_triplets(sketch, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(sketch),
        preprocess_image(positive),
        preprocess_image(negative),
    )

def preprocess_images(images):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return preprocess_image(images)
    

def generate_Train_dataset(positive_pairs_Train, icon_name_category, possible_negative_pairs_Train, BATCH_SIZE):
    """[summary]

    Args:
        positive_pairs_Train np.array(sketch name, sketch category, icon name, icon category): Numpy array of positive pairs
        (the sketch and the corresponding icon for this sketch)
        icon_name_category np.array(icon_name, icon_category): Array with icon name and icon category as columns
        possible_negative_pairs dic(sketch name: list((icon name, icon category))): Dictionary of sketch names as keys and the 
        negative pairs on the same category
        BATCH_SIZE (int): Batch size

        This function creates the train dataset as a tensorflow dataset. With this way and some optimizations as prefetching, caching etc.
        the reading of the data is faster.
    Returns:
        train_dataset (tf.data.Dataset): Tensorflow dataset
    """
    triplet_pairs_Train = triplets_generator(positive_pairs_Train, icon_name_category, possible_negative_pairs_Train)
    triplet_pairs_Train = shuffle(triplet_pairs_Train)

    sketches_Train, positive_icons_Train, negative_icons_Train = triplet_path_generator(triplet_pairs_Train)
    sketches_dataset = tf.data.Dataset.from_tensor_slices(sketches_Train)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_icons_Train)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_icons_Train)

    train_dataset = tf.data.Dataset.zip((sketches_dataset, positive_dataset, negative_dataset))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.map(preprocess_triplets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset

def generate_Test_sets(sketches_Test, icons_Test, BATCH_SIZE):
    """[summary]

    Args:
        sketches_Test np.array(sketch name, sketch category): Numpy array
        icons_Test np.array(icon name, icon category): Numpy array
        BATCH_SIZE (int): Batch size

    Returns:
        sketches_dataset(tf.data.Dataset): Tensorflow dataset
        icons_dataset (tf.data.Dataset): Tensorflow dataset
    """
    sketches_paths_Test, icons_paths_Test = path_generator(sketches_Test, icons_Test)

    sketches_dataset = tf.data.Dataset.from_tensor_slices(sketches_paths_Test)
    sketches_dataset = sketches_dataset.cache()
    sketches_dataset = sketches_dataset.map(preprocess_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    sketches_dataset = sketches_dataset.batch(BATCH_SIZE, drop_remainder=False)
    sketches_dataset = sketches_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    icons_dataset = tf.data.Dataset.from_tensor_slices(icons_paths_Test)
    icons_dataset = icons_dataset.cache()
    icons_dataset = icons_dataset.map(preprocess_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    icons_dataset = icons_dataset.batch(BATCH_SIZE, drop_remainder=False)
    icons_dataset = icons_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return sketches_dataset, icons_dataset

def generate_classification_icon_dataset(name_category_array, icon_categories_dic, BATCH_SIZE):

    icons_paths, icons_class = icon_path_class_generator(name_category_array, icon_categories_dic)
    class_dataset = tf.data.Dataset.from_tensor_slices(icons_class)

    icons_dataset = tf.data.Dataset.from_tensor_slices(icons_paths)
    icons_dataset = icons_dataset.map(preprocess_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = tf.data.Dataset.zip((icons_dataset, class_dataset))
    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def generate_classification_sketch_dataset(name_category_array, sketch_categories_dic, BATCH_SIZE):

    sketches_paths, sketches_class = sketch_path_class_generator(name_category_array, sketch_categories_dic)
    class_dataset = tf.data.Dataset.from_tensor_slices(sketches_class)

    sketches_dataset = tf.data.Dataset.from_tensor_slices(sketches_paths)
    sketches_dataset = sketches_dataset.map(preprocess_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = tf.data.Dataset.zip((sketches_dataset, class_dataset))
    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset