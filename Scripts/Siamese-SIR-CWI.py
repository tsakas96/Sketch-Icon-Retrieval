import tensorflow as tf
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from datetime import datetime
import utils_siamese1
import model1
import results_generator
importlib.reload(utils_siamese1)
importlib.reload(model1)
importlib.reload(results_generator)
from utils_siamese1 import *
from model1 import *
from results_generator import *
tf.test.is_built_with_cuda()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

icon_name_category, sketch_name_category = get_icons_and_sketches()
icon_dictionary, sketch_dictionary = load_icons_sketches_dic(icon_name_category, sketch_name_category)

def shuffle_in_unison(a, b):
    n_elem = a.shape[0]
    indeces = np.random.permutation(n_elem)
    return a[indeces], b[indeces]

# Split the dataset into train and test with randomness
# or load the train and test dataset which is created as numpy arrays.
# The purpose of the predefined datasets is to have the same datasets for training
# the classification of sketches and the sketch-icon retrieval, so the test set will be the same,
# and unknown for both trainings
random_generation = False
if random_generation:
    positive_pairs, positive_labels = positive_pairs_generator(icon_name_category, sketch_name_category)
    positive_pairs = shuffle(positive_pairs)

    len_data = len(positive_pairs)
    p_train=0.9
    p_test=0.1
    num_train = int(np.ceil(len_data*p_train))
    num_test = int(np.floor(len_data*p_test))

    positive_pairs_Train, positive_labels_Train = positive_pairs[:num_train], positive_labels[:num_train]
    possible_negative_pairs_Train = possible_negative_pairs_generator(positive_pairs_Train, icon_name_category)
    negative_pairs_Train, negative_labels_Train = negative_pairs_generator(positive_pairs_Train, possible_negative_pairs_Train)
    pairs_Train = np.concatenate((positive_pairs_Train, negative_pairs_Train), axis=0)
    labels_Train = np.concatenate((positive_labels_Train, negative_labels_Train), axis=0)
    pairs_Train, labels_Train = shuffle_in_unison(pairs_Train, labels_Train)

    positive_pairs_Test = positive_pairs[-num_test:]
    sketches_Test = positive_pairs_Test[:, [0, 1]]

    icons_Test = positive_pairs_Test[:, [2, 3]]
    # create the icons test dataset by removing the dublicates of the array
    _, unique_indices = np.unique(icons_Test[:,0], return_index=True)
    unique_icons_Test = icons_Test[unique_indices]

    print(f'We have {len(pairs_Train)} samples in the training set.')
    print(f'We have {len(sketches_Test)} sketches in the test set.')
    print(f'We have {len(unique_icons_Test)} unique icons in the test set.')
else:
    sketch_name_category_Train = load_train_set()
    sketch_name_category_Test = load_test_set()

    positive_pairs_Train, positive_labels_Train =  positive_pairs_generator(icon_name_category, sketch_name_category_Train)
    possible_negative_pairs_Train = possible_negative_pairs_generator(positive_pairs_Train, icon_name_category)
    negative_pairs_Train, negative_labels_Train = negative_pairs_generator(positive_pairs_Train, possible_negative_pairs_Train)
    pairs_Train = np.concatenate((positive_pairs_Train, negative_pairs_Train), axis=0)
    labels_Train = np.concatenate((positive_labels_Train, negative_labels_Train), axis=0)
    pairs_Train, labels_Train = shuffle_in_unison(pairs_Train, labels_Train)

    positive_pairs_Test,_ =  positive_pairs_generator(icon_name_category, sketch_name_category_Test)

    sketches_Test = positive_pairs_Test[:, [0, 1]]

    icons_Test = positive_pairs_Test[:, [2, 3]]
    
    # create the icons test dataset by removing the dublicates of the array
    _, unique_indices = np.unique(icons_Test[:,0], return_index=True)
    unique_icons_Test = icons_Test[unique_indices]

    print(f'We have {len(pairs_Train)} samples in the training set.')
    print(f'We have {len(sketches_Test)} sketches in the test set.')
    print(f'We have {len(unique_icons_Test)} unique icons in the test set.')

algorithm = "Siamese Loss-CWI" # CWI = Classification Weights Initialization
model_name = "MyNet"
create_directory(model_name + "/")
now = datetime.now()
date_time_folder = now.strftime("%d-%m-%Y %H-%M-%S")
train_weights_folder = "Train Weights"
choice = 0


# Creation of folders for storing stats, weights and the hyperparameters of the model
create_directory(model_name + "/" + algorithm)
current_run_path = model_name + "/" + algorithm + "/" + date_time_folder + "/"
create_directory(current_run_path)
train_weights_path = model_name + "/" + algorithm + "/" + date_time_folder + "/" + train_weights_folder
create_directory(train_weights_path)
#=====================================================================================

# Hyperparameter initializations
BATCH_SIZE = 256
num_epochs = 1000
margin = 1
learing_rate = 0.0001

# Store the hyperparameters of the model into a file to remember what I used in a specific run
write_hyperparameters_in_file(current_run_path, learing_rate, BATCH_SIZE, margin)

# Initializations of optimizer, loss and network and set the backend (model variables) to float32
optimizer = tf.keras.optimizers.Adam(learing_rate)
tf.keras.backend.set_floatx('float32')
weights_path_sketch = "MyNet/Sketch Classification/26-03-2021 23-13-04/Train Weights/"
weights_path_icon = "MyNet/Icon Classification/26-03-2021 14-53-18/Train Weights/"
iconClassificationModel = mynet(NUM_CLASSES=66)
iconClassificationModel.load_weights(weights_path_icon + 'iconClassification')
sketchClassificationModel = mynet(NUM_CLASSES=66)
sketchClassificationModel.load_weights(weights_path_sketch + 'sketchClassification')
loss = siamese_loss

# Function which is responsible to calculate the loss and update the grads
def train_step(sketches, icons, labels):
    with tf.GradientTape(persistent=True) as tape:
        sketch_features,_ = sketchClassificationModel(sketches, training = True)
        icon_features,_ = iconClassificationModel(icons, training = True) 
        tape.watch(sketch_features)
        tape.watch(icon_features)
        current_loss = loss(sketch_features, icon_features, labels)
    grads = tape.gradient(current_loss, sketchClassificationModel.trainable_variables)
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, sketchClassificationModel.trainable_variables) if grad is not None)
    grads = tape.gradient(current_loss, iconClassificationModel.trainable_variables)
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, iconClassificationModel.trainable_variables) if grad is not None)
    return current_loss

top_acc = 0
top_acc1 = 0
top_acc10 = 0
for epoch in range(num_epochs):

    # Training loop throught the training dataset
    epoch_loss_avg = tf.keras.metrics.Mean()
    for i in range(0, len(pairs_Train), BATCH_SIZE):
        batch_pairs = pairs_Train[i:i+BATCH_SIZE]
        batch_labels = labels_Train[i:i+BATCH_SIZE]
        sketches, icons = get_batch(batch_pairs, icon_dictionary, sketch_dictionary)
        loss_value = train_step(sketches, icons, batch_labels)
        epoch_loss_avg.update_state(loss_value)
    print("Epoch {:d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))

    if epoch%1==0:
      # Test loop through the test datatet
        acc_1 = 0
        acc_10 = 0

        # extract features for sketches
        sketch_representations = []
        for j in range(0, len(sketches_Test), BATCH_SIZE):
            batch_sketches = sketches_Test[j:j+BATCH_SIZE]
            sketches_array = get_batch_sketches(batch_sketches)
            sketch_repr,_ =  sketchClassificationModel(sketches_array, training = False)
            sketch_representations.append(sketch_repr)
        sketch_representations = np.vstack(sketch_representations)

        # extract features for icons
        icon_representations = []
        for j in range(0, len(unique_icons_Test), BATCH_SIZE):
            batch_icons = unique_icons_Test[j:j+BATCH_SIZE]
            icons_array = get_batch_icons(batch_icons)
            icons_repr,_ =  iconClassificationModel(icons_array, training = False)
            icon_representations.append(icons_repr)
        icon_representations = np.vstack(icon_representations)

        # check using euclidean distance the top 1 and top 10 accuracy
        for k in range(len(sketch_representations)):
            sketch_repr = sketch_representations[k]
            sketch_representations_tile = np.tile(sketch_repr, len(unique_icons_Test)).reshape(len(unique_icons_Test), 64)
            diff = np.sqrt(np.mean((sketch_representations_tile - icon_representations)**2, -1))
            top_k = np.argsort(diff)[:10]
            
            for j in range(len(top_k)):
                index = top_k[j]
                if j == 0 and sketches_Test[k][0].split("_")[0] == unique_icons_Test[index][0].replace(".jpg",""):
                    acc_1 = acc_1 + 1
                    acc_10 = acc_10 + 1
                    break
                elif sketches_Test[k][0].split("_")[0] == unique_icons_Test[index][0].replace(".jpg",""):
                    acc_10 = acc_10 + 1
                    break
        if top_acc < acc_1 + acc_10:
            save_weights(iconClassificationModel, train_weights_path + "/iconSiameseWithCWI")
            save_weights(sketchClassificationModel, train_weights_path + "/sketchSiameseWithCWI")
            top_acc = acc_1 + acc_10
            top_acc1 = acc_1
            top_acc10 = acc_10
        print("Accuracy of top 1: " + str(acc_1/len(sketches_Test)))
        print("Accuracy of top 10: " + str(acc_10/len(sketches_Test)))
        write_triplet_stats_in_file(current_run_path, epoch, epoch_loss_avg.result(), acc_1/len(sketches_Test), acc_10/len(sketches_Test))
    
    # generate different negative pairs and shuffle
    negative_pairs_Train, negative_labels_Train = negative_pairs_generator(positive_pairs_Train, possible_negative_pairs_Train)
    pairs_Train = np.concatenate((positive_pairs_Train, negative_pairs_Train), axis=0)
    labels_Train = np.concatenate((positive_labels_Train, negative_labels_Train), axis=0)
    pairs_Train, labels_Train = shuffle_in_unison(pairs_Train, labels_Train)