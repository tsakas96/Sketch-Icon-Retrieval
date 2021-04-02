import tensorflow as tf
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import utils_triplet1
import model1
import results_generator
importlib.reload(utils_triplet1)
importlib.reload(model1)
importlib.reload(results_generator)
from utils_triplet1 import *
from model1 import *
from results_generator import *
tf.test.is_built_with_cuda()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

icon_name_category, sketch_name_category = get_icons_and_sketches()
icon_dictionary, sketch_dictionary = load_icons_sketches_dic(icon_name_category, sketch_name_category)

# both icon categories and sketch categories have the same order when we are reading them
# as a result the categories have the same one hot encoding in both dictionaries
icon_categories =  np.unique(icon_name_category[:, 1], return_index=False)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(icon_categories)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
icon_categories_dic = {}
for i in range(len(icon_categories)):
    icon_categories_dic[icon_categories[i]] = onehot_encoded[i]

sketch_categories =  np.unique(sketch_name_category[:, 1], return_index=False)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(sketch_categories)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
sketch_categories_dic = {}
for i in range(len(sketch_categories)):
    sketch_categories_dic[sketch_categories[i]] = onehot_encoded[i]

random_generation = False
if random_generation:
    positive_pairs = positive_pairs_generator(icon_name_category, sketch_name_category)
    positive_pairs = shuffle(positive_pairs)

    len_data = len(positive_pairs)
    p_train=0.9
    p_test=0.1
    num_train = int(np.ceil(len_data*p_train))
    num_test = int(np.floor(len_data*p_test))

    positive_pairs_Train = positive_pairs[:num_train]
    possible_negative_pairs_Train = possible_negative_pairs_generator(positive_pairs_Train, icon_name_category)
    triplet_pairs_Train = triplets_generator(positive_pairs_Train, icon_name_category, possible_negative_pairs_Train)
    triplet_pairs_Train = shuffle(triplet_pairs_Train)
    positive_pairs_Test = positive_pairs[-num_test:]
    sketches_Test = positive_pairs_Test[:, [0, 1]]

    icons_Test = positive_pairs_Test[:, [2, 3]]
    # create the icons test dataset by removing the dublicates of the array
    _, unique_indices = np.unique(icons_Test[:,0], return_index=True)
    unique_icons_Test = icons_Test[unique_indices]

    print(f'We have {len(triplet_pairs_Train)} samples in the training set.')
    print(f'We have {len(sketches_Test)} sketches in the test set.')
    print(f'We have {len(unique_icons_Test)} unique icons in the test set.')
else:
    sketch_name_category_Train = load_train_set()
    sketch_name_category_Test = load_test_set()

    positive_pairs_Train =  positive_pairs_generator(icon_name_category, sketch_name_category_Train)
    possible_negative_pairs_Train = possible_negative_pairs_generator(positive_pairs_Train, icon_name_category)
    triplet_pairs_Train = triplets_generator(positive_pairs_Train, icon_name_category, possible_negative_pairs_Train)
    triplet_pairs_Train = shuffle(triplet_pairs_Train)

    positive_pairs_Test =  positive_pairs_generator(icon_name_category, sketch_name_category_Test)

    sketches_Test = positive_pairs_Test[:, [0, 1]]

    icons_Test = positive_pairs_Test[:, [2, 3]]
    # create the icons test dataset by removing the dublicates of the array
    _, unique_indices = np.unique(icons_Test[:,0], return_index=True)
    unique_icons_Test = icons_Test[unique_indices]

    print(f'We have {len(triplet_pairs_Train)} samples in the training set.')
    print(f'We have {len(sketches_Test)} sketches in the test set.')
    print(f'We have {len(unique_icons_Test)} unique icons in the test set.')

algorithmName = "Triplet-Classification Loss-CWI" # CWI = Classification Weights Initialization
model_name = "MyNet"
create_directory(model_name + "/")
now = datetime.now()
date_time_folder = now.strftime("%d-%m-%Y %H-%M-%S")
train_weights_folder = "Train Weights"

create_directory(model_name + "/" + algorithmName)
current_run_path = model_name + "/" + algorithmName + "/" + date_time_folder + "/"
create_directory(current_run_path)
train_weights_path = model_name + "/" + algorithmName + "/" + date_time_folder + "/" + train_weights_folder
create_directory(train_weights_path)
BATCH_SIZE = 128
num_epochs = 1000
margin = 1
learing_rate = 0.00001
write_hyperparameters_in_file(current_run_path, learing_rate, BATCH_SIZE, margin)

optimizer = tf.keras.optimizers.Adam(learing_rate)
tf.keras.backend.set_floatx('float32')
iconClassificationModel = mynet(len(icon_categories))
sketchClassificationModel = mynet(len(sketch_categories))

loss = triplet_loss
loss2 = compute_cross_entropy
def train_step(sketches, positive_icons, negative_icons, targets_icons, targets_sketches, margin):
  with tf.GradientTape(persistent=True) as tape:
      sketch_features,sketch_output = sketchClassificationModel(sketches, training = True)
      icon_positive_features,icon_output = iconClassificationModel(positive_icons, training = True) 
      icon_negative_features,_ = iconClassificationModel(negative_icons, training = True)   
      tape.watch(sketch_features)
      tape.watch(icon_positive_features)
      tape.watch(icon_negative_features)
      tape.watch(sketch_output)
      tape.watch(icon_output)
      current_loss = loss(sketch_features, icon_positive_features, icon_negative_features, margin)
      class_loss_icon = loss2(icon_output, targets_icons)
      class_loss_sketch = loss2(sketch_output, targets_sketches)
  grads = tape.gradient([current_loss, class_loss_sketch], sketchClassificationModel.trainable_variables)
  optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, sketchClassificationModel.trainable_variables) if grad is not None)
  grads = tape.gradient([current_loss, class_loss_icon], iconClassificationModel.trainable_variables)
  optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, iconClassificationModel.trainable_variables) if grad is not None)
  return current_loss + class_loss_sketch

top_acc = 0
top_acc1 = 0
top_acc10 = 0
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    for i in range(0, len(triplet_pairs_Train), BATCH_SIZE):
        batch_triple_pairs = triplet_pairs_Train[i:i+BATCH_SIZE]
        sketches, positive_icons, negative_icons, icon_targets, sketch_targets = get_batch_triplet_classification(batch_triple_pairs, icon_dictionary, sketch_dictionary, icon_categories_dic, sketch_categories_dic)
        loss_value = train_step(sketches, positive_icons, negative_icons, icon_targets, sketch_targets, margin)
        epoch_loss_avg.update_state(loss_value)
    print("Epoch {:d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))
    if epoch%1==0:
        acc_1 = 0
        acc_10 = 0
        sketch_representations = []
        for j in range(0, len(sketches_Test), BATCH_SIZE):
            batch_sketches = sketches_Test[j:j+BATCH_SIZE]
            sketches_array = get_batch_sketches(batch_sketches)
            sketch_repr,_ =  sketchClassificationModel(sketches_array, training = False)
            sketch_representations.append(sketch_repr)
        sketch_representations = np.vstack(sketch_representations)

        icon_representations = []
        for j in range(0, len(unique_icons_Test), BATCH_SIZE):
            batch_icons = unique_icons_Test[j:j+BATCH_SIZE]
            icons_array = get_batch_icons(batch_icons)
            icons_repr,_ =  iconClassificationModel(icons_array, training = False)
            icon_representations.append(icons_repr)
        icon_representations = np.vstack(icon_representations)

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
          save_weights(iconClassificationModel, train_weights_path + "/iconTripletClassification")
          save_weights(sketchClassificationModel, train_weights_path + "/sketchTripletClassification")
          top_acc = acc_1 + acc_10
          top_acc1 = acc_1
          top_acc10 = acc_10
        print("Accuracy of top 1: " + str(acc_1/len(sketches_Test)))
        print("Accuracy of top 10: " + str(acc_10/len(sketches_Test)))
        write_triplet_stats_in_file(current_run_path, epoch, epoch_loss_avg.result(), acc_1/len(sketches_Test), acc_10/len(sketches_Test))
    
    triplet_pairs_Train = triplets_generator(positive_pairs_Train, icon_name_category, possible_negative_pairs_Train)
    triplet_pairs_Train = shuffle(triplet_pairs_Train)
  
print("The best accuracy of top 1: " + str(top_acc1/len(sketches_Test)))
print("The best accuracy of top 10: " + str(top_acc10/len(sketches_Test)))
