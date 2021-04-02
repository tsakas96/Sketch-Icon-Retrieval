import tensorflow as tf
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import utils_triplet2
import model2
import results_generator
importlib.reload(utils_triplet2)
importlib.reload(model2)
importlib.reload(results_generator)
from utils_triplet2 import *
from model2 import *
from results_generator import *
tf.test.is_built_with_cuda()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

icon_name_category, sketch_name_category = get_icons_and_sketches()

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
    len_data = len(sketch_name_category)
    p_train=0.9
    p_test=0.1
    num_train = int(np.ceil(len_data*p_train))
    num_test = int(np.floor(len_data*p_test))

    sketch_name_category = shuffle(sketch_name_category)
    sketches_Train = sketch_name_category[:num_train]
    sketches_Test= sketch_name_category[-num_test:]
else:
    sketches_Train = load_train_set()
    sketches_Train = shuffle(sketches_Train)
    sketches_Test = load_test_set()

print(f'We have {len(sketches_Train)} samples in the training set.')
print(f'We have {len(sketches_Test)} samples in the test set.')

model_name = "GoogleNet"
algorithm_name = "Sketch Classification"
create_directory(model_name + "/")
now = datetime.now()
date_time_folder = now.strftime("%d-%m-%Y %H-%M-%S")
train_weights_folder = "Train Weights"
create_directory(model_name + "/" + algorithm_name)
current_run_path = model_name + "/" + algorithm_name + "/" + date_time_folder + "/"
create_directory(current_run_path)
train_weights_path = model_name + "/" + algorithm_name + "/" + date_time_folder + "/" + train_weights_folder
create_directory(train_weights_path)

BATCH_SIZE = 64
num_epochs = 1000
learing_rate = 0.001
write_classification_hyperparameters_in_file(current_run_path, learing_rate, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(learing_rate)
tf.keras.backend.set_floatx('float32')
sketchClassificationModel = googlenet(len(sketch_categories))
loss = compute_cross_entropy
def train_step(sketches, targets):
    with tf.GradientTape() as tape:
        outputs = sketchClassificationModel(sketches, training = True)
        main_output = outputs[0]
        aux1_output = outputs[1]
        aux2_output = outputs[2]
        tape.watch(main_output)
        tape.watch(aux1_output)
        tape.watch(aux2_output)
        current_loss_main = loss(main_output, targets)
        current_loss_aux1 = loss(aux1_output, targets)
        current_loss_aux2 = loss(aux2_output, targets)
        current_loss = current_loss_main + 0.3 * current_loss_aux1 + 0.3 * current_loss_aux2
    grads = tape.gradient(current_loss, sketchClassificationModel.trainable_variables)
    optimizer.apply_gradients(zip(grads, sketchClassificationModel.trainable_variables))
    return current_loss

top_acc = 0
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    
    for i in range(0, len(sketches_Train), BATCH_SIZE):
        sketch_name_category_batch = sketches_Train[i:i+BATCH_SIZE]
        sketches, targets = get_batch_sketches_targets_low_ram(sketch_name_category_batch, sketch_categories_dic)
        loss_value = train_step(sketches, targets)
        epoch_loss_avg.update_state(loss_value)
    print("Epoch {:d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))
    acc = 0
    for i in range(0, len(sketches_Test), BATCH_SIZE):
        sketch_name_category_batch = sketches_Test[i:i+BATCH_SIZE]
        sketches, targets = get_batch_sketches_targets_low_ram(sketch_name_category_batch, sketch_categories_dic)
        outputs_main = sketchClassificationModel(sketches, training = False)[0]
        correct_prediction = tf.equal(tf.argmax(outputs_main,1),tf.argmax(targets,1))
        for prediction in correct_prediction:
            if prediction:
                acc = acc + 1
    if top_acc<acc:
        top_acc = acc
        save_weights(sketchClassificationModel, train_weights_path + "/sketchClassification")
    print("Test accuracy is: " + str(acc/len(sketches_Test)))
    write_classification_stats_in_file(current_run_path, epoch, epoch_loss_avg.result(), acc/len(sketches_Test))
    sketches_Train = shuffle(sketches_Train)
print("The best accuracy is: " + str(top_acc/len(sketches_Test)))
