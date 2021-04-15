import os

folder_path = ""

def create_directory(path):
    try:
        os.mkdir(folder_path + path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path + path)
    else:
        print ("Successfully created the directory %s " % folder_path + path)

def write_triplet_stats_in_file(path, epoch, loss, acc1, acc10):
    f=open(folder_path + path + "stats.txt", "a+")
    f.write("{:d},{:.5f},{:.5f},{:.5f}\n".format(epoch,loss,acc1,acc10))
    f.close()

def write_hyperparameters_in_file(path, learning_rate, batch_size, margin):
    f=open(folder_path + path + "hyperparameters.txt", "a+")
    f.write("Learning rate: " + str(learning_rate) + "\n")
    f.write("Batch size: " + str(batch_size) + "\n")
    f.write("Margin: " + str(margin) + "\n")
    f.close()

def write_classification_stats_in_file(path, epoch, loss, acc):
    f=open(folder_path + path + "stats.txt", "a+")
    f.write("{:d},{:.5f},{:.5f}\n".format(epoch,loss,acc))
    f.close()

def write_classification_hyperparameters_in_file(path, learning_rate, batch_size):
    f=open(folder_path + path + "hyperparameters.txt", "a+")
    f.write("Learning rate: " + str(learning_rate) + "\n")
    f.write("Batch size: " + str(batch_size) + "\n")
    f.close()

def save_weights(model, path):
    model.save_weights(folder_path + path)