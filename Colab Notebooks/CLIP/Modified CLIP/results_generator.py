import os
import torch

folder_path = "/content/gdrive/MyDrive/Thesis/"

def create_directory(path):
    try:
        os.mkdir(folder_path + path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path + path)
    else:
        print ("Successfully created the directory %s " % folder_path + path)

def write_clip_stats_in_file(path, epoch, loss, acc1, acc10):
    f=open(folder_path + path + "stats.txt", "a+")
    f.write("{:d},{:.5f},{:.5f},{:.5f}\n".format(epoch,loss,acc1,acc10))
    f.close()

def save_weights(model, path):
    torch.save(model.state_dict(), folder_path + path)