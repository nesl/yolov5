import os

training_size = 0.7

frame_path = "frames/"
labels_path = "labels/"

training_dir = "train/"
validation_dir = "val/"

images_dir = "images/"
labels_dir = "labels/"

frame_files = os.listdir(frame_path)


num_training_files = round(len(frame_files)*training_size)

cwd = os.getcwd() + '/'

for n in range(len(frame_files)):

    if n >= num_training_files:
        partition_dir = validation_dir
    else:
        partition_dir = training_dir

    os.symlink(cwd + frame_path+frame_files[n], partition_dir+images_dir+frame_files[n])
    
    label_file = frame_files[n][:-4] + ".txt"
    os.symlink(cwd + labels_path+label_file, partition_dir+labels_dir+label_file)
