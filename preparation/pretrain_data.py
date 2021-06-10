import numpy as np
from PIL import Image

# *** Load the pretrain data from the nonhsa_modmirbase dataset ***

# import the data from the npz files
images_object = np.load("data_sources/nonhsa_modmirbase_images.npz")
images = images_object.f.arr_0

labels_object = np.load("data_sources/nonhsa_modmirbase_labels.npz")
labels = labels_object.f.arr_0

names_object = np.load("data_sources/nonhsa_modmirbase_names.npz")
names = list(names_object.f.arr_0)

# save images in folder with subfolders that have image label as name
i = 0
for image, label, name in zip(images, labels, names):
    image = np.asarray(image, dtype=np.uint8)
    img = Image.fromarray(image, 'RGB')
    img_name = str(name)[2:-1] + '.png'
    if img_name.startswith('>'):
        img_name = img_name[1::]
    img.save(f'datasets/nonhsa_modmirbase/{label[0]}/{img_name}')
    i += 1


# *** Load the train data from the modhsa dataset ***
images_object_test = np.load("data_sources/modhsa_train_images.npz")
images_test = images_object_test.f.arr_0

labels_object_test = np.load("data_sources/modhsa_train_labels.npz")
labels_test = labels_object_test.f.arr_0

names_object_test = np.load("data_sources/modhsa_train_names.npz")
names_test = list(names_object_test.f.arr_0)

# save images in folder with subfolders that have image label as name
i = 0
for image, label, name in zip(images_test, labels_test, names_test):
    print(i)
    image = np.asarray(image, dtype=np.uint8)
    img = Image.fromarray(image, 'RGB')
    img_name = str(name)[2:-1] + '.png'
    if img_name.startswith('>'):
        img_name = img_name[1::]
    print(img_name)
    img.save(f'datasets/modhsa_original/train/{label[0]}/{img_name}')
    i += 1

# *** Load the test data from the modhsa dataset ***
images_object_test = np.load("data_sources/modhsa_test_images.npz")
images_test = images_object_test.f.arr_0

labels_object_test = np.load("data_sources/modhsa_test_labels.npz")
labels_test = labels_object_test.f.arr_0

names_object_test = np.load("data_sources/modhsa_test_names.npz")
names_test = list(names_object_test.f.arr_0)

# save images in folder with subfolders that have image label as name
i = 0
for image, label, name in zip(images_test, labels_test, names_test):
    print(i)
    image = np.asarray(image, dtype=np.uint8)
    img = Image.fromarray(image, 'RGB')
    img_name = str(name)[2:-1] + '.png'
    if img_name.startswith('>'):
        img_name = img_name[1::]
    print(img_name)
    img.save(f'datasets/modhsa_original/test/{label[0]}/{img_name}')
    i += 1
