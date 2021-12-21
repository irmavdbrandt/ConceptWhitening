import numpy as np
from PIL import Image
import os


def load_from_path(img_path, label_path, names_path, dataset_name, train, test):
    """
    :param img_path: path where images are stored
    :param label_path: path where labels of images are stored
    :param names_path: path where names of images are stored
    :param dataset_name: name that will be used to store the dataset of interest
    :param train: boolean whether a training set is loaded
    :param test: boolean whether a test set is loaded
    """

    # import the data from the npz files
    images_object = np.load(img_path)
    images = images_object.f.arr_0
    labels_object = np.load(label_path)
    labels = labels_object.f.arr_0
    names_object = np.load(names_path)
    names = list(names_object.f.arr_0)

    # create a directory where the images will be stored
    path = os.path.join('preparation/datasets', dataset_name)
    os.mkdir(path)

    # save images in folder with subdirs based on their label and name
    i = 0
    for image, label, name in zip(images, labels, names):
        image = np.asarray(image, dtype=np.uint8)
        img = Image.fromarray(image, 'RGB')
        img_name = str(name)[2:-1] + '.png'
        if img_name.startswith('>'):
            img_name = img_name[1::]
        # in case of a training and test set, create subdirectories storing these sets separately
        if train:
            if not os.path.isdir(os.path.join(path, 'train' + str(label[0]))):
                os.mkdir(os.path.join(path, str(label[0])))
            img.save(f'preparation/datasets/{dataset_name}/train/{str(label[0])}/{img_name}')
        elif test:
            if not os.path.isdir(os.path.join(path, 'test' + str(label[0]))):
                os.mkdir(os.path.join(path, str(label[0])))
            img.save(f'preparation/datasets/{dataset_name}/test/{str(label[0])}/{img_name}')
        else:
            if not os.path.isdir(os.path.join(path, str(label[0]))):
                os.mkdir(os.path.join(path, str(label[0])))
            img.save(f'preparation/datasets/{dataset_name}/{str(label[0])}/{img_name}')
        i += 1


load_from_path("preparation/data_sources/nonhsa_modmirbase_images.npz",
               "preparation/data_sources/nonhsa_modmirbase_labels.npz",
               "preparation/data_sources/nonhsa_modmirbase_names.npz",
               'nonhsa_modmirbase', False, False)

load_from_path("preparation/data_sources/modhsa_train_images.npz",
               "preparation/data_sources/modhsa_train_labels.npz",
               "preparation/data_sources/modhsa_train_names.npz",
               'modhsa_original', False, False)

load_from_path("preparation/data_sources/modhsa_test_images.npz",
               "preparation/data_sources/modhsa_test_labels.npz",
               "preparation/data_sources/modhsa_test_names.npz",
               'modhsa_original', False, False)
