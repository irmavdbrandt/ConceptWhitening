import random
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.measure
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from sklearn.metrics import roc_auc_score
from shutil import copyfile
from MODELS.iterative_normalization import iterative_normalization_py
from MODELS.model_resnet import *
from train_places import AverageMeter, accuracy
from sklearn import tree
from scipy.special import softmax
import graphviz
import os
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

matplotlib.use('Agg')


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def plot_concept_top50(args, test_loader, model, whitened_layers, print_other=False, activation_mode='pool_max',
                       evaluate=False, fold='0'):
    """
    :param args: arguments given by user
    :param test_loader: data loader containing the test images
    :param model: model used for training
    :param whitened_layers: whitened layer
    :param print_other: boolean specifying whether other neurons not linked to concepts should be used
    :param activation_mode: which activation mode is used to find the top50 most activated images
    :param evaluate: boolean specifying whether the we are in evaluation mode (testing) or not (validating)
    :param fold: fold number, required when evaluate = True
    :return: this function finds the top 50 images that gets the greatest activations with respect to the concepts.
    Concept activation values are obtained based on iternorm_rotation module outputs.
    Since concept corresponds to channels in the output, we look for the top50 images whose kth channel activations
    are high.
    """
    # switch to evaluate mode
    model.eval()
    # create directory where results will be stored
    if evaluate:
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
    else:
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/validation/' + fold + '/' + args.arch + '/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    # split the string of whitening layers to get the individual layers
    layer_list = whitened_layers.split(',')
    # if we want to visualize the top50 images for a neuron (dimension) not aligned with concepts, we have
    # print_other = True. Here we create a directory for this other dimension
    if print_other:
        folder = dst + '_'.join(layer_list) + '_rot_otherdim/'
    else:
        folder = dst + '_'.join(layer_list) + '_rot_cw/'
    if not os.path.exists(folder):
        os.mkdir(folder)

    model = model.module
    layers = None
    if args.arch == "resnet_cw":
        layers = model.layers
    model = model.model

    outputs = []

    def hook(module, input, output):
        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels,
                                                 module.T, module.eps, module.momentum, module.training)
        size_X = X_hat.size()
        size_R = module.running_rot.size()
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
        X_hat = X_hat.view(*size_X)

        outputs.append(X_hat.cpu().numpy())

    if args.arch == "resnet_cw":
        for layer in layer_list:
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
    if args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
        for layer in layer_list:
            if int(layer) == 1:
                model.bn1.register_forward_hook(hook)
            elif int(layer) == 2:
                model.bn2.register_forward_hook(hook)
            elif int(layer) == 3:
                model.bn3.register_forward_hook(hook)

    begin = 0
    end = len(args.concepts.split(','))
    if print_other:
        begin = print_other
        end = begin + 1
    concepts = args.concepts.split(',')
    with torch.no_grad():
        for k in range(begin, end):
            if k < len(concepts):
                output_path = os.path.join(folder, concepts[k])
            else:
                output_path = os.path.join(folder, 'other_dimension_' + str(k))
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            paths = []
            vals = None
            for i, (input_img, _, path) in enumerate(test_loader):
                paths += list(path)
                input_var = torch.autograd.Variable(input_img)
                outputs = []
                model(input_var)
                val = []
                for output in outputs:
                    if activation_mode == 'mean':
                        val = np.concatenate((val, output.mean((2, 3))[:, k]))
                    elif activation_mode == 'max':
                        val = np.concatenate((val, output.max((2, 3))[:, k]))
                    elif activation_mode == 'pos_mean':
                        pos_bool = (output > 0).astype('int32')
                        act = (output * pos_bool).sum((2, 3)) / (pos_bool.sum((2, 3)) + 0.0001)
                        val = np.concatenate((val, act[:, k]))
                    elif activation_mode == 'pool_max':
                        kernel_size = 2
                        r = output.shape[3] % kernel_size
                        if r == 0:
                            val = np.concatenate((val, skimage.measure.block_reduce(output[:, :, :, :],
                                                                                    (1, 1, kernel_size, kernel_size),
                                                                                    np.max).mean((2, 3))[:, k]))
                        else:
                            val = np.concatenate((val, skimage.measure.block_reduce(output[:, :, :-r, :-r],
                                                                                    (1, 1, kernel_size, kernel_size),
                                                                                    np.max).mean((2, 3))[:, k]))
                    elif activation_mode == 'pool_max_s1':
                        X_test = torch.Tensor(output)
                        maxpool_value, maxpool_indices = F.max_pool2d(X_test, kernel_size=3, stride=1,
                                                                      return_indices=True)
                        X_test_unpool = F.max_unpool2d(maxpool_value, maxpool_indices, kernel_size=3, stride=1)
                        maxpool_bool = X_test == X_test_unpool
                        act = (X_test_unpool.sum((2, 3)) / maxpool_bool.sum((2, 3)).float()).numpy()
                        val = np.concatenate((val, act[:, k]))

                val = val.reshape((len(outputs), -1))
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals, val), 1)

            for i, layer in enumerate(layer_list):
                arr = list(zip(list(vals[i, :]), list(paths)))
                # save the activation values and the accompanying file name in a numpy array for later use
                # (i.e., concept importance)
                # np.save(os.path.join(folder, f'activation_values_concept{concepts[k]}.npy'), arr)
                arr.sort(key=lambda t: t[0], reverse=True)
                for j in range(50):
                    src = arr[j][1]
                    copyfile(src, output_path + '/' + 'layer' + layer + '_' + str(j + 1) + '.jpg')

    return print("Done with searching for the top 50")


def plot_top10(args, plot_cpt=None, layer=None, evaluate=False, fold='0'):
    """
    :param args: arguments given by user
    :param plot_cpt: list of concepts
    :param layer: whitened layer
    :param evaluate: boolean specifying whether the we are in evaluation mode (testing) or not (validating)
    :param fold: fold number, required when evaluate = True
    :return: plot showing the top-10 most activated images along the concept axes. The images are obtained from the
    plot_concept_top50() function.
    """
    if len(layer) > 1:
        whitened_layer = [int(x) for x in layer.split(',')]
        # create a string from the list of whitened layers, remove the [] (first and last characters) and replace the
        # comma's in between the strings with _
        str_layer = str(whitened_layer)[1:-1].replace(', ', '_')
    else:
        str_layer = str(layer)

    if evaluate:
        folder = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/' + str_layer + '_rot_cw/'
    else:
        folder = './plot/' + '_'.join(args.concepts.split(',')) + '/validation/' + fold + '/' + args.arch + '/' \
                 + str_layer + '_rot_cw/'

    # case for when we only have 1 concept, we create a plot with 2 rows of 5 images each
    if len(plot_cpt) == 1:
        fig, axes = plt.subplots(figsize=(30, 3 * len(plot_cpt)), nrows=2, ncols=5)
        c = 0
        cpt = plot_cpt[0]
        for i in range(5):
            axes[c, i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 1) + '.jpg'))
            axes[c, i].set_yticks([])
            axes[c, i].set_xticks([])
        c = 1
        for i in range(5):
            axes[c, i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 5 + 1) + '.jpg'))
            axes[c, i].set_yticks([])
            axes[c, i].set_xticks([])

        for ax, row in zip(axes[:, 0], plot_cpt):
            ax.set_ylabel(row.replace('_', '\n'), rotation=90, size='large', fontsize=20, wrap=False)

        fig.tight_layout()
        plt.show()
        fig.savefig(folder + 'layer' + str(layer) + '.jpg')
    # in case we have more than 1 concept, we create a plot with 10 images in each row
    else:
        fig, axes = plt.subplots(figsize=(30, 3 * len(plot_cpt)), nrows=len(plot_cpt), ncols=10)

        if len(layer) > 1:
            layers = [int(x) for x in layer.split(',')]
            for layer in layers:
                for c, cpt in enumerate(plot_cpt):
                    for i in range(10):
                        axes[c, i].imshow(
                            mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 1) + '.jpg'))
                        axes[c, i].set_yticks([])
                        axes[c, i].set_xticks([])

                for ax, row in zip(axes[:, 0], plot_cpt):
                    ax.set_ylabel(row.replace('_', '\n'), rotation=90, size='large', fontsize=20, wrap=False)

                fig.tight_layout()
                plt.show()
                fig.savefig(folder + 'layer' + str(layer) + '.jpg')
            print('done')

        else:
            for c, cpt in enumerate(plot_cpt):
                for i in range(10):
                    axes[c, i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 1) + '.jpg'))
                    axes[c, i].set_yticks([])
                    axes[c, i].set_xticks([])

            for ax, row in zip(axes[:, 0], plot_cpt):
                ax.set_ylabel(row.replace('_', '\n'), rotation=90, size='large', fontsize=20, wrap=False)

            fig.tight_layout()
            plt.show()
            fig.savefig(folder + 'layer' + str(layer) + '.jpg')


def get_layer_representation(args, test_loader, layer, cpt_idx, model):
    """
    :param args: arguments given by user
    :param test_loader: data loader providing images from the validation/test set
    :param layer: whitened layer
    :param cpt_idx: index of neuron that is aligned with concept of interest
    :param model: trained model
    :return: This method gets the activations of output from iternorm_rotation for images (from val_loader) at
    channel (cpt_idx)
    """
    with torch.no_grad():
        model.eval()
        model = model.module
        if args.arch == "resnet_cw":
            layers = model.layers
        model = model.model
        outputs = []

        def hook(module, input, output):
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm,
                                                     module.num_channels, module.T,
                                                     module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat.cpu().numpy())

        layer = int(layer)
        if args.arch == "resnet_cw":
            if layer <= layers[0]:
                model.layer1[layer - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
        elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
            if layer == 1:
                model.bn1.register_forward_hook(hook)
            elif layer == 2:
                model.bn2.register_forward_hook(hook)
            elif layer == 3:
                model.bn3.register_forward_hook(hook)

        paths = []
        vals = None
        for i, (input_img, _, path) in enumerate(test_loader):
            paths += list(path)
            input_var = torch.autograd.Variable(input_img)
            outputs = []
            model(input_var)
            val = []
            for output in outputs:
                # get the activation value at the index of the concept of interest
                val.append(output.sum((2, 3))[:, cpt_idx])
            val = np.array(val)
            if i == 0:
                vals = val
            else:
                vals = np.concatenate((vals, val), 1)
    del model
    return paths, vals


def get_representation_distance_to_center(args, test_loader, layer, arch='resnet_cw', model=None):
    """
    :param args: arguments given by user
    :param test_loader: data loader providing images from the validation/test set
    :param layer: whitened layer
    :param arch: model architecture
    :param model: trained model
    :return: This method obtains the vector length of a representation (distance to origin)
    Can choose resnet_cw or resnet_original
    """

    with torch.no_grad():
        model.eval()
        model = model.module
        if args.arch == "resnet_cw":
            layers = model.layers
        model = model.model
        outputs = []

        def hook(module, input_tensor, output_tensor):
            if arch == 'resnet_original':
                outputs.append(output_tensor.cpu().numpy())
            else:
                X_hat = iterative_normalization_py.apply(input_tensor[0], module.running_mean, module.running_wm,
                                                         module.num_channels, module.T,
                                                         module.eps, module.momentum, module.training)
                size_X = X_hat.size()
                size_R = module.running_rot.size()
                X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
                X_hat = X_hat.view(*size_X)

                outputs.append(X_hat.cpu().numpy())

        if args.arch == "resnet_cw":
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
        elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
            for whitened_layer in layer:
                if int(whitened_layer) == 1:
                    model.bn1.register_forward_hook(hook)
                elif int(whitened_layer) == 2:
                    model.bn2.register_forward_hook(hook)
                elif int(whitened_layer) == 3:
                    model.bn3.register_forward_hook(hook)

        paths = []
        vals = []
        for i, (input_img, _) in enumerate(test_loader):
            # paths += list(path)
            # if i==500:
            #     break
            input_var = torch.autograd.Variable(input_img)
            outputs = []
            model(input_var)
            for output in outputs:
                # output_shape = output.size() #NCHW
                # reshaped = output.reshape((output_shape[0], output_shape[1], -1))
                # norms = LA.norm(reshaped, axis=2).flatten().tolist()
                output_shape = output.shape  # NCHW
                # reshaped = output.transpose((0,2,3,1)).reshape((-1, output_shape[1]))
                reshaped = output.mean((2, 3))
                norms = LA.norm(reshaped, axis=1).flatten().tolist()
                vals.extend(norms)
    del model
    # return paths, vals
    print(np.mean(vals), np.std(vals))
    # plt.figure()
    # plt.hist(vals, bins=100)
    # plt.xlim(left=0,right=30)
    # plt.savefig(dst+'layer'+str(layer)+'_mean.jpg')

    return vals


def intra_concept_dot_product_vs_inter_concept_dot_product(args, concept_dir, layer, plot_cpt=None, arch='resnet_cw',
                                                           model=None):
    """
    :param args: arguments given by user
    :param concept_dir: directory containing concept images of test/validation set
    :param layer: whitened layer
    :param plot_cpt: list of concepts
    :param arch: model architecture
    :param model: trained model
    :return: this method compares the intra concept group dot product with inter concept group dot product
    """
    # create directory where results will be stored
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/inner_product/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    # create concept loader for concept images of test set
    concept_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(concept_dir, transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # create a sorted lists of all concepts of interest
    concept_list = os.listdir(concept_dir)
    concept_list.sort()
    # MAC-specific constraint
    if '.DS_Store' in concept_list:
        concept_list = list(filter(lambda a: a != '.DS_Store', concept_list))

    # induce evaluation mode
    model.eval()

    model = model.module
    if arch == "resnet_cw":
        layers = model.layers
    model = model.model

    # intiliaze an empty dictionary that will store the concept representations
    representations = {}
    for cpt in plot_cpt:
        representations[cpt] = []

    for c, cpt in enumerate(plot_cpt):
        with torch.no_grad():

            outputs = []

            def hook(module, input_tensor, output_tensor):
                """
                :param module: model layer
                :param input_tensor: input for model layer
                :param output_tensor: output of model layer
                :return: gradients from forward pass on layer of interest (CW layer)
                """
                X_hat = iterative_normalization_py.apply(input_tensor[0], module.running_mean, module.running_wm,
                                                         module.num_channels, module.T,
                                                         module.eps, module.momentum, module.training)
                size_X = X_hat.size()
                size_R = module.running_rot.size()
                X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
                X_hat = X_hat.view(*size_X)

                outputs.append(X_hat.cpu().numpy())

            # apply the hooks to the CW layers, depending on which layer in the model is whitened
            layer = int(layer)
            if args.arch == "resnet_cw":
                if layer <= layers[0]:
                    model.layer1[layer - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1]:
                    model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2]:
                    model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                    model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
            elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
                if layer == 1:
                    model.bn1.register_forward_hook(hook)
                elif layer == 2:
                    model.bn2.register_forward_hook(hook)
                elif layer == 3:
                    model.bn3.register_forward_hook(hook)

            for j, (input, y, path) in enumerate(concept_loader):
                labels = y.cpu().numpy().flatten().astype(np.int32).tolist()
                input_var = torch.autograd.Variable(input)
                outputs = []
                model(input_var)
                for instance_index in range(len(labels)):  # batch size
                    instance_concept_index = labels[instance_index]
                    if concept_list[instance_concept_index] in plot_cpt:
                        representation_concept_index = plot_cpt.index(concept_list[instance_concept_index])
                        output_shape = outputs[0].shape
                        representation_mean = outputs[0][instance_index:instance_index + 1, :, :, :].transpose(
                            (0, 2, 3, 1)).reshape((-1, output_shape[1])).mean(axis=0)  # mean of all pixels of instance
                        # get the cpt_index channel of the output
                        representations[concept_list[instance_concept_index]].append(representation_mean)

    # representation of concepts in matrix form
    dot_product_matrix = np.zeros((len(plot_cpt), len(plot_cpt))).astype('float')
    m_representations = {}
    m_representations_normed = {}
    intra_dot_product_means = {}
    intra_dot_product_means_normed = {}
    for i, concept in enumerate(plot_cpt):
        m_representations[concept] = np.stack(representations[concept], axis=0)  # n * (h*w)
        m_representations_normed[concept] = m_representations[concept] / LA.norm(m_representations[concept], axis=1,
                                                                                 keepdims=True)
        intra_dot_product_means[concept] = np.matmul(m_representations[concept],
                                                     m_representations[concept].transpose()).mean()
        intra_dot_product_means_normed[concept] = np.matmul(m_representations_normed[concept],
                                                            m_representations_normed[concept].transpose()).mean()
        dot_product_matrix[i, i] = 1.0

    inter_dot_product_means = {}
    inter_dot_product_means_normed = {}
    for i in range(len(plot_cpt)):
        for j in range(i + 1, len(plot_cpt)):
            cpt_1 = plot_cpt[i]
            cpt_2 = plot_cpt[j]
            inter_dot_product_means[cpt_1 + '_' + cpt_2] = np.matmul(m_representations[cpt_1],
                                                                     m_representations[cpt_2].transpose()).mean()
            inter_dot_product_means_normed[cpt_1 + '_' + cpt_2] = np.matmul(m_representations_normed[cpt_1],
                                                                            m_representations_normed[
                                                                                cpt_2].transpose()).mean()
            dot_product_matrix[i, j] = abs(inter_dot_product_means_normed[cpt_1 + '_' + cpt_2]) / np.sqrt(
                abs(intra_dot_product_means_normed[cpt_1] * intra_dot_product_means_normed[cpt_2]))
            dot_product_matrix[j, i] = dot_product_matrix[i, j]

    print(intra_dot_product_means, inter_dot_product_means)
    print(intra_dot_product_means_normed, inter_dot_product_means_normed)
    print(dot_product_matrix)
    plt.figure()
    ticklabels = [s.replace('_', ' ') for s in plot_cpt]
    sns.set(font_scale=1)
    ax = sns.heatmap(dot_product_matrix, vmin=0, vmax=1, xticklabels=ticklabels, yticklabels=ticklabels,
                     annot=True)
    ax.figure.tight_layout()
    plt.savefig(dst + arch + '_' + str(layer) + '.jpg')

    return intra_dot_product_means, inter_dot_product_means, intra_dot_product_means_normed, \
            inter_dot_product_means_normed


def plot_trajectory(args, test_loader, whitened_layers, plot_cpt=None, model=None):
    """
    :param args: arguments given by user
    :param test_loader: data loader providing images from the validation/test set
    :param whitened_layers: whitened layers. Note, this function can only be used when multiple layers have been
    whitened!!
    :param plot_cpt: list of concepts
    :param model: trained model
    :return: This function plots the relative activations of a image on two different concepts.
    """
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + '/trajectory_all/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    concepts = args.concepts.split(',')
    cpt_idx = [concepts.index(plot_cpt[0]), concepts.index(plot_cpt[1])]
    vals = None
    paths = None
    layer_list = whitened_layers.split(',')
    for i, layer in enumerate(layer_list):
        if i == 0:
            paths, vals = get_layer_representation(args, test_loader, layer, cpt_idx, model)
        else:
            _, val = get_layer_representation(args, test_loader, layer, cpt_idx, model)
            vals = np.concatenate((vals, val), 0)
    try:
        os.mkdir('{}{}'.format(dst, '_'.join(plot_cpt)))
    except:
        pass

    num_examples = vals.shape[1]
    num_layers = vals.shape[0]
    max_vals = np.amax(vals, axis=1)
    min_vals = np.amin(vals, axis=1)
    vals = vals.transpose((1, 0, 2))
    # vals = (vals - min_vals)/(max_vals-min_vals)
    sort_idx = vals.argsort(0)
    for i in range(num_layers):
        for j in range(2):
            vals[sort_idx[:, i, j], i, j] = np.arange(num_examples) / num_examples
    idx = np.arange(num_examples)
    np.random.shuffle(idx)
    for k, i in enumerate(idx):
        if k == 50:
            break
        fig = plt.figure(figsize=(10, 5))
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        ax2.set_xlabel(plot_cpt[0])
        ax2.set_ylabel(plot_cpt[1])
        plt.scatter(vals[i, :, 0], vals[i, :, 1])
        start_x = vals[i, 0, 0]
        start_y = vals[i, 0, 1]
        for j in range(1, num_layers):
            dx, dy = vals[i, j, 0] - vals[i, j - 1, 0], vals[i, j, 1] - vals[i, j - 1, 1]
            plt.arrow(start_x, start_y, dx, dy, length_includes_head=True, head_width=0.01, head_length=0.02,
                      color='black')
            start_x, start_y = vals[i, j, 0], vals[i, j, 1]
        ax1 = plt.subplot(1, 2, 1)
        ax1.axis('off')
        image = Image.open(paths[i]).resize((100, 25), Image.ANTIALIAS)
        plt.imshow(np.asarray(image).astype(np.int32))
        plt.savefig('{}{}/{}.jpg'.format(dst, '_'.join(plot_cpt), k))


def plot_auc_cw(args, concept_dir, whitened_layers, plot_cpt=None, activation_mode='pool_max'):
    """
    :param args: arguments given by user
    :param concept_dir: directory containing concept images in the test/validation set
    :param whitened_layers: whitened layers
    :param plot_cpt: list of concepts
    :param activation_mode: activation mode chosen to compute the activation score for a concept image along the
     concept axis
    :return: For each layer and each concept, using activation value as the predicted probability of being a certain
    concept, auc score is computed with respect to label
    """
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/auc/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    dst += 'cw/'
    if not os.path.exists(dst):
        os.mkdir(dst)

    concept_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(concept_dir, transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

    layer_list = whitened_layers.split(',')
    # create a sorted lists of all concepts of interest
    concept_list = os.listdir(concept_dir)
    concept_list.sort()
    # MAC-specific constraint
    if '.DS_Store' in concept_list:
        concept_list = list(filter(lambda a: a != '.DS_Store', concept_list))

    aucs = np.zeros((len(plot_cpt), len(layer_list)))
    aucs_err = np.zeros((len(plot_cpt), len(layer_list)))
    for c, cpt in enumerate(plot_cpt):
        cpt_idx_2 = concept_list.index(cpt)
        cpt_idx = plot_cpt.index(cpt)
        for i, layer in enumerate(layer_list):
            with torch.no_grad():
                model = load_deepmir_resnet_cw_v2_model(args, checkpoint_folder="./checkpoints",
                                                        whitened_layer=args.whitened_layers,
                                                        fold_n=args.foldn_bestmodel)
                model.eval()
                model = model.module
                if args.arch == "resnet_cw":
                    layers = model.layers
                model = model.model
                outputs = []

                def hook(module, input, output):
                    X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm,
                                                             module.num_channels, module.T,
                                                             module.eps, module.momentum, module.training)
                    size_X = X_hat.size()
                    size_R = module.running_rot.size()
                    X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                    X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
                    X_hat = X_hat.view(*size_X)

                    outputs.append(X_hat.cpu().numpy())

                layer = int(layer)
                if args.arch == "resnet_cw":
                    if layer <= layers[0]:
                        model.layer1[layer - 1].bn1.register_forward_hook(hook)
                    elif layer <= layers[0] + layers[1]:
                        model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
                    elif layer <= layers[0] + layers[1] + layers[2]:
                        model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
                    elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                        model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
                elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
                    if layer == 1:
                        model.bn1.register_forward_hook(hook)
                    elif layer == 2:
                        model.bn2.register_forward_hook(hook)
                    elif layer == 3:
                        model.bn3.register_forward_hook(hook)

                labels = []
                vals = []
                for j, (input_image, y, path) in enumerate(concept_loader):
                    labels += list(y.cpu().numpy())
                    input_var = torch.autograd.Variable(input_image)
                    outputs = []
                    model(input_var)
                    for output in outputs:
                        if activation_mode == 'mean':
                            vals += list(output.mean((2, 3))[:, cpt_idx])
                        elif activation_mode == 'max':
                            vals += list(output.max((2, 3))[:, cpt_idx])
                        elif activation_mode == 'pos_mean':
                            pos_bool = (output > 0).astype('int32')
                            act = (output * pos_bool).sum((2, 3)) / (pos_bool.sum((2, 3)) + 0.0001)
                            vals += list(act[:, cpt_idx])
                        elif activation_mode == 'pool_max':
                            kernel_size = 2
                            r = output.shape[3] % kernel_size
                            if r == 0:
                                vals += list(
                                    skimage.measure.block_reduce(output[:, :, :, :], (1, 1, kernel_size, kernel_size),
                                                                 np.max).mean((2, 3))[:, cpt_idx])
                            else:
                                vals += list(skimage.measure.block_reduce(output[:, :, :-r, :-r],
                                                                          (1, 1, kernel_size, kernel_size),
                                                                          np.max).mean((2, 3))[:, cpt_idx])
                        elif activation_mode == 'pool_max_s1':
                            X_test = torch.Tensor(output)
                            maxpool_value, maxpool_indices = F.max_pool2d(X_test, kernel_size=3, stride=1,
                                                                          return_indices=True)
                            X_test_unpool = F.max_unpool2d(maxpool_value, maxpool_indices, kernel_size=3,
                                                           stride=1)
                            maxpool_bool = X_test == X_test_unpool
                            act = (X_test_unpool.sum((2, 3)) / maxpool_bool.sum((2, 3)).float()).numpy()
                            vals += list(act[:, cpt_idx])
                del model
            vals = np.array(vals)
            labels = np.array(labels)
            labels = (labels == cpt_idx_2).astype('int32')
            n_samples = labels.shape[0]
            t = 5
            idx = np.array_split(np.random.permutation(n_samples), t)
            auc_t = []
            for j in range(t):
                auc_t.append(roc_auc_score(labels[idx[j]], vals[idx[j]]))
            aucs[c, i] = np.mean(auc_t)
            aucs_err[c, i] = np.std(auc_t)
            print(aucs[c, i])
            print(aucs_err[c, i])

    print('AUC-CW', aucs)
    print('AUC-CW-err', aucs_err)
    np.save(dst + 'aucs_cw.npy', aucs)
    np.save(dst + 'aucs_cw_err.npy', aucs_err)

    concepts = list(args.concepts.split(','))
    x_pos = np.arange(len(concepts))
    aucs_plot = [auc for sublist in aucs for auc in sublist]
    aucs_err_plot = [auc_err for sublist in aucs_err for auc_err in sublist]

    print(aucs_plot)
    print(aucs_err_plot)

    plt.figure()
    plt.bar(x_pos, aucs_plot, yerr=aucs_err_plot, ecolor='black', align='center')
    plt.xticks(x_pos, concepts, rotation=90)
    plt.ylabel('Concept AUC scores')
    plt.tight_layout()
    plt.show()
    plt.savefig(dst + 'concept_aucs' + str(layer) + '.jpg')

    return aucs


def plot_auc(args, plot_cpt=None):
    """
    :param args: arguments given by user
    :param plot_cpt: list of concepts
    :return: barplot showing the auc scores obtained from plot_auc_cw for the concepts in plot_cpt
    """
    # create folder where images will be stored
    folder = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/auc/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    # load the auc scores and their std's saved from the plot_auc_cw function
    aucs_cw = np.load(folder + 'cw/' + 'aucs_cw.npy')
    aucs_cw_err = np.load(folder + 'cw/' + 'aucs_cw_err.npy')

    # create the barplot
    for c, cpt in enumerate(plot_cpt):
        fig = plt.figure(figsize=(5, 5))
        plt.errorbar([2, 3], aucs_cw[c], yerr=aucs_cw_err[c], label='CW')
        plt.xlabel('layer', fontsize=16)
        plt.ylabel('auc', fontsize=16)
        plt.legend(fontsize=13)
        plt.savefig('{}/{}.jpg'.format(folder, cpt))


def plot_concept_representation(args, test_loader, model, whitened_layers, plot_cpt=None, activation_mode='mean'):
    """
    :param args: arguments given by user
    :param test_loader: data loader providing images from the validation/test set
    :param model: trained model
    :param whitened_layers: whitened layers
    :param plot_cpt: list of concepts
    :param activation_mode: activation mode chosen to compute the activation score for a concept image along the
     concept axis
    :return: ....
    """
    with torch.no_grad():
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/representation/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        layer_list = whitened_layers.split(',')
        dst = dst + '_'.join(layer_list) + '/'
        if args.arch == "resnet_cw":
            layers = model.layers

        model = model.model
        if not os.path.exists(dst):
            os.mkdir(dst)
        outputs = []

        def hook(module, input, output):
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm,
                                                     module.num_channels, module.T,
                                                     module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat.cpu().numpy())

        if args.arch == "resnet_cw":
            for layer in layer_list:
                layer = int(layer)
                if layer <= layers[0]:
                    model.layer1[layer - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1]:
                    model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2]:
                    model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                    model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
        elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
            for whitened_layer in layer_list:
                if int(whitened_layer) == 1:
                    model.bn1.register_forward_hook(hook)
                elif int(whitened_layer) == 2:
                    model.bn2.register_forward_hook(hook)
                elif int(whitened_layer) == 3:
                    model.bn3.register_forward_hook(hook)

        concepts = args.concepts.split(',')
        cpt_idx = [concepts.index(plot_cpt[0]), concepts.index(plot_cpt[1])]

        paths = []
        vals = None
        for i, (input_img, _, path) in enumerate(test_loader):
            paths += list(path)
            input_var = torch.autograd.Variable(input_img)
            outputs = []
            model(input_var)
            val = []
            for output in outputs:
                # val.append(output.sum((2,3))[:,cpt_idx])
                if activation_mode == 'mean':
                    val.append(output.mean((2, 3))[:, cpt_idx])
                elif activation_mode == 'max':
                    val.append(output.max((2, 3))[:, cpt_idx])
                elif activation_mode == 'pos_mean':
                    pos_bool = (output > 0).astype('int32')
                    act = (output * pos_bool).sum((2, 3)) / (pos_bool.sum((2, 3)) + 0.0001)
                    val.append(act[:, cpt_idx])
                elif activation_mode == 'pool_max':
                    kernel_size = 3
                    r = output.shape[3] % kernel_size
                    if r == 0:
                        val.append(skimage.measure.block_reduce(output[:, :, :, :], (1, 1, kernel_size, kernel_size),
                                                                np.max).mean((2, 3))[:, cpt_idx])
                    else:
                        val.append(
                            skimage.measure.block_reduce(output[:, :, :-r, :-r], (1, 1, kernel_size, kernel_size),
                                                         np.max).mean((2, 3))[:, cpt_idx])
                elif activation_mode == 'pool_max_s1':
                    X_test = torch.Tensor(output)
                    maxpool_value, maxpool_indices = F.max_pool2d(X_test, kernel_size=3, stride=1,
                                                                  return_indices=True)
                    X_test_unpool = F.max_unpool2d(maxpool_value, maxpool_indices, kernel_size=3, stride=1)
                    maxpool_bool = X_test == X_test_unpool
                    act = (X_test_unpool.sum((2, 3)) / maxpool_bool.sum((2, 3)).float()).numpy()
                    val.append(act[:, cpt_idx])
            val = np.array(val)
            if i == 0:
                vals = val
            else:
                vals = np.concatenate((vals, val), 1)

        for l, layer in enumerate(layer_list):
            n_grid = 20
            img_size = 100
            img_merge = np.zeros((img_size * n_grid, img_size * n_grid, 3))
            idx_merge = -np.ones((n_grid + 1, n_grid + 1))
            cnt = np.zeros((n_grid + 1, n_grid + 1))
            arr = vals[l, :]
            for j in range(len(paths)):
                index = np.floor((arr[j, :] - arr.min(0)) / (arr.max(0) - arr.min(0)) * n_grid).astype(np.int32)
                idx_merge[index[0], index[1]] = j
                cnt[index[0], index[1]] += 1

            for i in range(n_grid):
                for j in range(n_grid):
                    index = idx_merge[i, j].astype(np.int32)
                    if index >= 0:
                        path = paths[index]
                        img = Image.open(path).resize((img_size, img_size), Image.ANTIALIAS).convert("RGB")
                        img_merge[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :] = np.asarray(img)
            plt.figure()
            plt.imshow(img_merge.astype(np.int32))
            plt.xlabel(plot_cpt[1])
            plt.ylabel(plot_cpt[0])
            plt.savefig(dst + 'layer' + layer + '_' + '_'.join(plot_cpt) + '.jpg', dpi=img_size * n_grid // 4)
            plt.figure()
            ax = sns.heatmap(cnt / cnt.sum(), linewidth=0.5)
            plt.xlabel(plot_cpt[1])
            plt.ylabel(plot_cpt[0])
            plt.savefig(dst + 'density_layer' + layer + '_' + '_'.join(plot_cpt) + '.jpg')

    return print("concept representation plot done")


def plot_correlation(args, test_loader, model, layer, evaluate=False, fold='0'):
    """
    :param args: arguments given by user
    :param test_loader: data loader containing the validation/test images
    :param model: trained model
    :param layer: whitened layer
    :param evaluate: boolean specifying whether the we are in evaluation mode (testing) or not (validating)
    :param fold: fold number, required when evaluate = True
    :return: correlation matrix showing the absolute correlation between the axes (neurons) of the whitened layers and
    the mean correlation.
    """
    with torch.no_grad():
        if evaluate:
            dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
        else:
            dst = './plot/' + '_'.join(args.concepts.split(',')) + '/validation/' + fold + '/' + args.arch + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        dst = dst + 'correlation_matrix/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        if args.arch == "resnet_cw":
            layers = model.layers
        model = model.model
        outputs = []

        def hook(module, input, output):
            size_X = output.size()
            X = output.transpose(0, 1).reshape(size_X[1], -1).transpose(0, 1)
            M = X.cpu().numpy()
            outputs.append(M)

        layer = int(layer)
        if args.arch == "resnet_cw":
            if layer <= layers[0]:
                model.layer1[layer - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
        elif args.arch == "deepmir_cw_bn":
            model.features[layer - 1].register_forward_hook(hook)
        elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
            if layer == 1:
                model.relu3.register_forward_hook(hook)
            elif layer == 2:
                model.relu6.register_forward_hook(hook)
            elif layer == 3:
                model.relu9.register_forward_hook(hook)

        for i, (input_tensor, _, path) in enumerate(test_loader):
            input_var = torch.autograd.Variable(input_tensor)
            model(input_var)
            # take only the activations of the first 50 images in the test set (which is not what I want for the test
            # set because they are not shuffled, so the first 50 will only be class 0)
            # if i == 50:
            #     break

        activation = np.vstack(outputs)
        activation -= activation.mean(0)
        activation = activation / activation.std(0)
        Sigma = np.dot(activation.transpose((1, 0)), activation) / activation.shape[0]
        Sigma = np.nan_to_num(Sigma, nan=0)
        plt.figure()
        sns.heatmap(np.abs(Sigma), cmap='hot')
        plt.tight_layout()
        plt.savefig(dst + str(layer) + '.jpg')

        # mean correlation
        mean_corr = np.mean(np.abs(Sigma))
        print('Mean correlation: ', mean_corr)

        return mean_corr


def plot_correlation_BN(args, test_loader, model, layer):
    """
    :param args: arguments given by user
    :param test_loader: data loader containing the validation/test images
    :param model: trained model
    :param layer: batch normalization layer
    :return: correlation matrix showing the absolute correlation between the axes (neurons) of a batch normalization
    layer and the mean correlation.
    """
    with torch.no_grad():
        # create directory where plot will be stored
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        dst = dst + 'correlation_matrix_BN/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        if args.arch == "resnet_cw":
            layers = model.layers
        model = model.model
        outputs = []

        def hook(module, input, output):
            """
            :param module: layer module
            :param input: input of layer
            :param output: output of layer
            :return: output values (i.e. activations) of layer of interest appended to outputs list
            """
            size_X = output.size()
            X = output.transpose(0, 1).reshape(size_X[1], -1).transpose(0, 1)
            M = X.cpu().numpy()
            outputs.append(M)

        # append hooks that save the inputs/outputs to the activation function of the batch normalization layer of
        # interest
        layer = int(layer)
        if args.arch == "resnet_cw":
            if layer <= layers[0]:
                model.layer1[layer - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
        elif args.arch == "deepmir_resnet_bn" or args.arch == "deepmir_resnet_bn_v2":
            if layer == 1:
                model.relu3.register_forward_hook(hook)
            elif layer == 2:
                model.relu6.register_forward_hook(hook)
            elif layer == 3:
                model.relu9.register_forward_hook(hook)

        for i, (input_tensor, _, path) in enumerate(test_loader):
            input_var = torch.autograd.Variable(input_tensor)
            model(input_var)
            # # only go over the first 50 images of the test/validation test
            # if i == 50:
            #     break
        activation = np.vstack(outputs)
        # normalize the activations
        activation -= activation.mean(0)
        activation = activation / activation.std(0)
        # compute the correlation by taking the dot product between different activations
        Sigma = np.dot(activation.transpose((1, 0)), activation) / activation.shape[0]
        # convert all empty and inf values to 0
        Sigma = np.nan_to_num(Sigma, nan=0, posinf=0, neginf=0)
        plt.figure()
        # take the absolute values of the correlations and plot them in a heatmap
        sns.heatmap(np.abs(Sigma), cmap='hot')
        plt.tight_layout()
        plt.savefig(dst + str(layer) + 'BN.jpg')

        # mean correlation
        mean_corr = np.mean(np.abs(Sigma))
        print('Mean correlation: ', mean_corr)

        return mean_corr


def concept_permutation_importance(args, test_loader, layer, criterion, num_concepts=7, model=None):
    """
    :param args: arguments given by user
    :param test_loader: data loader containing the validation/test images
    :param layer: whitened layer
    :param criterion: loss function
    :param num_concepts: number of concepts
    :param model: trained model
    :return: Will compute the concept importance of the top {num_concepts} concepts in the given layer
    """
    permutation_loss = []  # permutation_loss[i] represents the loss obtained when concept i is shuffled
    permutation_accuracy = []

    # create directory where plot and values will be stored
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
    dst = dst + 'importance_classifier/'
    if not os.path.exists(dst):
        os.mkdir(dst)

    with torch.no_grad():
        model.eval()
        model = model.module
        if args.arch == "resnet_cw":
            layers = model.layers
        model = model.model

        loss_avg = AverageMeter()
        accuracy_avg = AverageMeter()
        # first compute the accuracy and loss for the model trained how it is (without shuffling the concept axes)
        for i, (input, target) in enumerate(test_loader):
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)
            loss_avg.update(loss.data, input.size(0))

            [accuracy_batch] = accuracy(output.data, target_var, topk=(1,))
            accuracy_avg.update(accuracy_batch, input.size(0))

        base_loss = loss_avg.avg
        print('base loss', base_loss)
        base_accuracy = accuracy_avg.avg

        # compute the accuracy and loss while shuffling the concept axes (permuting), this will recover the importance
        # of a concept
        for axis_to_permute in range(num_concepts):
            loss_avg = AverageMeter()
            accuracy_avg = AverageMeter()

            def hook(module, input, output):
                """
                :param module: layer module
                :param input: input of layer
                :param output: output of layer
                :return: output values (i.e. activations) of layer of interest appended to outputs list. NOTE: to use
                this function, the batch size should be > 1 as the batch size size is used to define the switching
                of axes!
                """
                batch_size = output.size()[0]
                idx = list(range(batch_size))
                random.shuffle(idx)
                new_output = output.clone()
                new_output[:, axis_to_permute, :, :] = new_output[idx, axis_to_permute, :, :]
                return new_output

            layer = int(layer)
            if args.arch == "resnet_cw":
                if layer <= layers[0]:
                    model.layer1[layer - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1]:
                    model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2]:
                    model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                    model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
            elif args.arch == "deepmir_cw_bn":
                model.features[layer - 1].register_forward_hook(hook)
            elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
                if layer == 1:
                    model.relu3.register_forward_hook(hook)
                elif layer == 2:
                    model.relu6.register_forward_hook(hook)
                elif layer == 3:
                    model.relu9.register_forward_hook(hook)

            for i, (input, target) in enumerate(test_loader):
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                output = model(input_var)
                loss = criterion(output, target_var)
                loss_avg.update(loss.data, input.size(0))
                [accuracy_batch] = accuracy(output.data, target_var, topk=(1,))
                accuracy_avg.update(accuracy_batch, input.size(0))

            print(axis_to_permute, loss_avg.avg)
            permutation_loss.append(loss_avg.avg)
            permutation_accuracy.append(accuracy_avg.avg)

    print('max_i loss', np.argmax(permutation_loss), np.max(permutation_loss))
    print('min_i loss', np.argmin(permutation_loss), np.min(permutation_loss))
    print('max_i acc', np.argmax(permutation_accuracy), np.max(permutation_accuracy))
    print('min_i acc', np.argmin(permutation_accuracy), np.min(permutation_accuracy))
    print(permutation_loss)
    print(base_loss)
    print(permutation_accuracy)
    print(base_accuracy)

    concepts = list(args.concepts.split(','))
    concepts.append('base')
    y_pos = np.arange(len(concepts))
    permutation_accuracy_plot = list(permutation_accuracy)
    permutation_accuracy_plot.append(base_accuracy)

    plt.figure()
    plt.bar(y_pos, permutation_accuracy_plot, align='center')
    plt.xticks(y_pos, concepts, rotation=90)
    plt.ylabel('Permutation Accuracy')
    plt.tight_layout()
    plt.show()
    plt.savefig(dst + 'concept_importance_classifier' + str(layer) + '.jpg')


def concept_gradient_importance(args, test_loader, layer, num_classes=2):
    """
    :param args: arguments given by user
    :param test_loader: data loader containing the validation/test images
    :param layer: whitened layer
    :param num_classes: number of unique classes in the dataset
    :return: concept importance based on the sign of the gradient at the whitened layer of an image. Note: this function
    does an extra backward pass on the model, but requires a batch size of 1. Hence, the batch norm values are adjusted
    based on one instance of the test/validation set.
    """
    model = None
    if args.arch == "deepmir_resnet_cw_v2":
        model = load_deepmir_resnet_cw_v2_model(args, checkpoint_folder="./checkpoints",
                                                whitened_layer=args.whitened_layers, fold_n=args.foldn_bestmodel)
    # create directory to store the plot and values
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    dst = dst + 'importance_targets/'
    if not os.path.exists(dst):
        os.mkdir(dst)

    model = model.module
    if args.arch == "resnet_cw":
        layers = model.layers
    model = model.model

    outputs = []

    def hook(module, input, output):
        """
        :param module: layer module
        :param input: input of layer
        :param output: output of layer
        :return: output values (i.e. activations) of layer of interest appended to outputs list
        """
        outputs.append(input[0])

    # append hooks to the whitening layers
    layer = int(layer)
    if args.arch == "resnet_cw":
        if layer <= layers[0]:
            model.layer1[layer - 1].relu.register_backward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer - layers[0] - 1].relu.register_backward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer - layers[0] - layers[1] - 1].relu.register_backward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].relu.register_backward_hook(hook)
    elif args.arch == "deepmir_cw_bn":
        model.features[layer - 1].register_backward_hook(hook)
    elif args.arch == "deepmir_resnet_cw":
        if layer == 1:
            model.conv3.register_backward_hook(hook)
        elif layer == 2:
            model.conv5.register_backward_hook(hook)
        elif layer == 3:
            model.conv7.register_backward_hook(hook)
    elif args.arch == "deepmir_resnet_cw_v2":
        if layer == 1:
            model.relu3.register_backward_hook(hook)
        elif layer == 2:
            model.relu6.register_backward_hook(hook)
        elif layer == 3:
            model.relu9.register_backward_hook(hook)

    # initialize empty lists where the concept importances will be stored
    class_count = [0] * num_classes
    concept_importance_per_class = [None] * num_classes

    # dir_deriv_signs = []
    # go over all images in the test set, get the gradient for the image just before the CW layer. The value of this
    # gradient at a concept neuron will determine the importance. In case the value is 0, going in the direction
    # of this neuron will decrease the loss, and vice versa. We use 0 instead of a negative value because we have
    # applied ReLU, which will cause all negative values to become 0.
    for i, (input_img, target) in enumerate(test_loader):
        sample_fname, _ = test_loader.dataset.samples[i]
        input_var = torch.autograd.Variable(input_img)
        output = model(input_var)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1).flatten().tolist()[0]
        # increment the class count based on whether the prediction value (class 0 or class 1)
        class_count[prediction_result] += 1
        output[:, prediction_result].backward()
        # in resnet this is outputs[1] due to the structure (first the conv gradients and then the relu ones)
        directional_derivatives = outputs[0].mean(dim=1).flatten()
        directional_derivatives = directional_derivatives.detach().numpy()
        is_positive = (directional_derivatives > 0).astype(np.int64)
        # dir_deriv_signs.append((directional_derivatives > 0).astype(np.int64))
        if concept_importance_per_class[prediction_result] is None:
            concept_importance_per_class[prediction_result] = is_positive
        else:
            concept_importance_per_class[prediction_result] += is_positive
        outputs = []

    for i in range(num_classes):
        if concept_importance_per_class[i] is None:
            print(f'empty concept importance for class {i}')
            print(concept_importance_per_class[i])
        else:
            # divide the importance by the number of instances predicted to be a certain class
            concept_importance_per_class[i] = concept_importance_per_class[i].astype(np.float32)
            concept_importance_per_class[i] /= class_count[i]
            print(concept_importance_per_class[i])
            print(concept_importance_per_class[i].mean())

    np.save(os.path.join(dst, f'concept_importance_targets_{str(layer)}'), concept_importance_per_class)
    # np.save(os.path.join(dst, f'concept_dir_deriv_signs_targets_{str(layer)}'), dir_deriv_signs)

    # create a bar plot showing the concept importances per class
    labels = ['0', '1']
    concepts = args.concepts.split(',')
    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars
    plt.figure()
    for i in range(len(concepts)):
        concept1 = [list(concept_importance_per_class[0])[i], list(concept_importance_per_class[1])[i]]
        plt.bar(x + i * width, concept1, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Concept Importance')
    plt.xlabel('Target Class')
    plt.xticks(x, labels)
    plt.legend(concepts, bbox_to_anchor=(1.05, 0.6))
    plt.tight_layout()
    plt.show()
    plt.savefig(dst + 'concept_importance_targets' + str(layer) + '.jpg')

    # neurons = list(concept_importance_per_class[0])
    # neurons_name = [str(i) for i in range(len(neurons))]
    # width = 0.05  # the width of the bars
    # for i in range(len(neurons)):
    #     concept1 = [list(concept_importance_per_class[0])[i], list(concept_importance_per_class[1])[i]]
    #     plt.bar(x + i * width, concept1, width)
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # plt.ylabel('Concept Importance')
    # plt.xlabel('Target Class')
    # plt.xticks(x, labels)
    # plt.legend(neurons_name, bbox_to_anchor=(1.05, 0.6))
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(dst + 'concept_importance_targets_allneurons' + str(layer) + '.jpg')

    return concept_importance_per_class


def saliency_map_class(args, test_loader, layer, arch='resnet_cw', dataset='isic'):
    """

    """
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/saliency_map/'
    # dst = '/usr/xtmp/zhichen/temp_plots/'
    try:
        os.mkdir(dst)
    except:
        pass
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    # model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model

    for i, (input_img, target) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input_img).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        input_var.requires_grad = True
        output = model(input_var)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1).flatten().tolist()[0]
        output[:, prediction_result].backward()

        save_folder = os.path.join(dst, "class_" + str(prediction_result))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[0].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        grad = input_var.grad[0].permute(1, 2, 0).abs().cpu().numpy()
        grad = (grad / grad.max() * 255).astype(np.int8).max(axis=2)
        plt.imshow(grad, cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(save_folder, str(i) + '.png'))
        plt.close()


def saliency_map_concept(args, val_loader, layer, arch='resnet_cw', dataset='isic', num_concepts=7):
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/saliency_map_concept/'
    # dst = '/usr/xtmp/zhichen/temp_plots/'
    try:
        os.mkdir(dst)
    except:
        pass
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    # model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    layer = int(layer)
    if layer <= layers[0]:
        model.layer1[layer - 1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1]:
        model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2]:
        model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
        model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)

    for j in range(num_concepts):
        save_folder = os.path.join(dst, "concept_" + str(j))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    for i, (input_img, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input_img).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        input_var.requires_grad = True
        for j in range(num_concepts):
            output = model(input_var)
            model.zero_grad()
            outputs[0][0, j, :, :].mean().backward()
            save_folder = os.path.join(dst, "concept_" + str(j))
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(input_img[0].permute(1, 2, 0))
            plt.subplot(1, 2, 2)
            grad = input_var.grad[0].permute(1, 2, 0).abs().cpu().numpy()
            grad = (grad / grad.max() * 255).astype(np.int8).max(axis=2)
            plt.imshow(grad, cmap='hot', interpolation='nearest')
            plt.savefig(os.path.join(save_folder, str(i) + '.png'))
            plt.close()
            outputs = []


def saliency_map_concept_cover(args, layer,  num_concepts=7, model=None):
    """
    :param args: arguments given by user
    :param layer: whitened layer
    :param num_concepts: number of concepts
    :param model: trained model
    :return: images in test set with receptive field of concept is plotted over original image. This shows for all
    test images where the neuron that has been aligned with a concept looks at
    """
    # create directory that will store the plots
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/saliency_map_concept_cover/'
    try:
        os.mkdir(dst)
    except:
        pass

    model.eval()
    model = model.module
    if args.arch == "resnet_cw":
        layers = model.layers
    model = model.model

    outputs = []

    def hook(module, input, output):
        """
        :param module: layer module
        :param input: input of layer
        :param output: output of layer
        :return: output values (i.e. activations) of layer of interest appended to outputs list
        """
        outputs.append(output)

    # append hooks to the whitened layers
    layer = int(layer)
    if args.arch == "resnet_cw":
        if layer <= layers[0]:
            model.layer1[layer - 1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
    elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
        if layer == 1:
            model.bn1.register_forward_hook(hook)
        elif layer == 2:
            model.bn2.register_forward_hook(hook)
        elif layer == 3:
            model.bn3.register_forward_hook(hook)

    # create a folder for each concept that will store the plots
    for j in range(num_concepts):
        if not os.path.exists(os.path.join(dst, "concept_" + str(j))):
            os.mkdir(os.path.join(dst, "concept_" + str(j)))

    # start with an initialization of the size of the concept cover, since we have images of 25x100, we want the cover
    # not to exceed these sizes
    cover_size = 12
    with torch.no_grad():
        # create a data loader that will go over the 50 most activated images per concept, these are the most
        # interesting in my opinion
        base_dir = 'plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/' + \
                   '_'.join(args.whitened_layers) + '_rot_cw/'
        most_activated_img_loader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(base_dir, transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

        for i, (input, _, path) in enumerate(most_activated_img_loader):
            # create a random patch, based on the initialized cover size and the fact that it is an RGB image (3 dim)
            random_patch = torch.tensor(np.random.normal(size=(3, cover_size, cover_size)))
            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            # get the activation values of the input image from the output at the cw layer
            base_activations = F.max_pool2d(outputs[0], kernel_size=2, stride=1)
            base_activations = base_activations[0, :, :, :].clamp(min=0.0).mean(dim=(1, 2))
            outputs = []

            input_size = input.size()
            saliency = np.zeros((num_concepts, input_size[2], input_size[3]))
            counter = np.zeros((input_size[2], input_size[3])) + 0.00001
            for p in range(0, input_size[2] - cover_size + 1, 1):
                print("p={}\n".format(p))
                for q in range(0, input_size[3] - cover_size + 1, 1):
                    new_input = input.clone()
                    new_input[0, :, p:p + cover_size, q:q + cover_size] = random_patch
                    input_var = torch.autograd.Variable(new_input)
                    output = model(input_var)
                    new_activations = F.max_pool2d(outputs[0], kernel_size=2, stride=1)
                    new_activations = new_activations[0, :, :, :].clamp(min=0.0).mean(dim=(1, 2))
                    outputs = []
                    decrease_in_activations = base_activations - new_activations
                    for j in range(num_concepts):
                        saliency[j, p:p + cover_size, q:q + cover_size] += decrease_in_activations[j].cpu().numpy()
                    counter[p:p + cover_size, q:q + cover_size] += 1.0

            saliency = saliency / counter

            u_limit = np.percentile(saliency, 99.99)
            l_limit = np.percentile(saliency, 0.01)
            saliency = saliency.clip(l_limit, u_limit)

            saliency = (saliency - saliency.min()) / saliency.max()
            lower_limit = np.percentile(saliency, 94)
            saliency[saliency < lower_limit] = 0.3
            saliency[saliency >= lower_limit] = 1.0

            input_image = input[0, :, :, :].permute(1, 2, 0).cpu().numpy()

            for j in range(num_concepts):
                concept_index = str(_)[-3]
                base_folder = dst + f"activatedimg_concept{concept_index}"
                if not os.path.exists(base_folder):
                    os.mkdir(base_folder)
                save_folder = os.path.join(base_folder, "concept_" + str(j))
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                image = Image.fromarray(np.uint8(input_image * 255)).convert('RGBA')
                image = np.array(image)
                image[:, :, 3] = (saliency[j, :, :] * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(os.path.join(save_folder, str(i) + '.png'), 'PNG')
                print("saved: " + str(j))


def saliency_map_concept_cover_2(args, test_loader, layer, arch='resnet_cw', dataset='isic', num_concepts=7, model=None):
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/saliency_map_concept_cover/'
    try:
        os.mkdir(dst)
    except:
        pass

    model.eval()
    model = model.module
    if args.arch == "resnet_cw":
        layers = model.layers
    model = model.model

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    layer = int(layer)
    if args.arch == "resnet_cw":
        if layer <= layers[0]:
            model.layer1[layer - 1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hook)
    elif args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
        if int(layer) == 1:
            model.bn1.register_forward_hook(hook)
        elif int(layer) == 2:
            model.bn2.register_forward_hook(hook)
        elif int(layer) == 3:
            model.bn3.register_forward_hook(hook)

    for j in range(num_concepts, num_concepts + 1):
        save_folder = os.path.join(dst, "concept_" + str(j))
        try:
            os.mkdir(save_folder)
        except:
            pass

    cover_size = 6
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            random_patch = torch.tensor(np.random.normal(size=(3, cover_size, cover_size)))
            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            base_activations = F.max_pool2d(outputs[0], kernel_size=2, stride=1)
            base_activations = base_activations[0, :, :, :].clamp(min=0.0).mean(dim=(1, 2))
            outputs = []

            input_size = input.size()
            saliency = np.zeros((1, input_size[2], input_size[3]))
            counter = np.zeros((input_size[2], input_size[3])) + 0.00001
            for p in range(0, input_size[2] - cover_size + 1, 1):
                print("p={}\n".format(p))
                for q in range(0, input_size[3] - cover_size + 1, 1):
                    new_input = input.clone()
                    new_input[0, :, p:p + cover_size, q:q + cover_size] = random_patch
                    input_var = torch.autograd.Variable(new_input)
                    output = model(input_var)
                    new_activations = F.max_pool2d(outputs[0], kernel_size=2, stride=1)
                    new_activations = new_activations[0, :, :, :].clamp(min=0.0).mean(dim=(1, 2))
                    outputs = []
                    decrease_in_activations = base_activations - new_activations
                    for j in range(num_concepts, num_concepts + 1):
                        saliency[0, p:p + cover_size, q:q + cover_size] += decrease_in_activations[j].cpu().numpy()
                    counter[p:p + cover_size, q:q + cover_size] += 1.0
            saliency = saliency / counter

            u_limit = np.percentile(saliency, 99.99)
            l_limit = np.percentile(saliency, 0.01)
            saliency = saliency.clip(l_limit, u_limit)

            saliency = (saliency - saliency.min()) / saliency.max()
            lower_limit = np.percentile(saliency, 94)
            saliency[saliency < lower_limit] = 0.3
            saliency[saliency >= lower_limit] = 1.0

            input_image = input[0, :, :, :].permute(1, 2, 0).cpu().numpy()

            for j in range(num_concepts, num_concepts + 1):
                save_folder = os.path.join(dst, "concept_" + str(j))
                image = Image.fromarray(np.uint8(input_image * 255)).convert('RGBA')
                image = np.array(image)
                image[:, :, 3] = (saliency[0, :, :] * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(os.path.join(save_folder, str(i) + '.png'), 'PNG')
                print("saved: " + str(j))


def get_activations_finalpart(args, test_loader, model, layer):
    """ function that saves all activations of the relu layer after the CW layer and the activations of linear1"""

    with torch.no_grad():
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        dst = dst + 'activations/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        model = model.model

        outputs_relucw = []

        def hook_relu(module, input, output):
            # I think this is the correct one, the one without extra iterative normalization (as this should already
            # have been applied in training)
            outputs_relucw.append(output.cpu().numpy())

        layer = int(layer)
        if args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2" or args.arch == \
                "deepmir_resnet_cw_v3":
            if layer == 1:
                model.relu3.register_forward_hook(hook_relu)
            elif layer == 2:
                model.relu6.register_forward_hook(hook_relu)
            elif layer == 3:
                model.bn3.register_forward_hook(hook_relu)

        outputs_linear1 = []

        def hook_linear1(module, input, output):
            outputs_linear1.append(output.numpy())

        inputs_linear1 = []
        #
        # def hook_linear1_beforerelu(module, input, output):
        #     # print(input[0].shape)
        #     inputs_linear1.append(input.numpy())

        if args.arch == "deepmir_resnet_cw" or args.arch == "deepmir_resnet_cw_v2" or args.arch == \
                "deepmir_resnet_cw_v3":
            model.relu10.register_forward_hook(hook_linear1)
            # model.linear1.register_forward_hook(hook_linear1_beforerelu)

        # change this range to be 0, len(neurons cw layer)
        for k in range(0, 72):
            paths = []
            vals = None
            for i, (input_tensor, _, path) in enumerate(test_loader):
                paths += list(path)
                input_var = torch.autograd.Variable(input_tensor)
                outputs_relucw = []
                model(input_var)
                val = []
                for output in outputs_relucw:
                    if args.act_mode == 'mean':
                        val = np.concatenate((val, output.mean((2, 3)[:, k])))
                    elif args.act_mode == 'max':
                        val = np.concatenate((val, output.max((2, 3)[:, k])))
                    elif args.act_mode == 'pos_mean':
                        pos_bool = (output > 0).astype('int32')
                        act = (output * pos_bool).sum((2, 3)) / (pos_bool.sum((2, 3)) + 0.0001)
                        val = np.concatenate((val, act[:, k]))
                    elif args.act_mode == 'pool_max':
                        kernel_size = 2
                        r = output.shape[3] % kernel_size
                        if r == 0:
                            val = np.concatenate((val, skimage.measure.block_reduce(output[:, :, :, :],
                                                                                    (1, 1, kernel_size, kernel_size),
                                                                                    np.max).mean((2, 3))[:, k]))
                        else:
                            val = np.concatenate((val, skimage.measure.block_reduce(output[:, :, :-r, :-r],
                                                                                    (1, 1, kernel_size, kernel_size),
                                                                                    np.max).mean((2, 3))[:, k]))

                val = val.reshape((len(outputs_relucw), -1))
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals, val), 1)

            # save the activations as numpy arrays
            arr = list(zip(vals[0], list(paths)))
            np.save(os.path.join(dst, f'activations_relu_cwlayer{layer}_neuron{k}'), arr)

        activation_relulinear1 = np.vstack(outputs_linear1)
        # input_act_linear1 = np.vstack(inputs_linear1)

        np.save(os.path.join(dst, f'activations_relulinear1'), activation_relulinear1)
        # np.save(os.path.join(dst, f'input_act_linear1'), input_act_linear1)


def tree_explainer(cpt=None, arch="deepmir_resnet_cw_v2", layer="3", learnable_cpt=None):
    """
    :param cpt:
    :param arch:
    :param layer:
    :param learnable_cpt:
    :return:
    """
    # create a sorted lists of all concepts of interest
    concepts = cpt.split(',')
    # create a string of the concepts by joining them with an underscore (this is also how they are saved)
    concepts_string = '_'.join(concepts)

    # base folder where activations of model of interest are stored
    base_folder = f'./plot/{concepts_string}/{arch}/activations'

    # get the index of the neurons that have learned a concept (high AUC...)
    learnable_cpt_neuron_indices = [concepts.index(concept) for concept in learnable_cpt]

    # initialize an empty list that will store the activations of the test images on the concept neurons
    activations_allconcepts = []
    for neuron_index in learnable_cpt_neuron_indices:
        # create the complete file name where the activations for a certain neuron are stored
        file_name = f'activations_relu_cwlayer{layer}_neuron{neuron_index}.npy'
        # load the activations
        file = np.load(os.path.join(base_folder, file_name))

        # initialize an empty list that will store the concept activation values for 1 concept
        activations_neuron = []
        for i in range(len(file)):
            value = float(file[i][0])  # convert from string to float, the act value is stored first, the path second
            activations_neuron.append(value)
        # add the activations from 1 neuron to the list with all activations
        activations_allconcepts.append(activations_neuron)

    # create bins for the data based on the 25th, 50th, 75th, and 100th percentile of the activation values for each
    # concept
    binned_act_values = []
    for i in range(len(learnable_cpt_neuron_indices)):
        stats_concept = pd.Series(activations_allconcepts[i]).describe()
        concept_25 = stats_concept.T['25%']
        concept_50 = stats_concept.T['50%']
        concept_75 = stats_concept.T['75%']
        concept_100 = stats_concept.T['max']

        bins_concept = [concept_25, concept_50, concept_75, concept_100]
        binned_values_concept = np.digitize(activations_allconcepts[i], bins_concept)
        binned_act_values.append(binned_values_concept)

    # create the feature matrix X for the decision tree containing the binned act values for the concepts of interest
    X = np.array(binned_act_values)
    # transpose the rows and columns in X to match the requirements of the decision tree
    X = np.transpose(np.asarray(X))

    # create the targets y, which are in my case the prediction outcomes
    # the test predictions are in the same order as the test data (test loader model is without shuffling)
    y = np.load(f'./output/{concepts_string}.npy')
    # apply softmax (pytorch Crossentropy() does this and this has not yet been done on the saved predictions)
    y = softmax(y, axis=2)
    y = np.argmax(y, axis=2)
    y = y.flatten()

    # initiate the classifier itself using the X and y matrices
    clf = tree.DecisionTreeClassifier(random_state=2)
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    plt.show()

    # obtain feature names for the tree, which are the concept names with predicted_similarity in front of them
    feature_names = []
    for concept in learnable_cpt:
        feature_name_concept = f"predicted_similarity_{concept}"
        feature_names.append(feature_name_concept)

    # create a nice formatting of the tree
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=np.unique(y).astype(str),
                                    filled=True, rounded=True, proportion=True,
                                    special_characters=True)

    # piece of code to add True/False to each edge, instead of just the first 2
    # first find the lowest node number (last in the string)
    lowest_node = dot_data[-6:-3]  # this is the case when this index is an integer with two numbers
    lowest_node = int(lowest_node)
    # check whether this node is an integer
    if lowest_node in range(0, 100):
        for i in range(0, lowest_node):
            for j in range(i + 1, lowest_node + 1):
                index = dot_data.find(f'{str(i)} -> {str(j)} ;')
                if index == -1:
                    continue
                else:
                    if j == i + 1:
                        if (len(str(i)) == 2) and (len(str(j)) == 2):
                            dot_data = dot_data[:index + 8] + ' [labeldistance=2.5, labelangle=45, headlabel="True"]' + \
                                       dot_data[index + 8:]
                        elif len(str(i)) == 2:
                            dot_data = dot_data[:index + 7] + ' [labeldistance=2.5, labelangle=45, headlabel="True"]' + \
                                       dot_data[index + 7:]
                        elif len(str(j)) == 2:
                            dot_data = dot_data[:index + 7] + ' [labeldistance=2.5, labelangle=45, headlabel="True"]' + \
                                       dot_data[index + 7:]
                        else:
                            dot_data = dot_data[:index + 6] + ' [labeldistance=2.5, labelangle=45, headlabel="True"]' + \
                                       dot_data[index + 6:]
    if lowest_node in range(0, 100):
        for i in range(0, lowest_node):
            for j in range(i + 1, lowest_node + 1):
                index = dot_data.find(f'{str(i)} -> {str(j)} ;')
                if index == -1:
                    continue
                else:
                    if (len(str(i)) == 2) and (len(str(j)) == 2):
                        dot_data = dot_data[:index + 8] + ' [labeldistance=2.5, labelangle=-75, headlabel="False"]' + \
                                   dot_data[index + 8:]
                    elif len(str(i)) == 2:
                        dot_data = dot_data[:index + 7] + ' [labeldistance=2.5, labelangle=-75, headlabel="False"]' + \
                                   dot_data[index + 7:]
                    elif len(str(j)) == 2:
                        dot_data = dot_data[:index + 7] + ' [labeldistance=2.5, labelangle=-75, headlabel="False"]' + \
                                   dot_data[index + 7:]
                    else:
                        dot_data = dot_data[:index + 6] + ' [labeldistance=2.5, labelangle=-75, headlabel="False"]' + \
                                   dot_data[index + 6:]

    # create a graph of the tree
    graph = graphviz.Source(dot_data)
    # render the graph and save it as premirna.pdf
    graph.render("premirna")


def load_resnet_model(args, checkpoint_folder="./checkpoints", whitened_layer=None):
    """
    :param args: arguments given by user
    :param checkpoint_folder: folder where saved weights can be found
    :param whitened_layer: index of layer that needs to be whitened
    :return: resnet model with weights loaded and layer of interest whitened
    """

    prefix_name = args.prefix[:args.prefix.rfind('_')]  # rfind finds the last occurrence of _

    concept_names = '_'.join(args.concepts.split(','))
    if whitened_layer is None:
        raise Exception("whitened_layer argument is required")
    else:
        n_classes = 2
        model = ResidualNetTransfer(n_classes, args, [int(whitened_layer)], arch='resnet18', layers=[2, 2, 2, 2],
                                    model_file=None)
        checkpoint_name = '{}_{}_checkpoint.pth.tar'.format(prefix_name, whitened_layer)

    # parallelize model
    model = torch.nn.DataParallel(model)
    # load weights from trained model saved in checkpoints
    checkpoint_path = os.path.join(checkpoint_folder, concept_names, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_path))

    return model


def load_deepmir_model(args, arch='deepmir_cw_bn', checkpoint_folder="./checkpoints", whitened_layer=None):
    prefix_name = args.prefix[:args.prefix.rfind('_')]

    concept_names = '_'.join(args.concepts.split(','))
    if whitened_layer is None:
        raise Exception("whitened_layer argument is required")
    else:
        model = DeepMirTransfer(args, [int(whitened_layer)], arch='deepmir_cw_bn',
                                model_file=os.path.join(checkpoint_folder, f'{arch}.pth'))
        checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, whitened_layer)

    model = torch.nn.DataParallel(model)

    checkpoint_path = os.path.join(checkpoint_folder, concept_names, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_path))

    return model


def load_deepmir_resnet_model(args, arch='deepmir_resnet_cw', checkpoint_folder="./checkpoints", whitened_layer=None):
    if len(whitened_layer) == 1:
        # rfind finds the last occurrence of _
        prefix_name = args.prefix[:args.prefix.rfind('_')]
    else:
        # find the first occurrence of _ with the first instance in the whitened layers
        prefix_name = args.prefix[:args.prefix.find(f'_{whitened_layer[0]}')]

    concept_names = '_'.join(args.concepts.split(','))
    if whitened_layer is None:
        raise Exception("whitened_layer argument is required")
    elif len(whitened_layer) == 1:
        arch = 'DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint'
        model = DeepMirResNetTransfer(args, [int(whitened_layer)], arch='deepmir_resnet_cw',
                                      model_file=os.path.join(checkpoint_folder,
                                                              f'resnet_premirna_checkpoints/{arch}.pth.tar'))
        checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, whitened_layer)
    else:
        arch = 'DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint'
        whitened_layer = [int(x) for x in args.whitened_layers.split(',')]
        model = DeepMirResNetTransfer(args, whitened_layer, arch='deepmir_resnet_cw',
                                      model_file=os.path.join(checkpoint_folder,
                                                              f'resnet_premirna_checkpoints/{arch}.pth.tar'))
        # create a string from the list of whitened layers, remove the [] (first and last characters) and replace the
        # comma's in between the strings with _
        whitened_layer = str(whitened_layer)[1:-1].replace(', ', '_')
        checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, whitened_layer)

    model = torch.nn.DataParallel(model)

    checkpoint_path = os.path.join(checkpoint_folder, concept_names, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_path))

    return model


def load_deepmir_resnet_bn_model(args, checkpoint_folder="./checkpoints", whitened_layer=None):
    prefix_name = args.prefix[:args.prefix.rfind('_')]

    if whitened_layer is None:
        raise Exception("whitened_layer argument is required")
    else:
        arch = './checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint'
        model = DeepMirResNetBN(args, model_file='./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar')
        checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, 2)

    model = torch.nn.DataParallel(model)

    print(checkpoint_name)

    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_path))

    return model


def load_deepmir_resnet_cw_v2_model(args, checkpoint_folder="./checkpoints", whitened_layer=None, fold_n=None):
    if len(whitened_layer) == 1:
        # rfind finds the last occurrence of _
        prefix_name = args.prefix[:args.prefix.rfind('_')]
    else:
        # find the first occurrence of _ with the first instance in the whitened layers
        prefix_name = args.prefix[:args.prefix.find(f'_{whitened_layer[0]}')]

    concept_names = '_'.join(args.concepts.split(','))
    if whitened_layer is None:
        raise Exception("whitened_layer argument is required")
    elif len(whitened_layer) == 1:
        arch = './checkpoints/resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_v2_BN_1_checkpoint.pth.tar'
        model = DeepMirResNetTransferv2(args, [int(whitened_layer)], model_file=arch)
        # checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, whitened_layer)
        checkpoint_name = '{}_{}_foldn{}_model_best.pth.tar'.format(prefix_name, whitened_layer, fold_n)
    else:
        arch = './checkpoints/resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_v2_BN_1_checkpoint.pth.tar'
        whitened_layer = [int(x) for x in args.whitened_layers.split(',')]
        model = DeepMirResNetTransferv2(args, whitened_layer, model_file=arch)
        # create a string from the list of whitened layers, remove the [] (first and last characters) and replace the
        # comma's in between the strings with _
        whitened_layer = str(whitened_layer)[1:-1].replace(', ', '_')
        # checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, whitened_layer)
        checkpoint_name = '{}_{}_foldn{}_model_best.pth.tar'.format(prefix_name, whitened_layer, fold_n)

    model = torch.nn.DataParallel(model)

    checkpoint_path = os.path.join(checkpoint_folder, concept_names, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_path))

    return model


def load_deepmir_resnet_v2_bn_model(args, checkpoint_folder="./checkpoints", whitened_layer=None):
    prefix_name = args.prefix[:args.prefix.rfind('_')]

    if whitened_layer is None:
        raise Exception("whitened_layer argument is required")
    else:
        arch = './checkpoints/resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_v2_BN_1_checkpoint.pth.tar'
        model = DeepMirResNetBNv2(args, model_file=arch)
        checkpoint_name = 'resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_v2_BN_1_model_best.pth.tar'

    model = torch.nn.DataParallel(model)

    print(checkpoint_name)

    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_path))

    return model
