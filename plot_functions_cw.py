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
from numpy import linalg as LA, random
from sklearn.metrics import roc_auc_score
from shutil import copyfile
from MODELS.iterative_normalization import iterative_normalization_py
from MODELS.model_resnet import *
import os
from helper_functions import load_deepmir_model

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
                       dst=None):
    """
    :param args: arguments given by user
    :param test_loader: data loader containing the test images
    :param model: model used for training
    :param whitened_layers: whitened layer
    :param print_other: boolean specifying whether other neurons not linked to concepts should be used
    :param activation_mode: which activation mode is used to find the top50 most activated images
    :param dst: destination folder where plots will be stored
    :return: this function finds the top 50 images that gets the greatest activations_test with respect to the concepts.
    Concept activation values are obtained based on iternorm_rotation module outputs.
    Since concept corresponds to channels in the predictions_test, we look for the top50 images whose kth channel
    activations are high.
    """
    # switch to evaluate mode
    model.eval()

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
    model = model.model

    outputs = []

    def hook(module, input_tensor, output_tensor):
        """
       :param module: model layer
       :param input_tensor: input for model layer
       :param output_tensor: predictions_test of model layer
       :return: gradients from forward pass on layer of interest (CW layer)
       """

        if args.arch == "deepmir_vfinal_cw":
            outputs.append(input_tensor[0])
        else:
            X_hat = iterative_normalization_py.apply(input_tensor[0], module.running_mean, module.running_wm,
                                                     module.num_channels, module.T, module.eps, module.momentum,
                                                     module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat)

    if args.arch == "deepmir_v2_cw":
        for layer in layer_list:
            if int(layer) == 1:
                model.bn1.register_forward_hook(hook)
            elif int(layer) == 2:
                model.bn2.register_forward_hook(hook)
            elif int(layer) == 3:
                model.bn3.register_forward_hook(hook)
    elif args.arch == "deepmir_vfinal_cw":
        model.pool4.register_forward_hook(hook)

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
                # create a similarity scalar from all the feature maps of 1 neuron (i.e. concept axis) based on the
                # activation mode of choice
                for output in outputs:
                    if activation_mode == 'pool_max':
                        kernel_size = 2
                        r = output.shape[3] % kernel_size
                        if args.arch == "deepmir_vfinal_cw":
                            val = np.concatenate((val, output.mean((2, 3))[:, k]))
                        else:
                            if r == 0:
                                val = np.concatenate((val, skimage.measure.block_reduce(output[:, :, :, :],
                                                                                        (
                                                                                            1, 1, kernel_size,
                                                                                            kernel_size),
                                                                                        np.max).mean((2, 3))[:, k]))
                            else:
                                val = np.concatenate((val, skimage.measure.block_reduce(output[:, :, :-r, :-r],
                                                                                        (
                                                                                            1, 1, kernel_size,
                                                                                            kernel_size),
                                                                                        np.max).mean((2, 3))[:, k]))

                # combine all the activation scalars for all test images in one list
                val = val.reshape((len(outputs), -1))
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals, val), 1)

            for i, layer in enumerate(layer_list):
                # zip the activation values for each concept obtained on a test image with the test image's path
                arr = list(zip(list(vals[i, :]), list(paths)))
                # sort the list based on the activation values (such that the most activated img are first)
                arr.sort(key=lambda t: t[0], reverse=True)
                # take the first 50 images and save them in the directory
                for activation_idx_highest in range(50):
                    src = arr[activation_idx_highest][1]
                    copyfile(src, output_path + '/' + 'layer' + layer + '_' + str(activation_idx_highest + 1) + '.jpg')
                # save the 10 least activated images for each concept
                for activation_idx_least in range(len(arr) - 10, len(arr)):
                    src = arr[activation_idx_least][1]
                    copyfile(src, output_path + '/' + 'layer' + layer + '_' + str(activation_idx_least + 1) + '.jpg')
                # save the 10 moderately activated images, with moderate we mean images with activation similar to the
                # mode of the activation
                for activation_idx_middle in range(round((len(arr) / 2) - 10), round((len(arr) / 2) + 10)):
                    src = arr[activation_idx_middle][1]
                    copyfile(src, output_path + '/' + 'layer' + layer + '_' + str(activation_idx_middle + 1) + '.jpg')

                for activation_idx_25 in range(round((len(arr) / 4) - 5), round((len(arr) / 4) + 5)):
                    src = arr[activation_idx_25][1]
                    copyfile(src, output_path + '/' + 'layer' + layer + '_' + str(activation_idx_25 + 1) + '.jpg')

                for activation_idx_75 in range(round(3 * (len(arr) / 4) - 5), round(3 * (len(arr) / 4) + 5)):
                    src = arr[activation_idx_75][1]
                    copyfile(src, output_path + '/' + 'layer' + layer + '_' + str(activation_idx_75 + 1) + '.jpg')

    return print("Done with searching for the top 50")


def plot_top10(plot_cpt=None, layer=None, type_training=None, dst=None):
    """
    :param plot_cpt: list of concepts
    :param layer: whitened layer
    :param type_training: string specifying whether the we are in evaluation mode (testing) or not (validating)
    :param dst: destination where plots will be stored
    :return: plot showing the top-10 most activated images along the concept axes. The images are obtained from the
    plot_concept_top50() function.
    """
    if type_training == 'evaluate':
        folder = dst + str(layer) + '_rot_cw/'
    else:
        folder = dst + str(layer) + '_rot_cw/'

    # case for when we only have 1 concept, we create a plot with 2 rows of 5 images each
    if len(plot_cpt) == 1:
        fig, axes = plt.subplots(figsize=(30, 3 * len(plot_cpt)), nrows=2, ncols=5)
        cpt = plot_cpt[0]
        for i in range(10):
            if i < 5:
                axes[0, i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 1) + '.jpg'))
                axes[0, i].set_yticks([])
                axes[0, i].set_xticks([])
            else:
                axes[1, i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 1) + '.jpg'))
                axes[1, i].set_yticks([])
                axes[1, i].set_xticks([])

    # in case we have more than 1 concept, we create a plot with 10 images in each row
    else:
        fig, axes = plt.subplots(figsize=(30, 3 * len(plot_cpt)), nrows=len(plot_cpt), ncols=10)

        for c, cpt in enumerate(plot_cpt):
            for i in range(10):
                axes[c, i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' + str(i + 1) + '.jpg'))
                axes[c, i].set_yticks([])
                axes[c, i].set_xticks([])

        for ax, row in zip(axes[:, 0], plot_cpt):
            ax.set_ylabel(row.replace('_', '\n'), rotation=90, size='large', fontsize=20, wrap=False)

        fig.tight_layout()
        plt.show()
        fig.savefig(folder + 'layer' + str(layer) + '.svg', format='svg')


def intra_concept_dot_product_vs_inter_concept_dot_product(args, concept_dir, layer, plot_cpt=None,
                                                           arch='deepmir_vfinal_cw', model=None, ticklabels_xaxis=None,
                                                           ticklabels_yaxis=None):
    """
    :param args: arguments given by user
    :param concept_dir: directory containing concept images of test/validation set
    :param layer: whitened layer
    :param plot_cpt: list of concepts
    :param arch: model architecture
    :param model: trained model
    :param ticklabels_xaxis: pretty concept labels for the x-axis of the plot
    :param ticklabels_yaxis: pretty concept labels for the y-axis of the plot
    :return: this method compares the intra concept group dot product with inter concept group dot product
    """
    # create directory where results will be stored
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/inner_product/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    # create concept loader for concept images of test set
    concept_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(concept_dir, transforms.Compose([transforms.ToTensor(), ])),
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
    model = model.model

    # initialize an empty dictionary that will store the concept representations
    representations = {}
    for cpt in plot_cpt:
        representations[cpt] = []

    with torch.no_grad():

        outputs = []

        def hook(module, input_tensor, output_tensor):
            """
            :param module: model layer
            :param input_tensor: input for model layer
            :param output_tensor: predictions_test of model layer
            :return: gradients from forward pass on layer of interest (CW layer)
            """
            # concept alignment code (CW)
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
        if args.arch == "deepmir_v2_cw" or args.arch == "deepmir_vfinal_cw":
            if layer == 1:
                model.bn1.register_forward_hook(hook)
            elif layer == 2:
                model.bn2.register_forward_hook(hook)
            elif layer == 3:
                model.bn3.register_forward_hook(hook)

        # obtain outputs for each concept image in terms of their activation on the concept axis
        for j, (input_img, y, path) in enumerate(concept_loader):
            labels = y.cpu().numpy().flatten().astype(np.int32).tolist()
            input_var = torch.autograd.Variable(input_img)
            outputs = []
            model(input_var)
            for instance_index in range(len(labels)):
                instance_concept_index = labels[instance_index]
                if concept_list[instance_concept_index] in plot_cpt:
                    output_shape = outputs[0].shape
                    representation_mean = outputs[0][instance_index:instance_index + 1, :, :, :].transpose(
                        (0, 2, 3, 1)).reshape((-1, output_shape[1])).mean(axis=0)  # mean of all pixels of instance
                    # get the cpt_index channel of the predictions_test
                    representations[concept_list[instance_concept_index]].append(representation_mean)

    # representation of concepts in matrix form
    dot_product_matrix = np.zeros((len(plot_cpt), len(plot_cpt))).astype('float')
    m_representations = {}
    m_representations_normed = {}
    intra_dot_product_mean = {}
    intra_dot_product_mean_normed = {}
    # compute the average pairwise cosine similarity between latent representations of same concepts to obtain the
    # intra similarity measure
    for i, concept in enumerate(plot_cpt):
        m_representations[concept] = np.stack(representations[concept], axis=0)  # n * (h*w)
        m_representations_normed[concept] = m_representations[concept] / LA.norm(m_representations[concept], axis=1,
                                                                                 keepdims=True)
        intra_dot_product_mean[concept] = np.matmul(m_representations[concept],
                                                    m_representations[concept].transpose()).mean()
        # normalize the values by taking the mean
        intra_dot_product_mean_normed[concept] = np.matmul(m_representations_normed[concept],
                                                           m_representations_normed[concept].transpose()).mean()
        dot_product_matrix[i, i] = 1.0

    inter_dot_product_mean = {}
    inter_dot_product_mean_normed = {}

    # compute the average pairwise cosine similarity between latent representations of 2 different concepts to obtain
    # the inter similarity measure
    for i in range(len(plot_cpt)):
        for j in range(i + 1, len(plot_cpt)):
            cpt_1 = plot_cpt[i]
            cpt_2 = plot_cpt[j]
            # normalize the values by taking the mean
            inter_dot_product_mean[cpt_1 + '_' + cpt_2] = np.matmul(m_representations[cpt_1],
                                                                    m_representations[cpt_2].transpose()).mean()
            inter_dot_product_mean_normed[cpt_1 + '_' + cpt_2] = np.matmul(m_representations_normed[cpt_1],
                                                                           m_representations_normed[
                                                                               cpt_2].transpose()).mean()
            dot_product_matrix[i, j] = abs(inter_dot_product_mean_normed[cpt_1 + '_' + cpt_2]) / np.sqrt(
                abs(intra_dot_product_mean_normed[cpt_1] * intra_dot_product_mean_normed[cpt_2]))
            dot_product_matrix[j, i] = dot_product_matrix[i, j]

    print(intra_dot_product_mean, inter_dot_product_mean)
    print(intra_dot_product_mean_normed, inter_dot_product_mean_normed)
    print(dot_product_matrix)

    # plot the inter and intra similarity measures as a heatmap
    plt.figure()
    if ticklabels_xaxis is None and ticklabels_yaxis is None:
        ticklabels = [s.replace('_', ' ') for s in plot_cpt]
        sns.set(font_scale=1)
        ax = sns.heatmap(dot_product_matrix, vmin=0, vmax=1, xticklabels=ticklabels, yticklabels=ticklabels,
                         annot=True, cmap='Oranges', annot_kws={"fontsize": 12})
    else:
        sns.set(font_scale=1)
        ax = sns.heatmap(dot_product_matrix, vmin=0, vmax=1, xticklabels=ticklabels_xaxis, yticklabels=ticklabels_yaxis,
                         annot=True, cmap='Oranges', annot_kws={"fontsize": 12})
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title('Normalized inter- and intra-concept similarities', fontweight='bold')
    ax.figure.tight_layout()
    plt.savefig(dst + arch + '_' + str(layer) + '.svg', format='svg')

    return intra_dot_product_mean, inter_dot_product_mean, intra_dot_product_mean_normed, inter_dot_product_mean_normed


def plot_auc_cw(args, concept_dir, whitened_layers, plot_cpt=None, activation_mode='pool_max', concept_labels=None):
    """
    :param args: arguments given by user
    :param concept_dir: directory containing concept images in the test/validation set
    :param whitened_layers: whitened layers
    :param plot_cpt: list of concepts
    :param activation_mode: activation mode chosen to compute the activation score for a concept image along the
     concept axis
    :param concept_labels: pretty labels of concept used in the plot
    :return: For each layer and each concept, using activation value as the predicted probability of being a certain
    concept, auc score is computed with respect to label
    """
    # create a directory that will store the plot
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/auc/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    dst += 'cw/'
    if not os.path.exists(dst):
        os.mkdir(dst)

    # create a data loader that feeds the concept images of the test/validation directory in shuffled order
    concept_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(concept_dir, transforms.Compose([transforms.ToTensor(), ])),
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
        # get the index of the concept of interest from the concept list
        cpt_idx_2 = concept_list.index(cpt)
        # get the index of the concept of interest from the plot_cpt list
        # note that this list can be in different order than the concept list, that is why we calculate both
        cpt_idx = plot_cpt.index(cpt)
        for i, layer in enumerate(layer_list):
            with torch.no_grad():
                model = load_deepmir_model(args, whitened_layer=args.whitened_layers,
                                           checkpoint_name=args.checkpoint_name)
                model.eval()
                model = model.module
                model = model.model
                outputs = []

                if args.arch == "deepmir_v2_cw":
                    # define the hook for the cw layer, this will save the activations_test on the concept axes
                    def hook(module, input_tensor, output_tensor):
                        """
                        :param module: model layer
                        :param input_tensor: input for model layer
                        :param output_tensor: predictions_test of model layer
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

                        outputs.append(X_hat)

                    # append the hooks to the concept whitening layers
                    layer = int(layer)
                    if layer == 1:
                        model.bn1.register_forward_hook(hook)
                    elif layer == 2:
                        model.bn2.register_forward_hook(hook)
                    elif layer == 3:
                        model.bn3.register_forward_hook(hook)

                elif args.arch == "deepmir_vfinal_cw":
                    def hook(module, input, output):
                        outputs.append(output.cpu().numpy())

                    model.pool4.register_forward_hook(hook)

                labels = []
                vals = []
                for j, (input_image, y, path) in enumerate(concept_loader):
                    # add the concept label to the list of labels
                    labels += list(y)
                    input_var = torch.autograd.Variable(input_image)
                    outputs = []
                    model(input_var)
                    # get the activation scalar using the activation mode of choice on all feature maps of 1 neuron
                    for output in outputs:
                        if args.arch == "deepmir_v2_cw":
                            if activation_mode == 'pool_max':
                                kernel_size = 2
                                r = output.shape[3] % kernel_size
                                if r == 0:
                                    vals += list(
                                        skimage.measure.block_reduce(output[:, :, :, :],
                                                                     (1, 1, kernel_size, kernel_size),
                                                                     np.max).mean((2, 3))[:, cpt_idx])
                                else:
                                    vals += list(skimage.measure.block_reduce(output[:, :, :-r, :-r],
                                                                              (1, 1, kernel_size, kernel_size),
                                                                              np.max).mean((2, 3))[:, cpt_idx])
                        elif args.arch == "deepmir_vfinal_cw":
                            value = np.mean(output[0], axis=2)[cpt_idx]
                            vals += [value]
                del model
            vals = np.array(vals)
            labels = np.array(labels)
            # give all labels in the labels list a 1 if they are equal to the concept index, and a 0 otherwise
            # this is done to be able to use the one-vs-all auc score
            labels = (labels == cpt_idx_2).astype('int32')
            n_samples = labels.shape[0]
            t = 5
            # create a split of all test samples in 5 different parts
            idx = np.array_split(np.random.permutation(n_samples), t)
            auc_t = []
            # compute the auc score for the 5 different parts of the test activation samples. The test activation
            # samples for a concept are regarded as the predicted probability to be the concept. This value, together
            # with the binary label for being the concept or not (ground truth label) is used in the auc score calc
            for j in range(t):
                auc_t.append(roc_auc_score(labels[idx[j]], vals[idx[j]]))
            # take the mean of all 5 parts of the auc scores on the test set, this is the overall auc score
            aucs[c, i] = np.mean(auc_t)
            aucs_err[c, i] = np.std(auc_t)
            print(aucs[c, i])
            print(aucs_err[c, i])

    print('AUC-CW', aucs)
    print('AUC-CW-err', aucs_err)
    np.save(dst + 'aucs_cw.npy', aucs)
    np.save(dst + 'aucs_cw_err.npy', aucs_err)

    # create a bar plot with the obtained mean auc scores and std
    if concept_labels is None:
        concept_labels = list(args.concepts.split(','))

    x_pos = np.arange(len(concept_labels))
    aucs_plot = [auc for sublist in aucs for auc in sublist]
    aucs_err_plot = [auc_err for sublist in aucs_err for auc_err in sublist]

    plt.figure()
    plt.bar(x_pos, aucs_plot, yerr=aucs_err_plot, ecolor='black', align='center')
    plt.xticks(x_pos, concept_labels, rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel('Average AUC scores')
    plt.title('Average concept AUC scores', fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.savefig(dst + 'concept_aucs' + str(layer) + '.svg', format='svg')

    return aucs


def plot_correlation(dst, args, test_loader, model, layer):
    """
    :param dst: destination where plots will be stored
    :param args: arguments given by user
    :param test_loader: data loader containing the validation/test images
    :param model: trained model
    :param layer: whitened layer
    :return: correlation matrix showing the absolute correlation between the axes (neurons) of the whitened or BN
    layers and the mean correlation.
    """
    with torch.no_grad():
        # create directory where plot will be stored
        if args.arch == "deepmir_vfinal_bn" or args.arch == "deepmir_v2_bn":
            dst = dst + 'correlation_matrix_BN/'
        elif args.arch == "deepmir_vfinal_cw" or args.arch == "deepmir_v2_cw":
            dst = dst + 'correlation_matrix/'
        if not os.path.exists(dst):
            os.mkdir(dst)

        model.eval()
        model = model.module
        model = model.model

        outputs = []

        def hook(module, input_tensor, output_tensor):
            """
            :param module: layer module
            :param input_tensor: input of layer
            :param output_tensor: predictions_test of layer
            :return: predictions_test values (i.e. activations_test) of layer of interest appended to outputs list
            """
            size_X = output_tensor.size()
            X = output_tensor.transpose(0, 1).reshape(size_X[1], -1).transpose(0, 1)
            M = X.cpu().numpy()
            outputs.append(M)

        layer = int(layer)
        if args.arch == "deepmir_v2_bn" or args.arch == "deepmir_vfinal_bn":
            model.relu9.register_forward_hook(hook)
        else:
            if layer == 1:
                model.relu3.register_forward_hook(hook)
            elif layer == 2:
                model.relu6.register_forward_hook(hook)
            elif layer == 3:
                model.relu9.register_forward_hook(hook)

        for i, (input_tensor, _, path) in enumerate(test_loader):
            input_var = torch.autograd.Variable(input_tensor)
            model(input_var)

        activation = np.vstack(outputs)
        # normalize the activation values
        activation -= activation.mean(0)
        activation = activation / activation.std(0)
        # compute the correlation by taking the dot product between different activation values
        Sigma = np.dot(activation.transpose((1, 0)), activation) / activation.shape[0]
        # convert all empty and inf values to 0
        Sigma = np.nan_to_num(Sigma, nan=0, posinf=0, neginf=0)
        plt.figure()
        # take the absolute values of the correlations and plot them in a heatmap
        sns.heatmap(np.abs(Sigma), cmap='hot')
        plt.tight_layout()
        if args.arch == "deepmir_vfinal_bn" or args.arch == "deepmir_v2_bn":
            plt.savefig(dst + 'BN.jpg')
        else:
            plt.savefig(dst + str(layer) + '.jpg')

        # mean correlation
        mean_corr = np.mean(np.abs(Sigma))
        print('Mean correlation: ', mean_corr)

        return mean_corr


def saliency_map_concept_cover(args, layer, num_concepts=2, model=None):
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
    if not os.path.exists(dst):
        os.mkdir(dst)

    model.eval()
    model = model.module
    model = model.model

    outputs = []

    def hook(module, input_tensor, output_tensor):
        """
        :param module: layer module
        :param input_tensor: input of layer
        :param output_tensor: predictions_test of layer
        :return: predictions_test values (i.e. activations_test) of layer of interest appended to outputs list
        """
        outputs.append(output_tensor)

    # append hooks to the whitened layers
    layer = int(layer)
    if args.arch == "deepmir_v2_cw":
        if layer == 1:
            model.bn1.register_forward_hook(hook)
        elif layer == 2:
            model.bn2.register_forward_hook(hook)
        elif layer == 3:
            model.bn3.register_forward_hook(hook)
    elif args.arch == "deepmir_vfinal_cw":
        model.pool4.register_forward_hook(hook)

    concepts = args.concepts.split(',')

    # start with an initialization of the size of the concept cover, since we have images of 25x100, we want the cover
    # not to exceed these sizes. 5 is most logical given the multiples of 5 (25 and 100) as image sizes
    cover_size = 5
    with torch.no_grad():
        # create a data loader that will go over the 50 most activated images per concept
        base_dir = 'plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/' + \
                   '_'.join(args.whitened_layers) + '_rot_cw/'

        most_activated_img_loader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(base_dir, transforms.Compose([transforms.ToTensor(), ])),
            batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

        for i, (input, _, path) in enumerate(most_activated_img_loader):
            # create a random patch, based on the initialized cover size and the fact that it is an RGB image (3 dim)
            random_patch = torch.tensor(np.random.normal(size=(3, cover_size, cover_size)))
            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            # get the activation values of the input image from the predictions_test at the cw layer
            if args.arch == "deepmir_vfinal_cw":
                # this model already applies maxpool2d to the predictions_test
                base_activations = outputs[0]
            else:
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
                    if args.arch == "deepmir_vfinal_cw":
                        # this model already applies maxpool2d to the predictions_test
                        new_activations = outputs[0]
                    else:
                        new_activations = F.max_pool2d(outputs[0], kernel_size=2, stride=1)
                    new_activations = new_activations[0, :, :, :].clamp(min=0.0).mean(dim=(1, 2))
                    outputs = []
                    decrease_in_activations = base_activations - new_activations
                    for j in range(len(concepts)):
                        saliency[j, p:p + cover_size, q:q + cover_size] += decrease_in_activations[j].cpu().numpy()
                    counter[p:p + cover_size, q:q + cover_size] += 1.0
            saliency = saliency / counter

            u_limit = np.percentile(saliency, 99.99)
            l_limit = np.percentile(saliency, 0.01)
            saliency = saliency.clip(l_limit, u_limit)

            lower_limit = np.percentile(saliency, 94)
            saliency[saliency < lower_limit] = 0.3
            saliency[saliency >= lower_limit] = 1.0

            input_image = input[0, :, :, :].permute(1, 2, 0)

            concepts_sorted = sorted(concepts)
            for j in range(num_concepts):
                concept_index = int(str(_)[-3])
                base_folder = dst + f"activatedimg_concept_{concepts_sorted[concept_index]}"
                if not os.path.exists(base_folder):
                    os.mkdir(base_folder)
                save_folder = os.path.join(base_folder, "concept_" + concepts[j])
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                image = Image.fromarray(np.uint8(input_image * 255)).convert('RGBA')
                image = np.array(image)
                image[:, :, 3] = (saliency[j, :, :] * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(os.path.join(save_folder, str(path)[-9:-7] + '.png'), 'PNG')
                print("saved: " + str(j))


def saliency_map_cover_most_activated_neuron(args, layer, neuron=None, neurons=None, model=None):
    """
    :param args: arguments given by user
    :param layer: whitened layer
    :param neuron: neuron for which we want the activation map plotted over the images
    :param neurons: the most activated neurons stored in a list
    :param model: trained model
    :return: images in test set with receptive field of concept is plotted over original image. This shows for all
    test images where the neuron that has been aligned with a concept looks at
    """
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/saliency_map_cover_mostactivated/'
    if not os.path.exists(dst):
        os.mkdir(dst)

    model.eval()
    model = model.module
    model = model.model

    outputs = []

    def hook(module, input_tensor, output_tensor):
        """
        :param module: layer module
        :param input_tensor: input of layer
        :param output_tensor: predictions_test of layer
        :return: predictions_test values (i.e. activations_test) of layer of interest appended to outputs list
        """
        outputs.append(output_tensor)

    # append hooks to the whitened layers
    layer = int(layer)
    if args.arch == "deepmir_v2_cw":
        if layer == 1:
            model.bn1.register_forward_hook(hook)
        elif layer == 2:
            model.bn2.register_forward_hook(hook)
        elif layer == 3:
            model.bn3.register_forward_hook(hook)
    elif args.arch == "deepmir_vfinal_cw":
        model.pool4.register_forward_hook(hook)

    cover_size = 5
    with torch.no_grad():
        base_dir = 'plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/' + \
                   '_'.join(args.whitened_layers) + '_rot_otherdim/'
        most_activated_img_loader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(base_dir, transforms.Compose([transforms.ToTensor(), ])),
            batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

        for i, (input, _, path) in enumerate(most_activated_img_loader):
            random_patch = torch.tensor(np.random.normal(size=(3, cover_size, cover_size)))
            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if args.arch == "deepmir_vfinal_cw":
                # this model already applies maxpool2d to the predictions_test
                base_activations = outputs[0]
            else:
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
                    if args.arch == "deepmir_vfinal_cw":
                        # this model already applies maxpool2d to the predictions_test
                        new_activations = outputs[0]
                    else:
                        new_activations = F.max_pool2d(outputs[0], kernel_size=2, stride=1)
                    new_activations = new_activations[0, :, :, :].clamp(min=0.0).mean(dim=(1, 2))
                    outputs = []
                    decrease_in_activations = base_activations - new_activations
                    saliency[0, p:p + cover_size, q:q + cover_size] += decrease_in_activations[neuron].cpu().numpy()
                    counter[p:p + cover_size, q:q + cover_size] += 1.0
            saliency = saliency / counter

            u_limit = np.percentile(saliency, 99.99)
            l_limit = np.percentile(saliency, 0.01)
            saliency = saliency.clip(l_limit, u_limit)

            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            lower_limit = np.percentile(saliency, 94)
            saliency[saliency < lower_limit] = 0.3
            saliency[saliency >= lower_limit] = 1.0

            input_image = input[0, :, :, :].permute(1, 2, 0).cpu().numpy()

            # for j in nodes:
            neuron_index = int(str(_)[-3])
            save_folder = dst + f"activatedimg_neuron{neuron}_imagesneuron{neurons[neuron_index]}"
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            image = Image.fromarray(np.uint8(input_image * 255)).convert('RGBA')
            image = np.array(image)
            image[:, :, 3] = (saliency[0, :, :] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(os.path.join(save_folder, str(path)[-9:-7] + '.png'), 'PNG')
            print("saved covered image")
