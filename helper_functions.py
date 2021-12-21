import numpy as np
import torch.utils.data
from MODELS.model_resnet import *
import os

np.seterr(divide='ignore', invalid='ignore')


def load_deepmir_model(args, whitened_layer=None, checkpoint_name=None):
    """
    :param args: arguments given by user
    :param whitened_layer: layer index of CW layer (e.g., 1, 2 or 3 for the DeepMir model)
    :param checkpoint_name: name of model whose weights (i.e., checkpoint) need to be loaded
    :return: deepmir model with weights of checkpoint specified by user
    """
    model = None

    if args.arch == "deepmir_vfinal_cw" or args.arch == "deepmir_vfinal_bn":
        arch = 'checkpoints/deepmir_vfinal_bn/DEEPMIR_vfinal_BN_finetune_checkpoint.pth.tar'
        if args.arch == "deepmir_vfinal_cw":
            model = DeepMir_vfinal_Transfer(args, [int(whitened_layer)], model_file=arch)
        else:
            model = DeepMir_vfinal_BN(args, model_file=arch)
    elif args.arch == "deepmir_v2_cw" or args.arch == "deepmir_v2_bn":
        arch = './checkpoints/deepmir_v2_bn/DEEPMIR_v2_BN_finetune_checkpoint.pth.tar'
        if args.arch == "deepmir_v2_cw":
            model = DeepMir_v2_Transfer(args, [int(whitened_layer)], model_file=arch)
        else:
            model = DeepMir_v2_BN(args, model_file=arch)

    model = torch.nn.DataParallel(model)

    if os.path.isfile(checkpoint_name):
        print("=> loading checkpoint '{}'".format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("checkpoint {} not found!".format(checkpoint_name))

    return model


def get_activations_CWlayer(args, test_loader, model, whitened_layer, type_training, neurons_cwlayer):
    """
    :param args: arguments given by user
    :param test_loader: loader for test dataset
    :param model: trained CW model
    :param whitened_layer: layer index of CW layer (e.g., 1, 2 or 3 for the DeepMir model)
    :param type_training: argument specifying whether the training or test set activations should be stored
    :param neurons_cwlayer: number of neurons in the CW layer
    """

    with torch.no_grad():
        dst = './output/activations/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        if type_training == "get_activations_train":
            dst = dst + 'activations_train/'
        else:
            dst = dst + 'activations_test/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        model = model.model

        outputs_cw = []

        def hook_relu(module, input_tensor, output_tensor):
            """
            This function collects the activation values of the CW layer of interest by attaching a hook to the layer
            of interest
            """
            outputs_cw.append(output_tensor.cpu().numpy())

        if args.arch == "deepmir_v2_cw":
            model.relu9.register_forward_hook(hook_relu)
        if args.arch == "deepmir_vfinal_cw":
            model.pool4.register_forward_hook(hook_relu)

        for k in range(0, neurons_cwlayer):
            paths = []
            val = []
            for i, (input_tensor, _, path) in enumerate(test_loader):
                paths += list(path)
                input_var = torch.autograd.Variable(input_tensor)
                outputs_cw = []
                model(input_var)
                for output in outputs_cw:
                    val.append(output[0].mean(2)[k])

            # save the activation values as numpy arrays
            arr = list(zip(val, list(paths)))
            np.save(os.path.join(dst, f'activations_relu_cwlayer{whitened_layer}_neuron{k}'), arr)


def load_activations(neurons_cwlayer, folder_activations, file_name):
    """
    :param neurons_cwlayer: number of neurons in the CW layer
    :param folder_activations: name of folder where the activation values are stored
    :param file_name: name of file with activation values for all data instances and for all neurons of the CW layer
    :return: list containing lists of activation values (one for each neuron) for all data instances
    """
    # collect the activations values from all neurons in the prediction layer of the model
    activations_neurons = []
    for neuron in range(0, neurons_cwlayer):
        activations_file = np.load(os.path.join(f'{folder_activations}', f'{file_name}_neuron{neuron}.npy'),
                                   allow_pickle=True)
        activation_values = []
        for activation_value in activations_file:
            activation_values.append(float(activation_value[0]))
        activations_neurons.append(activation_values)

    return activations_neurons


def calculate_weighted_activation(activations_neurons, weights, predictions_test, class_label_of_interest):
    """
    :param activations_neurons: activations values for all neurons in the prediction layer of a CW model
    :param weights: the weights of the prediction layer of a CW model
    :param predictions_test: the predictions made using the test set instances and a CW model
    :param class_label_of_interest: for which class (i.e., positive or negative pre-miRNA) the weighted activation
    values should be collected
    :return: all weighted activation and base activation values for all neurons in the prediction layer of a CW model
    for a specific pre-miRNA class
    """
    weighted_activations = []
    activations_class_of_interest = []
    for activation_values, weight in zip(activations_neurons, weights):
        weighted_activations_neuron = []
        activations_neuron = []
        for activation_value, prediction in zip(activation_values, predictions_test):
            if prediction == class_label_of_interest:
                weighted_activations_neuron.append(weight * activation_value)
                activations_neuron.append(activation_value)
        weighted_activations.append(weighted_activations_neuron)
        activations_class_of_interest.append(activations_neuron)

    return weighted_activations, activations_class_of_interest


def mean_max_activations(weighted_activations):
    """
    :param weighted_activations: list with weighted activation values for all instances of a dataset and all neurons
    of a specific model layer
    :return: the weighted activation values for all neurons averaged over all data instances and the max weighted acti-
    vation value of all data instances for each neuron
    """
    mean_weighted_activations = []
    max_weighted_activations = []

    for vector in weighted_activations:
        mean_weighted_activations.append(np.mean(vector))
        max_weighted_activations.append(np.max(vector))

    return mean_weighted_activations, max_weighted_activations
