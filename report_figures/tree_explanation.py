from sklearn import tree
import numpy as np
from scipy.special import softmax
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree._tree import TREE_LEAF
import torch.optim
from helper_functions import load_activations


def collect_data_tree(neurons_cwlayer, CW_model_path, output_layer, concept_names):
    """
    :param neurons_cwlayer: number of neurons in the CW layer
    :param CW_model_path: path to where the trained CW model is stored
    :param output_layer: name of output layer of the CW model
    :param concept_names: names of concepts used in the CW model
    :return: weighted activation (= weight neuron * activation value) values collected using the output layer of the
    CW model and the predictions. Both are collected for the training and test set
    """
    # collect the activations values from all neurons in the prediction layer of the model
    activations_train = load_activations(neurons_cwlayer, 'output/activations/' + '_'.join(concept_names)
                                         + '/' + 'activations_train', 'activations_relu_cwlayer3')
    activations_test = load_activations(neurons_cwlayer, 'output/activations/' + '_'.join(concept_names)
                                        + '/' + 'activations_test', 'activations_relu_cwlayer3')

    # collect the final predictions made by the model on the training and test set
    concepts_string = '_'.join(concept_names)

    pred_train = np.load(f'output/predictions/{concepts_string}/predictions_train.npy')
    # apply softmax, followed by an argmax and a flatten procedure on the predictions
    pred_train = list(np.argmax(softmax(pred_train, axis=2), axis=2).flatten())
    pred_train = np.array(['positive' if int(x) == 1 else 'negative' for x in pred_train])

    pred_test = np.load(f'output/predictions/{concepts_string}/predictions_test.npy')
    # apply softmax, followed by an argmax and a flatten procedure on the predictions
    pred_test = list(np.argmax(softmax(pred_test, axis=2), axis=2).flatten())
    pred_test = np.array(['positive' if int(x) == 1 else 'negative' for x in pred_test])

    # get the weights of the prediction layer
    weights = torch.load(CW_model_path, map_location='cpu')['state_dict'][f'module.model.{output_layer}.weight']
    # rescale the weights so that the influences are not misleading by normalizing them (min-max norm)
    norm_weights_class0 = (weights[0] - weights[0].min()) / (weights[0].max() - weights[0].min())
    norm_weights_class1 = (weights[1] - weights[1].min()) / (weights[1].max() - weights[1].min())

    # calculate the weighted activations_test (weight * act) with the weights of the negative class node if the final
    weighted_activation_train = []
    for activation, weight_class0, weight_class1 in zip(activations_train, norm_weights_class0, norm_weights_class1):
        weighted_activation_neuron = []
        weighted_activation = None
        for value, prediction in zip(activation, pred_train):
            if prediction == 'negative':
                weighted_activation = weight_class0 * value
            elif prediction == 'positive':
                weighted_activation = weight_class1 * value
            weighted_activation_neuron.append(weighted_activation)
        weighted_activation_train.append(weighted_activation_neuron)

    weighted_activation_test = []
    for activation, weight_class0, weight_class1 in zip(activations_test, norm_weights_class0, norm_weights_class1):
        weighted_activation_neuron = []
        weighted_activation = None
        for value, prediction in zip(activation, pred_test):
            if prediction == 'negative':
                weighted_activation = weight_class0 * value
            elif prediction == 'positive':
                weighted_activation = weight_class1 * value
            weighted_activation_neuron.append(weighted_activation)
        weighted_activation_test.append(weighted_activation_neuron)

    # the weighted activations are used as training and test data for the tree classifier
    data_train = np.array(weighted_activation_train[0:len(concept_names)])
    data_train = np.transpose(np.asarray(data_train))
    data_test = np.array(weighted_activation_test[0:len(concept_names)])
    data_test = np.transpose(np.asarray(data_test))

    return data_train, data_test, pred_train, pred_test


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF


def prune_duplicate_leaves(mdl):
    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(mdl.tree_, decisions)


def tree_CI_plot(data, predictions, min_samples_leaf):
    """
    :param data: data used for making the tree (i.e. weighted activation values of the training/test set)
    :param predictions: predictions that are associated with the data given in the data parameter
    :param min_samples_leaf: minimum number of samples in a leaf of the tree
    :return: decision tree classifier trained with the data and predictions
    """
    tree_clf = tree.DecisionTreeClassifier(random_state=3, min_samples_leaf=min_samples_leaf)
    tree_clf = tree_clf.fit(data, predictions)
    prune_duplicate_leaves(tree_clf)
    tree.plot_tree(tree_clf)
    plt.show()

    dot_data = tree.export_graphviz(tree_clf,
                                    out_file=None,
                                    feature_names=['Normalized influence of "large asymmetric bulge" concept',
                                                   'Normalized influence of "at least 90% base pairs and wobbles in '
                                                   'stem" concept'],
                                    class_names=np.unique(y_train).astype(str),
                                    rounded=True,
                                    proportion=True,
                                    special_characters=True,
                                    impurity=True)  # impurity takes away the gini value in the viz

    # first find the lowest node number (last in the string)
    lowest_node = dot_data[-6:-3]  # this is the case when this index is an integer with two numbers
    if lowest_node == '"] ':
        lowest_node = dot_data[-61:-58]
    # check whether this node is an integer
    if lowest_node in range(0, 100):
        for i in range(0, int(lowest_node)):
            for j in range(i + 1, int(lowest_node) + 1):
                index = dot_data.find(f'{str(i)} -> {str(j)} ;')
                if index == -1:
                    continue
                elif j == i + 1:
                    if (len(str(i)) == 2) and (len(str(j)) == 2):
                        dot_data = dot_data[:index + 8] + ' [labeldistance=2.5, labelangle=45, ' \
                                                          'headlabel="True"]' + dot_data[index + 8:]
                    elif len(str(i)) == 2:
                        dot_data = dot_data[:index + 7] + ' [labeldistance=2.5, labelangle=45, ' \
                                                          'headlabel="True"]' + dot_data[index + 7:]
                    elif len(str(j)) == 2:
                        dot_data = dot_data[:index + 7] + ' [labeldistance=2.5, labelangle=45, ' \
                                                          'headlabel="True"]' + dot_data[index + 7:]
                    else:
                        dot_data = dot_data[:index + 6] + ' [labeldistance=2.5, labelangle=45, ' \
                                                          'headlabel="True"]' + dot_data[index + 6:]
    if lowest_node in range(0, 100):
        for i in range(0, int(lowest_node)):
            for j in range(i + 1, int(lowest_node) + 1):
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

    graph = graphviz.Source(dot_data)
    graph.render("report_figures/concept_influence_tree/premirna")

    return clf


# path to CW model to be used when collecting activation values
model_path = 'checkpoints/large_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final_' \
             'twoouputs_3_foldn1_checkpoint.pth.tar'

X_train, X_test, y_train, y_test = collect_data_tree(72, model_path, 'linear2', ['large_asymmetric_bulge',
                                                                                 'base_pairs_wobbles_in_stem'])
clf = tree_CI_plot(X_train, y_train, 0.05)

y_pred_train = clf.predict(X_train)
accuracy_train = clf.score(X_train, y_train)
y_pred_test = clf.predict(X_test)
accuracy_test = clf.score(X_test, y_test)
print(f'Accuracy of tree classifier on training set: {accuracy_train} and on test set: {accuracy_test}')
