from sklearn import tree
import numpy as np
from scipy.special import softmax
import graphviz
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree._tree import TREE_LEAF

# X is the input data with features (in my case the activations at the neurons that have been aligned with concepts),
# Y is the target (in my case the predictions)
# to obtain X, we need to go over all the activation numpy arrays stored in the activations folder in the plot folder

concepts_dir = ['large_asymmetric_bulge', 'base_pairs_wobbles_in_stem', 'gap_start', 'presence_terminal_loop']
concepts_string = '_'.join(concepts_dir)
base_folder = f'./plot/{concepts_string}/deepmir_resnet_cw_v3/activations'

activations_allconcepts = []
learnable_concepts = ['large_asymmetric_bulge', 'base_pairs_wobbles_in_stem', 'gap_start']
for neuron in range(0, len(learnable_concepts)):
    file_name = f'activations_relu_cwlayer3_neuron{neuron}.npy'
    file = np.load(os.path.join(base_folder, file_name))
    activations_neuron = []
    for i in range(len(file)):
        value = float(file[i][0])
        activations_neuron.append(value)
    activations_allconcepts.append(activations_neuron)
#%%
# so, now the activations values are used as feature values
# however, they can be very varying....
# maybe try some sort of binning or use the mean/median... to create binary conditions

# # create binning (3 values) based on max values per concept activation
# print(np.max(activations_allconcepts[0]))
# print(np.min(activations_allconcepts[0]))
# print(np.max(activations_allconcepts[1]))
# print(np.min(activations_allconcepts[1]))
#
#
#
# bins_concept0 = np.array(range(0, round(np.max(activations_allconcepts[0])), 1))
# binned_values_concept0 = np.digitize(activations_allconcepts[0], bins_concept0)
# bins_concept1 = np.array(range(0, round(np.max(activations_allconcepts[1])), 1))
# binned_values_concept1 = np.digitize(activations_allconcepts[1], bins_concept1)

# %%
# now create binning based on percentiles (0.25,0.5,0.75,1)
stats_concept0 = pd.Series(activations_allconcepts[0]).describe()
stats_concept1 = pd.Series(activations_allconcepts[1]).describe()

concept0_25 = stats_concept0.T['25%']
concept0_50 = stats_concept0.T['50%']
concept0_75 = stats_concept0.T['75%']
concept0_100 = stats_concept0.T['max']
concept1_25 = stats_concept1.T['25%']
concept1_50 = stats_concept1.T['50%']
concept1_75 = stats_concept1.T['75%']
concept1_100 = stats_concept1.T['max']

bins_concept0 = [concept0_25, concept0_50, concept0_75, concept0_100]
binned_values_concept0 = np.digitize(activations_allconcepts[0], bins_concept0)
bins_concept1 = [concept1_25, concept1_50, concept1_75, concept1_100]
binned_values_concept1 = np.digitize(activations_allconcepts[1], bins_concept1)
# %%
# X = np.array([binned_values_concept0, binned_values_concept1])
X = np.array(activations_allconcepts)
X = np.transpose(np.asarray(X))

# %%
# the test predictions are in the same order as the test data (test loader model is without shuffling)

y = np.load(f'./output/{concepts_string}.npy')
# apply softmax (pytorch Crossentropy() does this and this has not yet been done on the saved predictions)
y = softmax(y, axis=2)
y = np.argmax(y, axis=2)
y = y.flatten()

# %%
clf = tree.DecisionTreeClassifier(random_state=3, min_samples_split=0.15)
clf = clf.fit(X, y)


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


# %%
prune_duplicate_leaves(clf)
tree.plot_tree(clf)
plt.show()

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['predicted_similarity_large_asymmetric_bulge',
                                               'predicted_similarity_base_pairs_wobbles_in_stem',
                                               'predicted_similarity_gap_start'],
                                class_names=np.unique(y).astype(str),
                                rounded=True,
                                proportion=True,
                                special_characters=True,
                                # filled=True,
                                impurity=False)  # impurity takes away the gini value in the viz

# %%
# first find the lowest node number (last in the string)
lowest_node = dot_data[-6:-3]  # this is the case when this index is an integer with two numbers
if lowest_node == '"] ':
    lowest_node = dot_data[-61:-58]
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

#%%
# remove all value items from the string
for i in range(0, lowest_node):
    if 'value' in dot_data:
        value_index = dot_data.find('value')
        value_end_index = dot_data[value_index:].find('>')
        dot_data = dot_data[:value_index] + dot_data[value_index+value_end_index+1:]
    i += 1

non_leaf_nodes = []
for node_index in range(0, lowest_node + 1):
    # all the nodes than end up having an arrow to another are not leaves
    if f'{str(node_index)} -> ' in dot_data:
        non_leaf_nodes.append(node_index)
    else:
        continue

# remove all abundant class labels from the string (abundant: not in leaf)
for node_index in non_leaf_nodes:
    index = dot_data.find(f'{node_index} [label=<predicted')
    index_bracket = dot_data[index:].find('] ')
    if 'class = 1' in dot_data[index:index+index_bracket]:
        dot_data = dot_data[:index] + dot_data[index:].replace('<br/>class = 1', '', 1)
    elif 'class = 0' in dot_data[index:index+index_bracket]:
        dot_data = dot_data[:index] + dot_data[index:].replace('<br/>class = 0', '', 1)

graph = graphviz.Source(dot_data)
graph.render("premirna")
# %%
text_representation = tree.export_text(clf)
print(text_representation)
