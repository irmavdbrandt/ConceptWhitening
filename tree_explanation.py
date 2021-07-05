from sklearn import tree
# from preparation.concept_saving import create_annotated_df
import numpy as np
from scipy.special import softmax
import graphviz
import os
import matplotlib.pyplot as plt

# # X is the input data with features (in my case the concepts), Y is the target (in my case the predictions)
# all_data = create_annotated_df()
# test_data = all_data.loc[all_data['set'] == 'test']
# test_data = test_data.loc[test_data['class_label'] != 'p']
# test_data = test_data.reset_index(drop=True)
# # convert the continuous features to binary ones and fill the na values with 0
# test_data = test_data.fillna(value=0)
# # %%
# test_data.loc[test_data['base_pairs_wobbles_in_stem'] >= 0.9, 'base_pairs_wobbles_in_stem'] = True
# test_data.loc[test_data['base_pairs_wobbles_in_stem'] < 0.9, 'base_pairs_wobbles_in_stem'] = False
# test_data.loc[test_data['largest_asymmetric_bulge'] < 6.5, 'largest_asymmetric_bulge'] = False
# test_data.loc[test_data['largest_asymmetric_bulge'] != False, 'largest_asymmetric_bulge'] = True
# concepts = ['base_pairs_wobbles_in_stem', 'presence_terminal_loop', 'largest_asymmetric_bulge',
#             'AU_pair_begin_maturemiRNA']
# # concepts = ['base_pairs_wobbles_in_stem', 'largest_asymmetric_bulge']
# X = test_data[concepts]


# X is the input data with features (in my case the activations at the neurons that have been aligned with concepts),
# Y is the target (in my case the predictions)
# to obtain X, we need to go over all the activation numpy arrays stored in the activations folder in the plot folder

concepts_dir = ['largest_asymmetric_bulge', 'base_pairs_wobbles_in_stem']
concepts_string = '_'.join(concepts_dir)
base_folder = f'./plot/{concepts_string}/deepmir_resnet_cw_v2/activations'

activations_allconcepts = []
for neuron in range(0, len(concepts_dir)):
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

# create binning (3 values) based on max values per concept activation
print(np.max(activations_allconcepts[0]))
print(np.min(activations_allconcepts[0]))
print(np.max(activations_allconcepts[1]))
print(np.min(activations_allconcepts[1]))



bins_concept0 = np.array(range(0, round(np.max(activations_allconcepts[0])), 1))
binned_values_concept0 = np.digitize(activations_allconcepts[0], bins_concept0)
bins_concept1 = np.array(range(0, round(np.max(activations_allconcepts[1])), 1))
binned_values_concept1 = np.digitize(activations_allconcepts[1], bins_concept1)

#%%
# now create binning based on percentiles (0.25,0.5,0.75,1)
import pandas as pd
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
#%%
X = np.array([binned_values_concept0, binned_values_concept1])
X = np.transpose(np.asarray(X))

#%%
# the test predictions are in the same order as the test data (test loader model is without shuffling)

y = np.load(f'./output/{concepts_string}.npy')
# apply softmax (pytorch Crossentropy() does this and this has not yet been done on the saved predictions)
y = softmax(y, axis=2)
y = np.argmax(y, axis=2)
y = y.flatten()
# # %%
# dir_der_signs = np.load(f'./plot/{concepts_string}/deepmir_resnet_cw_v2/importance_targets/'
#                         f'concept_dir_deriv_signs_targets_3.npy')
# num_of_learnable_concepts = 2
# X = dir_der_signs[:, 0:2]

#%%

clf = tree.DecisionTreeClassifier(random_state=2)
clf = clf.fit(X, y)
tree.plot_tree(clf)
plt.show()

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['predicted_similarity_large_asymmetric_bulge',
                                               'predicted_similarity_base_pairs_wobbles_in_stem'],
                                class_names=np.unique(y).astype(str),
                                filled=True, rounded=True, proportion=True,
                                special_characters=True)

# # change conditions to be concept == False instead of <= 0.5
# indices_False = [i for i in range(len(dot_data)) if dot_data.startswith(f' &le; 0.5', i)]
#
# for i in indices_False:
#     dot_data = dot_data[:i] + f' = False' + dot_data[i + len(f' = False'):]
#     dot_data = dot_data.replace('False5', 'False')

#%%
# first find the lowest node number (last in the string)
lowest_node = dot_data[-6:-3]  # this is the case when this index is an integer with two numbers
# lowest_node = '4'
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

graph = graphviz.Source(dot_data)
graph.render("premirna")
#%%
text_representation = tree.export_text(clf)
print(text_representation)

#%%
activations_basepairs = np.load(
    'plot/largest_asymmetric_bulge_base_pairs_wobbles_in_stem_presence_terminal_loop_AU_pair_begin_maturemiRNA/'
    'deepmir_resnet_cw_v2/3_rot_cw/activation_values_conceptbase_pairs_wobbles_in_stem.npy')

activations_bulges = np.load(
    'plot/largest_asymmetric_bulge_base_pairs_wobbles_in_stem_presence_terminal_loop_AU_pair_begin_maturemiRNA/'
    'deepmir_resnet_cw_v2/3_rot_cw/activation_values_conceptlargest_asymmetric_bulge.npy')

#%%
class0 = int(len(activations_basepairs)/2)
act_basepairs_class0 = activations_basepairs[:, 0][0:class0]
act_basepairs_class0 = [float(x) for x in act_basepairs_class0]

act_basepairs_class1 = activations_basepairs[:, 0][class0:]
act_basepairs_class1 = [float(x) for x in act_basepairs_class1]

mean_act_basepairs_class0 = np.mean(act_basepairs_class0)
mean_act_basepairs_class1 = np.mean(act_basepairs_class1)


class0 = int(len(activations_bulges)/2)
act_bulges_class0 = activations_bulges[:, 0][0:class0]
act_bulges_class0 = [float(x) for x in act_bulges_class0]

act_bulges_class1 = activations_bulges[:, 0][class0:]
act_bulges_class1 = [float(x) for x in act_bulges_class1]

mean_act_bulges_class0 = np.mean(act_bulges_class0)
mean_act_bulges_class1 = np.mean(act_bulges_class1)


#%%
activations_termloop = np.load(
    'plot/presence_terminal_loop_AU_pair_begin_maturemiRNA/'
    'deepmir_resnet_cw_v2/3_rot_cw/activation_values_conceptpresence_terminal_loop.npy')

activations_AU = np.load(
    'plot/presence_terminal_loop_AU_pair_begin_maturemiRNA/'
    'deepmir_resnet_cw_v2/3_rot_cw/activation_values_conceptAU_pair_begin_maturemiRNA.npy')

#%%
class0 = int(len(activations_termloop)/2)
act_termloop_class0 = activations_termloop[:, 0][0:class0]
act_termloop_class0 = [float(x) for x in act_termloop_class0]

act_termloop_class1 = activations_termloop[:, 0][class0:]
act_termloop_class1 = [float(x) for x in act_termloop_class1]

mean_act_termloop_class0 = np.mean(act_termloop_class0)
mean_act_termloop_class1 = np.mean(act_termloop_class1)


class0 = int(len(activations_AU)/2)
act_AU_class0 = activations_AU[:, 0][0:class0]
act_AU_class0 = [float(x) for x in act_AU_class0]

act_AU_class1 = activations_AU[:, 0][class0:]
act_AU_class1 = [float(x) for x in act_AU_class1]

mean_act_AU_class0 = np.mean(act_AU_class0)
mean_act_AU_class1 = np.mean(act_AU_class1)

#%%
import numpy as np

activations_relucw_neuron0 = np.load(
    'plot/largest_asymmetric_bulge_base_pairs_wobbles_in_stem/'
    'deepmir_resnet_cw_v2/activations/activations_relu_cwlayer3_neuron0.npy')

activations_relucw_neuron1 = np.load(
    'plot/largest_asymmetric_bulge_base_pairs_wobbles_in_stem/'
    'deepmir_resnet_cw_v2/activations/activations_relu_cwlayer3_neuron1.npy')

# activations_relucw_neuron1 = np.load(
#     'plot/largest_asymmetric_bulge_AU_pair_begin_maturemiRNA/'
#     'deepmir_resnet_cw_v2/activations/activations_neuron1.npy')

# activations_relulinear1 = np.load(
#     'plot/largest_asymmetric_bulge_AU_pair_begin_maturemiRNA/'
#     'deepmir_resnet_cw_v2/activations/activations_relulinear1.npy')
