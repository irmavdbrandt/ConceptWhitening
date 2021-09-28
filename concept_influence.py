import numpy as np
from scipy.special import softmax
import os
import matplotlib.pyplot as plt
import torch.optim
import seaborn as sns
import pandas as pd
from scipy.stats import rankdata

concepts_dir = ['large_asymmetric_bulge', 'base_pairs_wobbles_in_stem']
concepts_string = '_'.join(concepts_dir)
base_folder_test = f'./plot/{concepts_string}_OLD/deepmir_resnet_cw_v3/activations'
model_path = 'checkpoints/large_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_RESNET_' \
             'PREMIRNA_v3_CPT_WHITEN_TRANSFER_final_twoouputs_3_foldn1_model_best.pth.tar'


def collect_data_plots(n_neurons_cwlayer, model_path):
    # collect the activations from the test set
    activations_test_allconcepts = []
    for neuron in range(0, n_neurons_cwlayer):
        file_name = f'activations_relu_cwlayer3_neuron{neuron}.npy'
        file = np.load(os.path.join(base_folder_test, file_name), allow_pickle=True)
        activations_neuron = []
        for i in range(len(file)):
            value = float(file[i][0])
            activations_neuron.append(value)
        activations_test_allconcepts.append(activations_neuron)

    # get the weights of the final class prediction layer
    state_dict = torch.load(model_path, map_location='cpu')['state_dict']

    # specify the weights of the two outputs nodes, the negative class output node is first, then the positive one
    weights_linear2 = state_dict['module.model.linear2.weight']
    weights_linear2_class0 = weights_linear2[0]
    weights_linear2_class1 = weights_linear2[1]

    # collect the final predictions made by the model
    # the test predictions are in the same order as the test data (test loader model is without shuffling)
    predictions = np.load(f'./output/{concepts_string}.npy')
    # apply softmax (pytorch Crossentropy() does this and this has not yet been done on the saved predictions)
    predictions = softmax(predictions, axis=2)
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.flatten()

    # calculate the weighted activations (weight * act) with the weights of the negative class node if the final
    # prediction is the negative class
    weighted_activation_class0_pred0 = []
    act_class0_pred0 = []
    for activation, weight in zip(activations_test_allconcepts, weights_linear2_class0):
        weighted_activation_class0_pred0_neuron = []
        act_class0_pred0_neuron = []
        for value, prediction in zip(activation, predictions):
            if prediction == 0:
                weighted_act = weight * value
                act_class0_pred0_neuron.append(value)
                weighted_activation_class0_pred0_neuron.append(weighted_act)
        weighted_activation_class0_pred0.append(weighted_activation_class0_pred0_neuron)
        act_class0_pred0.append(act_class0_pred0_neuron)

    # same as before but now for the positive class weights and predictions = 1
    weighted_activation_class1_pred1 = []
    act_class1_pred1 = []
    for activation, weight in zip(activations_test_allconcepts, weights_linear2_class1):
        weighted_activation_class1_pred1_neuron = []
        act_class1_pred1_neuron = []
        for value, prediction in zip(activation, predictions):
            if prediction == 1:
                weighted_act = weight * value
                act_class1_pred1_neuron.append(value)
                weighted_activation_class1_pred1_neuron.append(weighted_act)
        weighted_activation_class1_pred1.append(weighted_activation_class1_pred1_neuron)
        act_class1_pred1.append(act_class1_pred1_neuron)

    return weighted_activation_class0_pred0, weighted_activation_class1_pred1, act_class0_pred0, act_class1_pred1


#%%
# collect the activations from the test set
activations_test_allconcepts = []
for neuron in range(0, 72):
    file_name = f'activations_relu_cwlayer3_neuron{neuron}.npy'
    file = np.load(os.path.join(base_folder_test, file_name), allow_pickle=True)
    activations_neuron = []
    for i in range(len(file)):
        value = float(file[i][0])
        activations_neuron.append(value)
    activations_test_allconcepts.append(activations_neuron)
#%%
# collect the data we need for the plot
weighted_activation_class0_pred0, weighted_activation_class1_pred1, act_class0_pred0, act_class1_pred1 = collect_data_plots(72, model_path)
#%%
# collect the final predictions made by the model
# the test predictions are in the same order as the test data (test loader model is without shuffling)
predictions_before_softmax = np.load(f'./output/{concepts_string}.npy')
#%%
# apply softmax (pytorch Crossentropy() does this and this has not yet been done on the saved predictions)
predictions = softmax(predictions_before_softmax, axis=2)
predictions = np.argmax(predictions, axis=2)
predictions = predictions.flatten()

# get the weights of the final class prediction layer
state_dict = torch.load(model_path, map_location='cpu')['state_dict']

# specify the weights of the two outputs nodes, the negative class output node is first, then the positive one
weights_linear2 = state_dict['module.model.linear2.weight']
weights_linear2_class0 = weights_linear2[0]
weights_linear2_class1 = weights_linear2[1]

#%%
print(activations_test_allconcepts[0][15])
print(activations_test_allconcepts[1][15])
print(weights_linear2_class0[0])
print(weights_linear2_class0[1])
print(activations_test_allconcepts[0][626]*weights_linear2_class0[0])
print(activations_test_allconcepts[1][626]*weights_linear2_class0[1])

#%%
weighted_activations = []
for weight, i in zip(weights_linear2_class0[2:len(weights_linear2_class0)], range(len(activations_test_allconcepts)-2)):
    i = i+2
    activation = activations_test_allconcepts[i][618]
    weighted_act = weight*activation
    weighted_activations.append(weighted_act)
#%%
print(weighted_activations)
print(np.sum(weighted_activations))
#%%
print(np.sum(weighted_activations) + 0.023 + -0.161)
#%%
print(0.196 + 0.095 + -3.989)
# %%
def dist_influence_plot(class_of_interest, concept_names):
    # collect the data we need for the plot
    weighted_activation_class0_pred0, weighted_activation_class1_pred1, act_class0_pred0, act_class1_pred1 = \
        collect_data_plots(72, model_path)

    activations_concepts = []
    for i in range(len(concept_names)):
        if class_of_interest == 'positive':
            activations_concepts.append(act_class1_pred1[i])
        elif class_of_interest == 'negative':
            activations_concepts.append(act_class0_pred0[i])

    activations_remainingnodes = []
    if class_of_interest == 'positive':
        for j in range(len(act_class1_pred1[0])):
            sum_act = 0
            for i in range(2, len(act_class1_pred1)):
                 sum_act += act_class1_pred1[i][j]
            activations_remainingnodes.append(sum_act)
    elif class_of_interest == 'negative':
        for j in range(len(act_class0_pred0[0])):
            sum_act = 0
            for i in range(2, len(act_class0_pred0)):
                sum_act += act_class0_pred0[i][j]
            activations_remainingnodes.append(sum_act)

    print(len(activations_remainingnodes))


    # first we need to rank the activations values so that we can use a high-low colorscale
    ranked_activations_combined = []
    for activations in activations_concepts:
        array_activations = np.array(activations)
        ranked_act = rankdata(array_activations)
        ranked_activations_combined = ranked_activations_combined + list(ranked_act)

    # rank the sum of the activation values of the remaining nodes
    ranked_act_remaining = rankdata(np.array(activations_remainingnodes))
    ranked_activations_combined = ranked_activations_combined + list(ranked_act_remaining)
    print(len(ranked_activations_combined))

    # collect the other data: a list of the concept names * number of instances and the weighted activations in 1 list
    # first the concepts
    concepts_list = []
    for concept_name in concept_names:
        # we need a concept name x number of times, where x is the total length of the number of instances in the
        # test set that have been predicted 0/1
        concept_list = len(activations_concepts[0]) * [concept_name]
        concepts_list = concepts_list + concept_list

    # append sum of remaining 70 nodes in the concept list
    concept_list_remaining = len(activations_remainingnodes) * ['sum of remaining 70 nodes']
    concepts_list = concepts_list + concept_list_remaining
    print(len(concepts_list))

    # create SHAP inspired plot of weighted activations and the activations values itself (high-low)
    # do this for each for both weight vectors
    weighted_activations = []
    for i in range(len(concept_names)):
        weighted_activations_singleconcept = []
        if class_of_interest == 'positive':
            for item in weighted_activation_class1_pred1[i:i + 1][0]:
                weighted_activations_singleconcept.append(item.item())
        elif class_of_interest == 'negative':
            for item in weighted_activation_class0_pred0[i:i + 1][0]:
                weighted_activations_singleconcept.append(item.item())
        weighted_activations = weighted_activations + weighted_activations_singleconcept

    weighted_activations_remaining = []
    if class_of_interest == 'positive':
        for i in range(len(weighted_activation_class1_pred1[0:1][0])):
            weighted_activations_singleconcept = []
            sum_weighted_activations_singleconcept = 0
            for j in range(2, len(weighted_activation_class1_pred1)):
                sum_weighted_activations_singleconcept += weighted_activation_class1_pred1[j:j + 1][0][i].item()
            weighted_activations_remaining.append(sum_weighted_activations_singleconcept)
    elif class_of_interest == 'negative':
        for i in range(len(weighted_activation_class0_pred0[0:1][0])):
            weighted_activations_singleconcept = []
            sum_weighted_activations_singleconcept = 0
            for j in range(2, len(weighted_activation_class0_pred0)):
                sum_weighted_activations_singleconcept += weighted_activation_class0_pred0[j:j + 1][0][i].item()
            weighted_activations_remaining.append(sum_weighted_activations_singleconcept)
    print(np.min(weighted_activations_remaining))
    weighted_activations = weighted_activations + weighted_activations_remaining

    print(len(weighted_activations))

    # combine all data in a dictionary and convert to a dataframe
    data_dict = {'Influence concept through weighted activation': weighted_activations,
                 'Activations': ranked_activations_combined,
                 'Concept': concepts_list}
    dataframe = pd.DataFrame(data_dict)

    # start creating the plot
    fig, ax = plt.subplots()

    # set the style of the plot
    sns.set(style="ticks")
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')

    # create a normalizer for the colormap and specify the colormap color
    norm = plt.Normalize(np.min(data_dict['Activations']), np.max(data_dict['Activations']))
    cmap = sns.color_palette("flare", as_cmap=True)

    catplot = sns.catplot(data=dataframe, x='Influence concept through weighted activation', y='Concept',
                          hue='Activations', palette='flare', kind='strip', legend=False)

    catplot.fig.suptitle(f'Influence concept on predictions {class_of_interest} class', fontsize=12,
                         fontweight='bold')
    # todo: automate this part still based on the concepts names provided
    catplot.set_yticklabels(['Asymmetric\nbulge of\nwidth 15+\npixels', 'At least 90%\nbase pairs\nand wobbles\nin stem',
                             'Sum of\n70 remaining\nnodes'], fontsize=8)
    catplot.set_axis_labels('Influence concept through weighted activation', 'Concept', fontsize=9)
    # increase the limit of the x-axis so that it matches the positive and negative class results
    # catplot.set(xlim=[-0.5, 1])
    catplot.set(xlim=[-10, 10])
    # specify xaxis tick labels based on the new x-axis limits (there are 4 ticks)
    # catplot.set_xticklabels([-0.5, 0, 0.5, 1], fontsize=8)

    plt.grid()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(sm, ticks=[np.min(ranked_activations_combined), np.max(ranked_activations_combined)],
                            shrink=0.75)
    colorbar.ax.set_yticklabels(['Low', 'High'], fontsize=8)
    colorbar.set_label('Original activation value', fontsize=9)

    ax = catplot.axes[0]

    # create a black line at the 0-point of the weighted activations to accentuate this point
    ax[0].axvline(0, color='black', alpha=0.5)

    plt.tight_layout()
    plt.savefig('CI_shap_negative.svg', format='svg')
    plt.savefig('CI_shap_negative.png', format='png')
    plt.show()


dist_influence_plot('negative', ['large_asymmetric_bulge', 'base_pairs_wobbles_stem'])

#%%
# calculate the means of the weightedactivations for both classes
def mean_max_activations(activations):
    mean_weighted_act = []
    max_weighted_act = []

    for weighted_act_vector in activations:
        mean = np.mean(weighted_act_vector)
        max = np.max(weighted_act_vector)
        mean_weighted_act.append(mean)
        max_weighted_act.append(max)

    return mean_weighted_act, max_weighted_act


def concept_influence_barplot(n_neurons_cwlayer, class_of_interest, min_max):
    # collect the data we need for the plot
    weighted_activation_class0_pred0, weighted_activation_class1_pred1, act_class0_pred0, act_class1_pred1 = \
        collect_data_plots(72, model_path)

    mean_weighted_act_class0_pred0, maxweighted_act_class0_pred0 = mean_max_activations(
        weighted_activation_class0_pred0)
    mean_weighted_act_class1_pred1, maxweighted_act_class1_pred1 = mean_max_activations(
        weighted_activation_class1_pred1)

    # get the concept OR residual information that has the highest influence on the output
    print(np.max(mean_weighted_act_class0_pred0), np.argmax(mean_weighted_act_class0_pred0))
    print(np.max(mean_weighted_act_class1_pred1), np.argmax(mean_weighted_act_class1_pred1))
    print(np.min(mean_weighted_act_class0_pred0), np.argmin(mean_weighted_act_class0_pred0))
    print(np.min(mean_weighted_act_class1_pred1), np.argmin(mean_weighted_act_class1_pred1))
    max_neg = np.argmax(mean_weighted_act_class0_pred0)
    max_pos = np.argmax(mean_weighted_act_class1_pred1)
    min_neg = np.argmin(mean_weighted_act_class0_pred0)
    min_pos = np.argmin(mean_weighted_act_class1_pred1)

    # Create the bar plot showing the average concept influence of the concepts of interest and the average of all
    # remaining nodes in the cw layer
    labels = concepts_dir
    if min_max:
        labels = labels + [f'Average influence of all {n_neurons_cwlayer} nodes', f'Min weighted activation '
                            f'{class_of_interest} class',  f'Max weighted activation {class_of_interest} class']
    else:
        labels = labels + [f'Average influence of all {n_neurons_cwlayer} nodes']

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    # sns.set_style('whitegrid')
    sns.set(style="ticks")
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')

    if min_max:
        class1_pred1 = mean_weighted_act_class1_pred1[0:len(concepts_dir)] + [np.mean(mean_weighted_act_class1_pred1),
                                                                              np.min(mean_weighted_act_class1_pred1),
                                                                              np.max(mean_weighted_act_class1_pred1)]
        print(class1_pred1)
        class0_pred0 = mean_weighted_act_class0_pred0[0:len(concepts_dir)] + [np.mean(mean_weighted_act_class0_pred0),
                                                                              np.min(mean_weighted_act_class0_pred0),
                                                                              np.max(mean_weighted_act_class0_pred0)]
        print(class0_pred0)
    else:
        class1_pred1 = mean_weighted_act_class1_pred1[0:len(concepts_dir)] + [np.mean(mean_weighted_act_class1_pred1)]
        class0_pred0 = mean_weighted_act_class0_pred0[0:len(concepts_dir)] + [np.mean(mean_weighted_act_class0_pred0)]

    if class_of_interest == 'positive':
        ax.bar(x, class1_pred1, width, label='(Activation*Weight_class1 | pred=1)', zorder=3)
    elif class_of_interest == 'negative':
        ax.bar(x, class0_pred0, width, label='(Activation*Weight_class0 | pred=0)', zorder=3)

    # case for when I want to combine both class arrays
    # ax.bar(x + 0.5 * width, class1_pred1, width, label='(Activation*Weight_class1 | pred=1)', zorder=3)
    # ax.bar(x - 0.5 * width, class0_pred0, width, label='(Activation*Weight_class0 | pred=0)', zorder=3)

    ax.set_ylabel('Weighted activation', fontsize=10)
    ax.set_ylim([min(np.min(class1_pred1), np.min(class0_pred0)) - 0.05,
                 max(np.max(class1_pred1), np.max(class0_pred0)) + 0.05])
    ax.set_xlabel('Concept', fontsize=10)
    ax.set_title(f'Average influence concept on predictions {class_of_interest} class', fontweight='bold')
    ax.set_xticks(x)
    # todo: automate this part still based on the concepts names provided
    ax.set_xticklabels(['Large\nasymmetric\nbulge', 'At least 90%\nbase pairs\nand wobbles\nin stem',
                        f'Average influence\n of all {n_neurons_cwlayer} nodes', f'Min influence\nof a single node',
                        f'Max influence\nof a single node'], fontsize=9)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    # ax.get_legend().remove()

    # create a black line at the 0-point of the weighted activations to accentuate this point
    ax.axhline(0, color='black', alpha=0.5)

    fig.tight_layout()
    plt.savefig(f'CI_average_{class_of_interest}.svg', format='svg')
    plt.savefig(f'CI_average_{class_of_interest}.png', format='png')
    plt.show()


concept_influence_barplot(72, 'negative', True)



# %%
# add the means in a bar chart
# labels = [str(i) for i in range(4)]
labels = concepts_dir[0:2]
labels2 = labels + ['Average weighted activations of all 72 axes', 'Max weighted activation class 0', 'Max weighted '
                                                                                                      'activation '
                                                                                                      'class 1']

x = np.arange(len(labels2))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
ax.grid(zorder=0)

class0_pred0 = mean_weighted_act_class0_pred0[0:2] + [np.mean(mean_weighted_act_class0_pred0),
                                                      mean_weighted_act_class0_pred0[10],
                                                      mean_weighted_act_class0_pred0[36]]
# class0_pred1 = mean_weighted_act_class0_pred1[0:4] + [np.mean(mean_weighted_act_class0_pred1)]
# class1_pred0 = mean_weighted_act_class1_pred0[0:4] + [np.mean(mean_weighted_act_class1_pred0)]
class1_pred1 = mean_weighted_act_class1_pred1[0:2] + [np.mean(mean_weighted_act_class1_pred1),
                                                      mean_weighted_act_class1_pred1[10],
                                                      mean_weighted_act_class1_pred1[36]]

# rects1 = ax.bar(x - 1.5*width, class0_pred0, width, label='(Activation*Weight_class0 | pred=0)')
# rects2 = ax.bar(x - 0.5*width, class0_pred1, width, label='(Activation*Weight_class0 | pred=1)')
# rects3 = ax.bar(x + 0.5*width, class1_pred0, width, label='(Activation*Weight_class1 | pred=0)')
# rects4 = ax.bar(x + 1.5*width, class1_pred1, width, label='(Activation*Weight_class1 | pred=1)')
rects1_v2 = ax.bar(x - 0.5 * width, class0_pred0, width, label='(Activation*Weight_class0 | pred=0)', zorder=3)
rects4_v3 = ax.bar(x + 0.5 * width, class1_pred1, width, label='(Activation*Weight_class1 | pred=1)', zorder=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weighted activation')
ax.set_xlabel('Concept')
ax.set_title('Average influence concept on class prediction')
ax.set_xticks(x)
ax.set_xticklabels(labels2, rotation=90)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')

fig.tight_layout()
plt.show()

# %%
weighted_activation_class0 = []
for activation, weight in zip(activations_test_allconcepts, weights_linear2_class0):
    weighted_activation_class0_neuron = []
    for value in activation:
        weighted_act = weight * value
        weighted_activation_class0_neuron.append(weighted_act)
    weighted_activation_class0.append(weighted_activation_class0_neuron)

weighted_activation_class1 = []
for activation, weight in zip(activations_test_allconcepts, weights_linear2_class1):
    weighted_activation_class1_neuron = []
    for value in activation:
        weighted_act = weight * value
        weighted_activation_class1_neuron.append(weighted_act)
    weighted_activation_class1.append(weighted_activation_class1_neuron)

# %%
pred_class0 = 0
for item in weighted_activation_class0:
    pred_class0 += item[0]
pred_class1 = 0
for item in weighted_activation_class1:
    pred_class1 += item[0]
pred_class0_final = pred_class0 + state_dict['module.model.linear2.bias'][0]
pred_class1_final = pred_class1 + state_dict['module.model.linear2.bias'][1]

# %%
y = np.load(f'./output/{concepts_string}.npy')

# %%
print(np.mean(list(weights_linear2_class0)))
print(np.mean(list(weights_linear2_class1)))

print(np.min(list(weights_linear2_class0)))
print(np.min(list(weights_linear2_class1)))

print(np.max(list(weights_linear2_class0)))
print(np.max(list(weights_linear2_class1)))
