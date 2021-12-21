import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import torch.optim
import seaborn as sns
import pandas as pd
from scipy.stats import rankdata
from helper_functions import load_activations, calculate_weighted_activation, mean_max_activations


def collect_data_CI_plots(concept_names, neurons_cwlayer, CW_model_path, output_layer, class_of_interest):
    """
    :param concept_names: names of concepts used in the CW model
    :param neurons_cwlayer: number of neurons in the CW layer
    :param CW_model_path: path to where the trained CW model is stored
    :param output_layer: name of output layer of the CW model
    :param class_of_interest: pre-miRNA class of interest for collecting data values for the plot
    :return: weighted activation (= weight neuron * activation value) and activation values collected using the output
    layer of the CW model and its predictions
    """
    # collect the activations values from all neurons in the prediction layer of the model
    activations_neurons = load_activations(neurons_cwlayer, 'output/activations/' + '_'.join(concept_names)
                                           + '/' + 'activations_test', 'activations_relu_cwlayer3')

    # collect the final predictions made by the model
    # Note: the test predictions are in the same order as the test data (test loader does not shuffle data)
    concepts_string = '_'.join(concept_names)
    predictions_test = np.load(f'output/predictions/{concepts_string}/predictions_test.npy')
    # apply softmax, followed by an argmax and a flatten procedure on the predictions
    predictions_test = np.argmax(softmax(predictions_test, axis=2), axis=2).flatten()

    # get the weights of the prediction layer
    weights = torch.load(CW_model_path, map_location='cpu')['state_dict'][f'module.model.{output_layer}.weight']

    if class_of_interest == 'negative':
        # calculate weighted activations using the weights of the negative class node
        weighted_activations, activations = calculate_weighted_activation(activations_neurons, weights[0],
                                                                          predictions_test, 0)
        return weighted_activations, activations

    elif class_of_interest == 'positive':
        # calculate weighted activations using the weights of the positive class node
        weighted_activations, activations = calculate_weighted_activation(activations_neurons, weights[1],
                                                                          predictions_test, 1)
        return weighted_activations, activations


def SHAP_CI_plot(neurons_cwlayer, class_of_interest, concept_names, y_ticklabels, xaxis_limits):
    """
    :param neurons_cwlayer: number of neurons in the CW layer
    :param class_of_interest: pre-miRNA class of interest for collecting data values for the plot
    :param concept_names: names of the concepts used in the CW model
    :param y_ticklabels: pretty labels for the y-axis
    :param xaxis_limits: limits used for the x-axis (= weighted activation values)
    """
    # STEP 1: collect all required data for the plot
    # collect all activations values and weighted activation values from the prediction layer of the CW model
    weighted_activations, activations = collect_data_CI_plots(concept_names, neurons_cwlayer, model_path, 'linear2',
                                                              class_of_interest)

    # get the activation values of the concept neurons
    activations_conceptneurons = activations[0:len(concept_names)]

    # get the activation values of all remaining neurons and sum them
    activations_remainingneurons = []
    for j in range(len(activations[0])):
        sum_act = 0
        for i in range(len(concept_names), len(activations)):
            sum_act += activations[i][j]
        activations_remainingneurons.append(sum_act)

    # rank the activations values of the different concept neurons
    ranked_activations = []
    for activations in activations_conceptneurons:
        ranked_activations = ranked_activations + list(rankdata(np.array(activations)))

    # rank the sum of the activation values of the remaining nodes and add them to all the ranking data
    ranked_activations = ranked_activations + list(rankdata(np.array(activations_remainingneurons)))

    # create a list of the concept names * number of data instances predicted "class of interest"
    concepts_column = []
    for concept in concept_names:
        concepts_column = concepts_column + (len(activations_conceptneurons[0]) * [concept])

    # append 'sum of remaining 70 nodes' in the concept column to name the remaining values
    concepts_column = concepts_column + (len(activations_remainingneurons) * [f'sum of remaining '
                                                                              f'{neurons_cwlayer - len(concept_names)} '
                                                                              f'nodes'])

    # collect the weighted activation values for the concept neurons
    weighted_activations_list = []
    for concept in range(len(concept_names)):
        weighted_activations_singleconcept = []
        for item in weighted_activations[concept:concept + 1][0]:
            weighted_activations_singleconcept.append(item.item())
        weighted_activations_list = weighted_activations_list + weighted_activations_singleconcept

    # collect the weighted activation values for the remaining neurons
    weighted_activations_remaining = []
    for i in range(len(weighted_activations[0:1][0])):
        sum_weighted_activations_remainingneurons = 0
        for j in range(len(concept_names), len(weighted_activations)):
            sum_weighted_activations_remainingneurons += weighted_activations[j:j + 1][0][i].item()
        weighted_activations_remaining.append(sum_weighted_activations_remainingneurons)
    weighted_activations_list = weighted_activations_list + weighted_activations_remaining

    # combine all data in a dictionary and convert to a dataframe
    dataframe = pd.DataFrame({'Influence concept through weighted activation': weighted_activations_list,
                              'Activations': ranked_activations, 'Concept': concepts_column}).sort_values('Concept')

    # STEP 2: create the actual plot
    fig, ax = plt.subplots()

    # set the style of the plot
    sns.set(style="ticks")
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')

    catplot = sns.catplot(data=dataframe, x='Influence concept through weighted activation', y='Concept',
                          hue='Activations', palette='flare', kind='strip', legend=False)

    catplot.fig.suptitle(f'Influence concept on predictions {class_of_interest} class', fontsize=12, fontweight='bold')
    catplot.set_yticklabels(y_ticklabels, fontsize=8)
    catplot.set_axis_labels('Influence concept through weighted activation', 'Concept', fontsize=9)
    catplot.set(xlim=xaxis_limits)

    # add a colorbar to the plot that shows the color scaling used for the original activation values
    norm = plt.Normalize(np.min(dataframe['Activations']), np.max(dataframe['Activations']))  # colormap normalizer
    cmap = sns.color_palette("flare", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(sm, ticks=[np.min(ranked_activations), np.max(ranked_activations)], shrink=0.75)
    colorbar.ax.set_yticklabels(['Low', 'High'], fontsize=8)
    colorbar.set_label('Original activation value', fontsize=9)

    # create a black line at the 0-point of the weighted activation axis to accentuate this point
    ax = catplot.axes[0]
    ax[0].axvline(0, color='black', alpha=0.5)

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'report_figures/concept_influence_figures/CI_shap_{class_of_interest}.svg', format='svg')
    plt.savefig(f'report_figures/concept_influence_figures/CI_shap_{class_of_interest}.png', format='png')
    plt.show()


def CI_barplot(concept_names, neurons_cwlayer, class_of_interest, x_ticklabels):
    """
    :param concept_names: names of concepts used in the CW model
    :param neurons_cwlayer: number of neurons in the CW layer
    :param class_of_interest: pre-miRNA class of interest for collecting data values for the plot
    :param x_ticklabels: pretty labels for the x-axis
    """
    # initiate the plot and style it by increasing the linewidth and coloring them black
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    sns.set(style="ticks")
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')

    # create the labels for the plot
    labels = concept_names + [f'Average influence of all {neurons_cwlayer} neurons', f'Min weighted activation '
                                                                                     f'{class_of_interest} class',
                              f'Max weighted activation {class_of_interest} class']

    # collect all required data, compute the means and add them to the plot
    if class_of_interest == 'positive':
        weighted_activations, activations = collect_data_CI_plots(concept_names, 72, model_path, 'linear2', 'positive')
        mean_weighted_activations, max_weighted_activations = mean_max_activations(weighted_activations)
        data = mean_weighted_activations[0:len(concept_names)][::-1] + [np.mean(mean_weighted_activations),
                                                                        np.min(mean_weighted_activations),
                                                                        np.max(mean_weighted_activations)]
        ax.bar(np.arange(len(labels)), data, 0.3, label='(Activation*Weight_class1 | pred=1)', zorder=3)
        ax.set_ylim([np.min(data) - 0.05, np.max(data) + 0.05])

    elif class_of_interest == 'negative':
        weighted_activations, activations = collect_data_CI_plots(concept_names, 72, model_path, 'linear2', 'negative')
        mean_weighted_activations, max_weighted_activations = mean_max_activations(weighted_activations)
        data = mean_weighted_activations[0:len(concept_names)][::-1] + [np.mean(mean_weighted_activations),
                                                                        np.min(mean_weighted_activations),
                                                                        np.max(mean_weighted_activations)]
        ax.bar(np.arange(len(labels)), data, 0.3, label='(Activation*Weight_class0 | pred=0)', zorder=3)
        ax.set_ylim([np.min(data) - 0.05, np.max(data) + 0.05])

    ax.set_ylabel('Weighted activation', fontsize=10)
    ax.set_xlabel('Concept', fontsize=10)
    ax.set_title(f'Average influence concept on predictions {class_of_interest} class', fontweight='bold')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(x_ticklabels, fontsize=9)

    # create a black line at the 0-point of the weighted activations to accentuate this point
    ax.axhline(0, color='black', alpha=0.5)

    fig.tight_layout()
    plt.savefig(f'report_figures/concept_influence_figures/CI_average_{class_of_interest}.svg', format='svg')
    plt.savefig(f'report_figures/concept_influence_figures/CI_average_{class_of_interest}.png', format='png')
    plt.show()


# path to CW model to be used when collecting activation values
model_path = 'checkpoints/large_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final_' \
             'twoouputs_3_foldn1_checkpoint.pth.tar'

SHAP_CI_plot(72, 'negative', ['large_asymmetric_bulge', 'base_pairs_wobbles_in_stem'],
             ['At least 90%\nbase pairs\nand wobbles\nin stem', 'Asymmetric\nbulge of\nwidth 15+\npixels',
              'Sum of\n70 remaining\nnodes'], [-10, 10])

CI_barplot(['large_asymmetric_bulge', 'base_pairs_wobbles_in_stem'], 72, 'positive',
           ['At least 90%\nbase pairs\nand wobbles\nin stem', 'Large\nasymmetric\nbulge',
            f'Average influence\n of all 72 neurons', f'Min influence\nof a single neuron',
            f'Max influence\nof a single neuron'])
