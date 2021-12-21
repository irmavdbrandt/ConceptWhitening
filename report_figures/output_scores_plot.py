import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def lineplot_acc_loss(data_info_list, x_var, y_var, x_label, y_label, x_lim, y_lim, title, fig_name):
    """
    NOTE: for this function to be used, one needs to download the output scores of the model stored on Neptune (the
    scores can be found in the Neptune web app)!


    :param data_info_list: list containing the path to the datafile, the columns to be used when reading this datafile,
    and the name used in the lineplot to define the data
    :param x_var: coordinates for x-axis of plot
    :param y_var: coordinates for y-axis of plot
    :param x_label: label used for x-axis
    :param y_label: label used for y-axis
    :param x_lim: limits of the x-axis (i.e., list with [min, max])
    :param y_lim: limits of the y-axis (i.e., list with [min, max])
    :param title: title of the figure
    :param fig_name: name used to store the figure
    """
    sns.set_style('whitegrid')

    ax = plt.axes()

    for item in data_info_list:
        data = pd.read_csv(item[0], header=None, names=item[1])
        ax.plot(data[x_var], data[y_var], label=item[2])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.legend()
    plt.title(title, fontsize=15, fontweight='bold')

    plt.savefig(f'report_figures/output_scores_figures/{fig_name}.svg', format='svg')
    plt.show()


data_info_finetuning_accuracies = [('report_figures/neptune_output/base_model_finetuning/training_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Training accuracy'),
                                   ('report_figures/neptune_output/base_model_finetuning/validation_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Test accuracy')]
lineplot_acc_loss(data_info_finetuning_accuracies, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 100], [85, 100],
                  'Fine-tuning classification accuracy', 'finetuning_accuracies')

data_info_finetuning_losses = [('report_figures/neptune_output/base_model_finetuning/training_loss.csv',
                                ['epoch', '??', 'loss'], 'Training loss'),
                               ('report_figures/neptune_output/base_model_finetuning/validation_loss.csv',
                                ['epoch', '??', 'loss'], 'Test loss')]
lineplot_acc_loss(data_info_finetuning_losses, 'epoch', 'loss', 'Epoch', 'Loss', [0, 100], [0, 0.3],
                  'Fine-tuning cross-entropy loss', 'finetuning_loss')

data_info_6concepts_cv_training = [('report_figures/neptune_output/CW_model_6concepts/training_0_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 1'),
                                   ('report_figures/neptune_output/CW_model_6concepts/training_1_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 2'),
                                   ('report_figures/neptune_output/CW_model_6concepts/training_2_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 3'),
                                   ('report_figures/neptune_output/CW_model_6concepts/training_3_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 4'),
                                   ('report_figures/neptune_output/CW_model_6concepts/training_4_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 5')]

lineplot_acc_loss(data_info_6concepts_cv_training, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 45], [50, 100],
                  'Training accuracies CW model with 6 concepts', 'train_accuracies_CWmodel_6concepts')

data_info_6concepts_cv_validation = [('report_figures/neptune_output/CW_model_6concepts/validation_0_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 1'),
                                     ('report_figures/neptune_output/CW_model_6concepts/validation_1_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 2'),
                                     ('report_figures/neptune_output/CW_model_6concepts/validation_2_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 3'),
                                     ('report_figures/neptune_output/CW_model_6concepts/validation_3_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 4'),
                                     ('report_figures/neptune_output/CW_model_6concepts/validation_4_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 5')]

lineplot_acc_loss(data_info_6concepts_cv_validation, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 45],
                  [50, 100],
                  'Validation accuracies CW model with 6 concepts', 'val_accuracies_CWmodel_6concepts')

data_info_2concepts_cv_training = [('report_figures/neptune_output/CW_model_2concepts/training_0_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 1'),
                                   ('report_figures/neptune_output/CW_model_2concepts/training_1_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 2'),
                                   ('report_figures/neptune_output/CW_model_2concepts/training_2_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 3'),
                                   ('report_figures/neptune_output/CW_model_2concepts/training_3_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 4'),
                                   ('report_figures/neptune_output/CW_model_2concepts/training_4_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 5')]

lineplot_acc_loss(data_info_2concepts_cv_training, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 100], [60, 100],
                  'Training accuracies CW model with 2 concepts', 'train_accuracies_CWmodel_2concepts')

data_info_2concepts_cv_validation = [('report_figures/neptune_output/CW_model_2concepts/validation_0_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 1'),
                                     ('report_figures/neptune_output/CW_model_2concepts/validation_1_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 2'),
                                     ('report_figures/neptune_output/CW_model_2concepts/validation_2_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 3'),
                                     ('report_figures/neptune_output/CW_model_2concepts/validation_3_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 4'),
                                     ('report_figures/neptune_output/CW_model_2concepts/validation_4_acc.csv',
                                      ['epoch', '??', 'accuracy'], 'Fold 5')]

lineplot_acc_loss(data_info_2concepts_cv_validation, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 100],
                  [60, 100],
                  'Validation accuracies CW model with 2 concepts', 'val_accuracies_CWmodel_2concepts')

data_info_1output_cv_training = [('report_figures/neptune_output/CW_model_oneoutput/training_0_acc.csv',
                                  ['epoch', '??', 'accuracy'], 'Fold 1'),
                                 ('report_figures/neptune_output/CW_model_oneoutput/training_1_acc.csv',
                                  ['epoch', '??', 'accuracy'], 'Fold 2'),
                                 ('report_figures/neptune_output/CW_model_oneoutput/training_2_acc.csv',
                                  ['epoch', '??', 'accuracy'], 'Fold 3'),
                                 ('report_figures/neptune_output/CW_model_oneoutput/training_3_acc.csv',
                                  ['epoch', '??', 'accuracy'], 'Fold 4'),
                                 ('report_figures/neptune_output/CW_model_oneoutput/training_4_acc.csv',
                                  ['epoch', '??', 'accuracy'], 'Fold 5')]

lineplot_acc_loss(data_info_1output_cv_training, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 30], [40, 100],
                  'Training accuracies CW model with 1 predictions_test node', 'train_accuracies_CWmodel_1output')

data_info_1output_cv_validation = [('report_figures/neptune_output/CW_model_oneoutput/validation_0_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 1'),
                                   ('report_figures/neptune_output/CW_model_oneoutput/validation_1_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 2'),
                                   ('report_figures/neptune_output/CW_model_oneoutput/validation_2_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 3'),
                                   ('report_figures/neptune_output/CW_model_oneoutput/validation_3_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 4'),
                                   ('report_figures/neptune_output/CW_model_oneoutput/validation_4_acc.csv',
                                    ['epoch', '??', 'accuracy'], 'Fold 5')]

lineplot_acc_loss(data_info_1output_cv_validation, 'epoch', 'accuracy', 'Epoch', 'Accuracy (in %)', [0, 30], [40, 100],
                  'Validation accuracies CW model with 1 predictions_test node', 'val_accuracies_CWmodel_1output')
