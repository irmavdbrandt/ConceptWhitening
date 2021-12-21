from PIL import Image
import os
import numpy as np
import pandas as pd
from preparation.concept_detection import loop_concepts, ugu_motif, pairs_stem, palindrome, \
    AU_pairs_begin_maturemiRNA, large_asymmetric_bulge
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt


# STEP 1: collect the data and concept annotations
def create_annotated_df(data_path):
    """
    :param data_path: path to storage location of dataframe
    :return: dataframe containing paths to all images, the class label associated to the image and information on the
    sequence concepts
    """
    # initialize an empty list that will store all data entries
    tables = []

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            # mac-specific error in stored dataset file
            if filename == ".DS_Store":
                continue
            # create a dataframe entry based on the filepath of the image in the dataset folder
            entry = pd.DataFrame([os.path.join(dirname, filename)], columns=['path'])
            # define the class label and set (train or test) of the entry
            if "train" in dirname:
                # the class label can be found right after the set specification in the dirname
                entry['class_label'] = dirname.split('train/', maxsplit=1)[-1].split(maxsplit=1)[0][0]
                entry['set'] = "train"
            elif "concept_test" in dirname:
                # in case there is a concept dataset in the directory already, ignore it
                continue
            else:
                # the class label can be found right after the set specification in the dirname
                entry['class_label'] = dirname.split('test/', maxsplit=1)[-1].split(maxsplit=1)[0][0]
                entry['set'] = "test"
            # read image from filename
            image = Image.open(os.path.join(dirname, filename))
            # convert image to numpy array
            data = np.array(image)
            # generate all the concept information
            sequence_pairs, stem_begin = pairs_stem(data)
            terminal_loop, loop_start_pixel, loop_highest_row, loop_highest_pixel, loop_length, \
                loop_width, width_gap_start = loop_concepts(data, sequence_pairs)
            ugu_motif_present = ugu_motif(data, terminal_loop, loop_highest_pixel, loop_start_pixel)
            au_pair = AU_pairs_begin_maturemiRNA(sequence_pairs)
            palindrome_score, upper_half_counts, lower_half_counts, len_premiRNA = palindrome(data)
            largest_bulge, largest_bulge_location = large_asymmetric_bulge(data)
            # start adding the concept information to the dataframe entry
            entry['presence_terminal_loop'] = terminal_loop
            entry['start_loop_upperhalf_col'] = loop_start_pixel
            entry['highest_point_loop_upperhalf_row'] = loop_highest_row
            entry['highest_point_loop_upperhalf_col'] = loop_highest_pixel
            entry['loop_length'] = loop_length
            entry['loop_width'] = loop_width
            entry['gap_start'] = width_gap_start
            entry['palindrome_score'] = palindrome_score[1]
            if palindrome_score[0] > 0.6:
                entry['asymmetric'] = True
            else:
                entry['asymmetric'] = False
            entry['large_asymmetric_bulge'] = largest_bulge
            entry['largest_asym_bulge_strand_location'] = largest_bulge_location[0]
            entry['largest_asym_bulge_sequence_location'] = largest_bulge_location[1]
            entry['stem_begin'] = stem_begin
            if np.isnan(loop_start_pixel):
                entry['stem_end'] = 0
                entry['stem_length'] = stem_begin
                entry['total_length'] = stem_begin
                loop_start_pixel = 0
            else:
                entry['stem_end'] = loop_start_pixel + 1
                entry['stem_length'] = stem_begin - loop_width
                entry['total_length'] = stem_begin
            # the stem pairs are defined by the pairs in the pairs array until the start of the terminal loop
            stem_pairs = sequence_pairs[0:len(sequence_pairs) - loop_start_pixel]
            # base pairing propensity: # base pairs (defined by 1 and 2) / stem length
            entry['base_pairs_in_stem'] = (stem_pairs.count(1) + stem_pairs.count(2)) / (stem_begin - loop_start_pixel)
            # base pairing and wobble propensity: # base pairs (defined by 1 and 2) + wobbles (3) / stem length
            entry['base_pairs_wobbles_in_stem'] = (stem_pairs.count(1) + stem_pairs.count(2) + stem_pairs.count(3)) / \
                                                  (stem_begin - loop_start_pixel)
            entry['AU_pair_begin_maturemiRNA'] = au_pair
            entry['UGU'] = ugu_motif_present

            # after collecting all concept info for the entry of interest, append it to list containing all data entries
            tables.append(entry)

    # combine all data entries into one dataframe
    dataframe = pd.concat(tables, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class_label'].value_counts())

    return dataframe


data_tables = create_annotated_df(data_path='datasets/modhsa_original/original_dataset/')


# STEP 2: Collect some summary statistics on the concepts of interest
def calculate_summary_stats(dataframe, label_col, col_of_interest):
    """
    :param dataframe: dataframe containing concept annotations
    :param label_col: column with class label
    :param col_of_interest: concept column of which summary statistics are needed
    :return: summary statistics for concept of interest per class label
    """
    data_stats = dataframe[[label_col, col_of_interest]]
    data_stats = data_stats.groupby(label_col).describe().unstack(1).reset_index().pivot(index=label_col, values=0,
                                                                                         columns='level_1')
    print(f'Summary stats for {col_of_interest}: ', data_stats)
    return data_stats


print("General info")
length_stats = calculate_summary_stats(data_tables, 'class_label', 'total_length')
stem_length_stats = calculate_summary_stats(data_tables, 'class_label', 'stem_length')
stem_begin_stats = calculate_summary_stats(data_tables, 'class_label', 'stem_begin')

print("---------------")
print("---------------")
print("---------------")
print("Loop info")
loop_stats = calculate_summary_stats(data_tables, 'class_label', 'presence_terminal_loop')
loop_length_stats = calculate_summary_stats(data_tables, 'class_label', 'loop_length')
loop_width_stats = calculate_summary_stats(data_tables, 'class_label', 'loop_width')
gap_stats = calculate_summary_stats(data_tables, 'class_label', 'gap_start')

print("---------------")
print("---------------")
print("---------------")
print("Motif info")
UGU_stats = calculate_summary_stats(data_tables, 'class_label', 'UGU')
AU_pair_stats = calculate_summary_stats(data_tables, 'class_label', 'AU_pair_begin_maturemiRNA')

print("---------------")
print("---------------")
print("---------------")
print("Symmetry/Asymmetry info")
palindrome_stats = calculate_summary_stats(data_tables, 'class_label', 'palindrome_score')
asymmetric_premirna_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric')

print("---------------")
print("---------------")
print("---------------")
print("Base pairing info")
base_pairs_prop_stem_stats = calculate_summary_stats(data_tables, 'class_label', 'base_pairs_in_stem')
base_pairs_wobbles_prop_stem_stats = calculate_summary_stats(data_tables, 'class_label', 'base_pairs_wobbles_in_stem')

print("---------------")
print("---------------")
print("---------------")
print("Bulge info: large asymmetric bulge")
large_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'large_asymmetric_bulge')
strand_location_large_bulge_stats = calculate_summary_stats(data_tables, 'class_label',
                                                            'largest_asym_bulge_strand_location')
seq_location_large_bulge_stats = calculate_summary_stats(data_tables, 'class_label',
                                                         'largest_asym_bulge_sequence_location')


# %%
# STEP 3: Store the images and concept example images in the (5-fold cv) directories as defined in the original
# Concept Whitening repo

def make_dir_if_not_exists(dir_path):
    """
    :param dir_path: name of path where new directory has to be made
    """
    if not tf.io.gfile.exists(dir_path):
        tf.io.gfile.makedirs(dir_path)


def img_saver(dataframe, dir_path, cv_data):
    """
    :param dataframe: dataframe with images that need to be saved
    :param dir_path: path to directory where images need to be saved
    :param cv_data: boolean specifying whether the data is used for cross-validation
    """

    for idx, df_row in dataframe.iterrows():
        image_path = df_row['path']
        img = Image.open(image_path)
        if df_row['class_label'] == "0":
            # the image name starts after the class label specification in the image path
            img_name = image_path.split('0/', maxsplit=1)[-1].split(maxsplit=1)[0]
            dir_path_new = dir_path
        else:
            # the image name starts after the class label specification in the image path
            img_name = image_path.split('1/', maxsplit=1)[-1].split(maxsplit=1)[0]
            dir_path_new = dir_path
        if cv_data:
            label = df_row['class_label']
            # for the cross-validation data, we still need to make directories based on the class label so we adjust
            # the dir_path variable
            dir_path_new = dir_path + '/' + label
        img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
        make_dir_if_not_exists(dir_path_new)
        img.save(f'{dir_path_new}/{img_name}', 'PNG')


def save_concepts_img(binary_concepts_list, non_binary_concepts_list, dataframe, fold_n, train):
    """
    :param binary_concepts_list: list of binary concepts of interest
    :param non_binary_concepts_list: list of non-binary concepts of interest, where the concepts are combined with the
        threshold value that is used to select concepts
    :param dataframe: dataframe containing images where concepts example images are taken from
    :param fold_n: the current fold number (in case of cross-validation)
    :param train: boolean specifying whether the concept images belong to the training set or not
    """
    for concept in binary_concepts_list:
        target_dataframe = dataframe.loc[dataframe[concept] == True]

        print(f"Number of data instances including concept {concept}: {len(target_dataframe)}")
        calculate_summary_stats(target_dataframe, 'class_label', concept)

        if train:
            img_saver(target_dataframe, f'./datasets/modhsa_original/CW_dataset/concept_train_fold{fold_n}/{concept}/'
                                        f'{concept}', False)
        else:
            img_saver(target_dataframe, f'./datasets/modhsa_original/CW_dataset/concept_test/{concept}/{concept}',
                      False)

    for i in range(len(non_binary_concepts_list)):
        target_dataframe = dataframe.loc[dataframe[non_binary_concepts_list[i][0]] >= non_binary_concepts_list[i][1]]

        print(f"Number of data instances including concept {non_binary_concepts_list[i]}: {len(target_dataframe)}")
        calculate_summary_stats(target_dataframe, 'class_label', non_binary_concepts_list[i][0])

        if train:
            img_saver(target_dataframe, f'./datasets/modhsa_original/CW_dataset/concept_train_fold{fold_n}/'
                                        f'{non_binary_concepts_list[i][0]}/{non_binary_concepts_list[i][0]}', False)
        else:
            img_saver(target_dataframe, f'./datasets/modhsa_original/CW_dataset/concept_test/'
                                        f'{non_binary_concepts_list[i][0]}/{non_binary_concepts_list[i][0]}', False)


# here we specify the concepts of interest. In case of concepts with non-binary values, we should include which value
# is used as threshold for defining the concept.
# Example: we want terminal loops with a length >= 10: we add ('loop_length', 10) to the list of non-binary concepts
binary_concepts_of_interest = ['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA']
non_binary_concepts_of_interest = [('loop_length', 21), ('loop_width', 12), ('gap_start', 10),
                                   ('base_pairs_wobbles_in_stem', 0.9), ('large_asymmetric_bulge', 15)]

# STEP 3.1: Creating the training set: split the complete training set in 5 folds for cross validation. We collect
# concept example images for training based on the training folds.
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
train_data_tables = data_tables.loc[data_tables['set'] == 'train']
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data_tables, train_data_tables['class_label'])):
    # collect the train fold based on the indices assigned to the training set
    train_set = train_data_tables.iloc[train_ids]
    print(f'Number of instances in the training fold: {len(train_set)}')
    img_saver(train_set, f'./datasets/modhsa_original/CW_dataset/train_fold{fold}', True)
    save_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, train_set, fold, True)

    # collect the validation fold based on the indices assigned to the validation set
    val_set = train_data_tables.iloc[val_ids]
    print(f'Number of instances in the validation fold: {len(val_set)}')
    img_saver(val_set, f'./datasets/modhsa_original/CW_dataset/val_fold{fold}', True)


# STEP 3.2: Creating the test set: no cross validation required so we use the complete original test set and
# collect concept examples images from them
test_data_tables = data_tables.loc[data_tables['set'] == 'test']
save_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, test_data_tables, None, False)

#%%
# STEP 4: Calculate and visualize the correlation between the different concepts

# STEP 4.1: Calculate the correlations values for the base concepts (i.e., without using thresholds for specification
# of non-binary concepts)
binary_concepts = data_tables[binary_concepts_of_interest]
binary_concepts = binary_concepts.fillna(False)

non_binary_concepts = data_tables[[i[0] for i in non_binary_concepts_of_interest]]
# manually delete the concept columns that are not needed in the list
non_binary_concepts = non_binary_concepts.drop(columns=['loop_length', 'loop_width'])

concept_data = pd.concat([binary_concepts, non_binary_concepts], axis=1, join="inner")
# set the order of the columns in the dataframe to match the order of the concepts in the thesis report
concept_data = concept_data[['presence_terminal_loop', 'base_pairs_wobbles_in_stem', 'gap_start',
                             'large_asymmetric_bulge', 'UGU', 'AU_pair_begin_maturemiRNA']]


def concept_corr_matrix(dataframe, min_value, max_value, x_labels, y_labels, title, concept_type):
    """
    :param dataframe: dataframe with data annotated concepts of interest
    :param min_value: minimum value allowed in the correlation matrix
    :param max_value: maximum value allowed in the correlation matrix
    :param x_labels: pretty labels for x-ticks
    :param y_labels: pretty labels for y-ticks
    :param title: title of plot
    :param concept_type: string specifying whether the type of the concepts (e.g. preliminary, final,..) which
    # is used for saving the plot
    """
    # calculate the correlation between the different concepts
    concept_correlation = round(dataframe.corr(), 2)

    sns.set(font_scale=1)
    sns.heatmap(concept_correlation, vmin=min_value, vmax=max_value, xticklabels=x_labels, yticklabels=y_labels,
                annot=True, cmap='Oranges', annot_kws={"fontsize": 12})
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    # save the plot as svg file for better quality in report
    plt.savefig(f'report_figures/concept_correlation_figures/corr_matrix_{concept_type}concepts.svg', format='svg')
    plt.savefig(f'report_figures/concept_correlation_figures/corr_matrix_{concept_type}concepts.png', format='png')
    plt.show()


xticklabels = ['Terminal\nloop', 'Frequency\npairs\nand\nwobbles\nin\nstem',
               'Asymmetric\nbulge\ninstead\nof\nterminal\n loop', 'High\nasymmetric\nbulge', 'U-G-U\nmotif',
               'A-U\nmotif']
yticklabels = ['Terminal loop', 'Frequency\nbase pairs and\nwobbles in stem',
               'Asymmetric\nbulge instead\nof terminal loop', 'High\nasymmetric\nbulge', 'U-G-U motif', 'A-U motif']

concept_corr_matrix(concept_data, -1, 1, xticklabels, yticklabels, 'Correlation between preliminary concepts',
                    'preliminary')


# %%
# STEP 4.2: Calculate the correlations values for the final concepts (i.e., using thresholds for specification
# of non-binary concepts)
concepts = binary_concepts_of_interest + [i[0] for i in non_binary_concepts_of_interest]
concept_data_specificconcepts = data_tables[concepts]

concept_data_specificconcepts_vfinal = pd.DataFrame(columns=concepts)
new_rows = []
for index, row in concept_data_specificconcepts.iterrows():
    if row['loop_length'] >= 21 and row['loop_width'] >= 12:
        row['presence_terminal_loop'] = True
    else:
        row['presence_terminal_loop'] = False

    if row['gap_start'] >= 10:
        row['gap_start'] = True
    else:
        row['gap_start'] = False

    if row['base_pairs_wobbles_in_stem'] >= 0.9:
        row['base_pairs_wobbles_in_stem'] = True
    else:
        row['base_pairs_wobbles_in_stem'] = False

    if row['large_asymmetric_bulge'] >= 15:
        row['large_asymmetric_bulge'] = True
    else:
        row['large_asymmetric_bulge'] = False

    new_rows.append(row)

concept_data_specificconcepts_vfinal = concept_data_specificconcepts_vfinal.append(
    pd.DataFrame(new_rows, columns=concept_data_specificconcepts_vfinal.columns)).reset_index(drop=True)
concept_data_specificconcepts_vfinal = concept_data_specificconcepts_vfinal[['presence_terminal_loop',
                                                                             'base_pairs_wobbles_in_stem', 'gap_start',
                                                                             'large_asymmetric_bulge', 'UGU',
                                                                             'AU_pair_begin_maturemiRNA']]
# convert all Trues to 1 and all Falses to 0
concept_data_specificconcepts_vfinal = concept_data_specificconcepts_vfinal.astype(int)

xticklabels = ['Large\nterminal\nloop',
               'At least\n90% base\npairs and\nwobbles in\nstem',
               'Large\nasymmetric\nbulge\ninstead\nof terminal\nloop',
               'Large\nasymmetric\nbulge',
               'U-G-U\nmotif',
               'A-U\npairs\nmotif']
yticklabels = ['Large terminal loop',
               'At least 90%\nbase pairs and\nwobbles in stem',
               'Large asymmetric\nbulge instead\nof terminal loop',
               'Large asymmetric bulge',
               'U-G-U motif', 'A-U pairs motif']

concept_corr_matrix(concept_data_specificconcepts_vfinal, -1, 1, xticklabels, yticklabels,
                    'Correlation between specific pre-miRNA concepts', 'specific')
