from PIL import Image
import os
import numpy as np
import pandas as pd
from preparation.concept_detection import loop_concepts, ugu_motif, base_pairs_stem, bulges, palindrome, \
    AU_pairs_begin_maturemiRNA, large_asymmetric_bulge
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# STEP 1: collect the data and concept annotations

tables = []
symmetric_bulge_tables_class0 = []
symmetric_bulge_tables_class1 = []
asymmetric_bulge_tables_class0 = []
asymmetric_bulge_tables_class1 = []


def create_annotated_df():
    for dirname, _, filenames in os.walk('modhsa_original/original_dataset/'):
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            entry = pd.DataFrame([os.path.join(dirname, filename)], columns=['path'])
            if dirname[33:38] == "train":
                entry['class_label'] = dirname[39:40]
                entry['set'] = dirname[33:38]
            else:
                entry['class_label'] = dirname[38:39]
                entry['set'] = dirname[33:37]
            # add info on concepts
            image = Image.open(os.path.join(dirname, filename))
            # convert image to numpy array
            data = np.array(image)
            # generate all the concept information
            base_pairs, stem_begin = base_pairs_stem(data)
            terminal_loop, loop_start_row, loop_start_pixel, loop_highest_row, loop_highest_pixel, loop_length, \
            loop_width, round_loop, width_gap_start = loop_concepts(data, base_pairs)
            ugu_combined = ugu_motif(data, terminal_loop, loop_highest_pixel, loop_start_pixel)
            palindrome_score, palindrome_loop, largest_bulge, largest_bulge_color, largest_bulge_location = \
                palindrome(data, loop_start_pixel)
            au_pair = AU_pairs_begin_maturemiRNA(base_pairs)
            asymmetry = large_asymmetric_bulge(data, loop_start_pixel)
            # start adding the concept information to the entry of the dataframe
            entry['presence_terminal_loop'] = terminal_loop
            entry['start_loop_upperhalf_row'] = loop_start_row
            entry['start_loop_upperhalf_col'] = loop_start_pixel
            entry['highest_point_loop_upperhalf_row'] = loop_highest_row
            entry['highest_point_loop_upperhalf_col'] = loop_highest_pixel
            entry['loop_length'] = loop_length
            entry['loop_width'] = loop_width
            entry['round_loop'] = round_loop
            entry['stem_begin'] = stem_begin
            entry['gap_start'] = width_gap_start
            entry['asymmetric'] = asymmetry
            entry['large_asymmetric_bulge'] = largest_bulge
            if np.isnan(largest_bulge):
                entry['largest_asym_bulge_strand_location'] = np.nan
                entry['largest_asym_bulge_sequence_location'] = np.nan
            else:
                colors = ['red', 'blue', 'green', 'yellow']
                for color in colors:
                    if color in largest_bulge_color:
                        entry['most_occurring_color_largest_asym_bulge'] = color
                entry['largest_asym_bulge_strand_location'] = largest_bulge_location[0]
                entry['largest_asym_bulge_sequence_location'] = largest_bulge_location[1]

            entry['palindrome_True'] = palindrome_score[1]
            entry['palindrome_loop'] = palindrome_loop
            entry['AU_pair_begin_maturemiRNA'] = au_pair
            if np.isnan(loop_start_pixel):
                entry['stem_end'] = 0
                entry['stem_length'] = stem_begin
                entry['total_length'] = stem_begin
            else:
                entry['stem_end'] = loop_start_pixel + 1
                entry['stem_length'] = stem_begin - loop_width
                entry['total_length'] = stem_begin
            if (base_pairs[0:4].count(1) + base_pairs[0:4].count(2)) == 4:
                entry['base_stem_1nt_4nt'] = True
            else:
                entry['base_stem_1nt_4nt'] = False
            if (base_pairs[0:4].count(1) + base_pairs[0:4].count(2) + base_pairs[0:4].count(3)) == 4:
                entry['base_wobble_stem_1nt_4nt'] = True
            else:
                entry['base_wobble_stem_1nt_4nt'] = False
            if base_pairs[0:4].count(0) != 0:
                entry['gap_stem_1nt_4nt'] = True
            else:
                entry['gap_stem_1nt_4nt'] = False
            if (base_pairs[4:8].count(1)) + (base_pairs[4:8].count(2)) == 4:
                entry['base_stem_4nt_8nt'] = True
            else:
                entry['base_stem_4nt_8nt'] = False
            if (base_pairs[0:8].count(1)) + (base_pairs[0:8].count(2)) == 8:
                entry['base_stem_1nt_8nt'] = True
            else:
                entry['base_stem_1nt_8nt'] = False
            if (base_pairs[0:8].count(1) + base_pairs[0:8].count(2) + base_pairs[0:8].count(3)) == 8:
                entry['base_wobble_stem_1nt_8nt'] = True
            else:
                entry['base_wobble_stem_1nt_8nt'] = False
            if base_pairs[0:8].count(0) != 0:
                entry['gap_stem_1nt_8nt'] = True
            else:
                entry['gap_stem_1nt_8nt'] = False
            base_pair_successions = 0
            for i in range(len(base_pairs) - 2):
                if ((base_pairs[i] == 1) and (base_pairs[i + 1] == 1)) or \
                        ((base_pairs[i] == 1) and (base_pairs[i + 1] == 2)) or \
                        ((base_pairs[i] == 2) and (base_pairs[i + 1] == 1)) or \
                        ((base_pairs[i] == 2) and (base_pairs[i + 1] == 2)):
                    if (base_pairs[i + 2] == 0) or (base_pairs[i + 2] == 3):
                        base_pair_successions += 1
                    else:
                        if i == len(base_pairs) - 3:
                            base_pair_successions += 1
                        continue
            if base_pair_successions == 0:
                entry['base_pair_successions_presence'] = False
            else:
                entry['base_pair_successions_presence'] = True
            if base_pair_successions > 4:
                entry['5+_base_pair_successions'] = True
            else:
                entry['5+_base_pair_successions'] = False
            entry['base_pair_successions'] = base_pair_successions

            if np.isnan(loop_start_pixel):
                loop_start_pixel = 0
            base_pairs_stem_array = base_pairs[0:len(base_pairs) - loop_start_pixel]
            # base pairing propensity: # base pairs / stem length
            entry['base_pairs_in_stem'] = (base_pairs_stem_array.count(1) + base_pairs_stem_array.count(2)) / \
                                          (stem_begin - loop_start_pixel)
            entry['base_pairs_wobbles_in_stem'] = (base_pairs_stem_array.count(1) + base_pairs_stem_array.count(2) +
                                                   base_pairs_stem_array.count(3)) / \
                                                  (stem_begin - loop_start_pixel)
            entry['GC_CG_in_stem'] = base_pairs_stem_array.count(1) / (stem_begin - loop_start_pixel)
            entry['AU_UA_in_stem'] = base_pairs_stem_array.count(2) / (stem_begin - loop_start_pixel)

            # UGU-motif
            # we want the first occurrence of the combined nucleotides, so the motif with the smallest pixel as
            # starting point is assumed to be the motif of interest
            ugu_start = 100
            for i in range(len(ugu_combined)):
                if len(ugu_combined[i]) > 1:
                    if ugu_combined[i][0] < ugu_start:
                        entry['UGU'] = True
                        entry['UGU_start'] = ugu_combined[i][0]
                        entry['UGU_end'] = ugu_combined[i][2]
                        ugu_start = ugu_combined[i][0]
                    else:
                        print('motif but not first')
                else:
                    continue
            tables.append(entry)

    # collect the data and concept information
    data_tables = pd.concat(tables, ignore_index=True)  # create dataframe from list of tables and reset index
    print(data_tables['class_label'].value_counts())

    return data_tables


data_tables = create_annotated_df()

# %%
data_tables = data_tables.loc[data_tables['class_label'] != 'p']


# %%
def calculate_summary_stats(original_data, label_col, col_of_interest):
    """
    :param original_data: dataframe containing concept annotations
    :param label_col: column with class label
    :param col_of_interest: concept column of which summary statistics are needed
    :return: summary statistics for concept of interest per class label
    """
    data_stats = original_data[[label_col, col_of_interest]]
    data_stats = data_stats.groupby(label_col).describe().unstack(1).reset_index().pivot(index=label_col, values=0,
                                                                                         columns='level_1')
    print(f'Summary stats for {col_of_interest}: ', data_stats)
    return data_stats


print("General info")
length_stats = calculate_summary_stats(data_tables, 'class_label', 'total_length')
# todo: add length, maybe a bit weird with all the asymmetric bulges..?
stem_length_stats = calculate_summary_stats(data_tables, 'class_label', 'stem_length')
stem_begin_stats = calculate_summary_stats(data_tables, 'class_label', 'stem_begin')

print("---------------")
print("---------------")
print("---------------")
print("Loop info")
loop_stats = calculate_summary_stats(data_tables, 'class_label', 'presence_terminal_loop')
loop_length_stats = calculate_summary_stats(data_tables, 'class_label', 'loop_length')
loop_width_stats = calculate_summary_stats(data_tables, 'class_label', 'loop_width')
loop_structure_stats = calculate_summary_stats(data_tables, 'class_label', 'palindrome_loop')
loop_shape_stats = calculate_summary_stats(data_tables, 'class_label', 'round_loop')
gap_stats = calculate_summary_stats(data_tables, 'class_label', 'gap_start')

print("---------------")
print("---------------")
print("---------------")
print("Motif info")
UGU_stats = calculate_summary_stats(data_tables, 'class_label', 'UGU')
AU_pair_stats = calculate_summary_stats(data_tables, 'class_label', 'AU_pair_begin_maturemiRNA')
palindrome_stats = calculate_summary_stats(data_tables, 'class_label', 'palindrome_True')
asymmetric_premirna_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric')

print("---------------")
print("---------------")
print("---------------")
print("Base pairing info")
base_pairs_prop_stem_stats = calculate_summary_stats(data_tables, 'class_label', 'base_pairs_in_stem')
base_pairs_wobbles_prop_stem_stats = calculate_summary_stats(data_tables, 'class_label', 'base_pairs_wobbles_in_stem')
base_pair_succ_pres_stats = calculate_summary_stats(data_tables, 'class_label', 'base_pair_successions_presence')
base_5more_pair_succ_pres_stats = calculate_summary_stats(data_tables, 'class_label', '5+_base_pair_successions')
base_pair_succ_data = calculate_summary_stats(data_tables, 'class_label', 'base_pair_successions')
clean_4nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'base_stem_1nt_4nt')
error_4nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'gap_stem_1nt_4nt')
base_wobble_4nt_stats = calculate_summary_stats(data_tables, 'class_label', 'base_wobble_stem_1nt_4nt')
base_48nt_stats = calculate_summary_stats(data_tables, 'class_label', 'base_stem_4nt_8nt')
clean_8nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'base_stem_1nt_8nt')
error_8nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'gap_stem_1nt_8nt')
base_wobble_8nt_stats = calculate_summary_stats(data_tables, 'class_label', 'base_wobble_stem_1nt_8nt')
AU_UA_in_stem = calculate_summary_stats(data_tables, 'class_label', 'AU_UA_in_stem')
GC_CG_in_stem = calculate_summary_stats(data_tables, 'class_label', 'GC_CG_in_stem')

# print("---------------")
# print("---------------")
# print("---------------")
# print("Bulge info: symmetric")
# symmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'symmetric_bulge_presence')
# symmetric_bulge_count_stats = calculate_summary_stats(data_tables, 'class_label', 'symmetric_bulges_count')
# widths = []
# heights = []
# width_height_combi = []
# for bulge in symmetric_bulge_tables_class0:
#     widths.append(bulge[1][1])
#     heights.append(bulge[2][1])
#     width_height_combi.append(bulge[3][1])
# widths_values, width_counts = np.unique(widths, return_counts=True)
# heights_values, heights_counts = np.unique(heights, return_counts=True)
# width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
# print('stats symmetric bulge widths class 0: ', widths_values, width_counts)
# print('stats symmetric bulge heights class 0: ', heights_values, heights_counts)
# print('stats symmetric bulge width_height class 0: ', width_height_values, width_height_counts)
#
# widths = []
# heights = []
# width_height_combi = []
# for bulge in symmetric_bulge_tables_class1:
#     widths.append(bulge[1][1])
#     heights.append(bulge[2][1])
#     width_height_combi.append(bulge[3][1])
# widths_values, width_counts = np.unique(widths, return_counts=True)
# heights_values, heights_counts = np.unique(heights, return_counts=True)
# width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
# print('stats symmetric bulge widths class 1: ', widths_values, width_counts)
# print('stats symmetric bulge heights class 1: ', heights_values, heights_counts)
# print('stats symmetric bulge width_height class 1: ', width_height_values, width_height_counts)

print("---------------")
print("---------------")
print("---------------")
print("Bulge info: asymmetric")
large_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'large_asymmetric_bulge')
color_large_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'most_occurring_color_largest_asym_bulge')
strand_location_large_bulge_stats = calculate_summary_stats(data_tables, 'class_label',
                                                            'largest_asym_bulge_strand_location')
seq_location_large_bulge_stats = calculate_summary_stats(data_tables, 'class_label',
                                                         'largest_asym_bulge_sequence_location')


# asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric_bulge_presence')
asymmetric_bulge_count_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric_bulges_count')
high_asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'high_asymmetric_bulge')
wide_asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'wide_asymmetric_bulge')
large_asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'large_asymmetric_bulge')

widths = []
heights = []
width_height_combi = []
for bulge in asymmetric_bulge_tables_class0:
    widths.append(bulge[1][1])
    heights.append(bulge[2][1])
    width_height_combi.append(bulge[3][1])
widths_values, width_counts = np.unique(widths, return_counts=True)
heights_values, heights_counts = np.unique(heights, return_counts=True)
width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
print('stats asymmetric bulge widths class 0: ', widths_values, width_counts)
print('stats asymmetric bulge heights class 0: ', heights_values, heights_counts)
print('stats asymmetric bulge width_height class 0: ', width_height_values, width_height_counts)

widths = []
heights = []
width_height_combi = []
for bulge in asymmetric_bulge_tables_class1:
    widths.append(bulge[1][1])
    heights.append(bulge[2][1])
    width_height_combi.append(bulge[3][1])
widths_values, width_counts = np.unique(widths, return_counts=True)
heights_values, heights_counts = np.unique(heights, return_counts=True)
width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
print('stats asymmetric bulge widths class 1: ', widths_values, width_counts)
print('stats asymmetric bulge heights class 1: ', heights_values, heights_counts)
print('stats asymmetric bulge width_height class 1: ', width_height_values, width_height_counts)


# %%
def make_dir_if_not_exists(directory):
    if not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)


def save_train_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, large_asym_bulge_concepts,
                            dataframe, fold):
    for i in range(len(binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[binary_concepts_of_interest[i]] == True]

        if binary_concepts_of_interest[i] == 'presence_terminal_loop':
            target_data = target_data.loc[target_data['loop_length'] >= 21]
            target_data = target_data.loc[target_data['loop_width'] >= 12]

        print(len(target_data), binary_concepts_of_interest[i])
        calculate_summary_stats(target_data, 'class_label', binary_concepts_of_interest[i])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(
                f'./modhsa_original/concept_train_fold{fold}/{binary_concepts_of_interest[i]}/'
                f'{binary_concepts_of_interest[i]}')
            img.save(
                f'./modhsa_original/concept_train_fold{fold}/{binary_concepts_of_interest[i]}/'
                f'{binary_concepts_of_interest[i]}/{img_name}',
                'PNG')

    for i in range(len(non_binary_concepts_of_interest)):
        target_data = dataframe.loc[
            dataframe[non_binary_concepts_of_interest[i][0]] >= non_binary_concepts_of_interest[i][1]]

        print(len(target_data), non_binary_concepts_of_interest[i])
        calculate_summary_stats(target_data, 'class_label', non_binary_concepts_of_interest[i][0])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(
                f'./modhsa_original/concept_train_fold{fold}/{non_binary_concepts_of_interest[i][0]}/'
                f'{non_binary_concepts_of_interest[i][0]}')
            img.save(
                f'./modhsa_original/concept_train_fold{fold}/{non_binary_concepts_of_interest[i][0]}/'
                f'{non_binary_concepts_of_interest[i][0]}/{img_name}',
                'PNG')

    for i in range(len(large_asym_bulge_concepts)):
        target_data = dataframe.loc[
            dataframe[['large_asymmetric_bulge'][0]] >= large_asym_bulge_concepts[i][1]]

        target_data = target_data.loc[target_data['most_occurring_color_largest_asym_bulge'] ==
                                      large_asym_bulge_concepts[i][2]]

        print(len(target_data), large_asym_bulge_concepts[i][0])
        calculate_summary_stats(target_data, 'class_label', 'large_asymmetric_bulge')

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(
                f'./modhsa_original/concept_train_fold{fold}/{large_asym_bulge_concepts[i][0]}/'
                f'{large_asym_bulge_concepts[i][0]}')
            img.save(
                f'./modhsa_original/concept_train_fold{fold}/{large_asym_bulge_concepts[i][0]}/'
                f'{large_asym_bulge_concepts[i][0]}/{img_name}',
                'PNG')


def save_test_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, large_asym_bulge_concepts,
                           dataframe):
    for i in range(len(binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[binary_concepts_of_interest[i]] == True]

        if binary_concepts_of_interest[i] == 'presence_terminal_loop':
            target_data = target_data.loc[target_data['loop_length'] >= 21]
            target_data = target_data.loc[target_data['loop_width'] >= 12]

        print(len(target_data), binary_concepts_of_interest[i])
        calculate_summary_stats(target_data, 'class_label', binary_concepts_of_interest[i])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(f'./modhsa_original/original_dataset/concept_test/{binary_concepts_of_interest[i]}')
            img.save(f'./modhsa_original/original_dataset/concept_test/{binary_concepts_of_interest[i]}/{img_name}',
                     'PNG')

    for i in range(len(non_binary_concepts_of_interest)):
        target_data = dataframe.loc[
            dataframe[non_binary_concepts_of_interest[i][0]] >= non_binary_concepts_of_interest[i][1]]

        print(len(target_data), non_binary_concepts_of_interest[i])
        calculate_summary_stats(target_data, 'class_label', non_binary_concepts_of_interest[i][0])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(
                f'./modhsa_original/original_dataset/concept_test/{non_binary_concepts_of_interest[i][0]}')
            img.save(
                f'./modhsa_original/original_dataset/concept_test/{non_binary_concepts_of_interest[i][0]}/{img_name}',
                'PNG')

    for i in range(len(large_asym_bulge_concepts)):
        target_data = dataframe.loc[
            dataframe[['large_asymmetric_bulge'][0]] >= large_asym_bulge_concepts[i][1]]

        target_data = target_data.loc[target_data['most_occurring_color_largest_asym_bulge'] ==
                                      large_asym_bulge_concepts[i][2]]

        print(len(target_data), large_asym_bulge_concepts[i][0])
        calculate_summary_stats(target_data, 'class_label', 'large_asymmetric_bulge')

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(
                f'./modhsa_original/original_dataset/concept_test/{large_asym_bulge_concepts[i][0]}')
            img.save(
                f'./modhsa_original/original_dataset/concept_test/{large_asym_bulge_concepts[i][0]}/{img_name}',
                'PNG')


binary_concepts_of_interest = ['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA', 'asymmetric']
non_binary_concepts_of_interest = [('gap_start', 10), ('base_pairs_wobbles_in_stem', 0.9),
                                   ('large_asymmetric_bulge', 15)]
large_asym_bulge_concepts = [('large_asymmetric_bulge_yellow', 10, 'yellow'),
                             ('large_asymmetric_bulge_green', 10, 'green')]
# threshold base pairs used to be 0.75, width gap 6.5, large asym bulge 6 (now I am considering some extremer ones)
# (update 9-7) for the bulge: take a width closer to the max (the concepts can now still represent asymmetry for
# example)
# 18 is probably too strict (only around 30 images for each fold...)

# # split the training set in 5 folds for cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
train_data_tables = data_tables.loc[data_tables['set'] == 'train']
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data_tables, train_data_tables['class_label'])):
    training_set = train_data_tables.iloc[train_ids]
    val_set = train_data_tables.iloc[test_ids]

    print(len(training_set))
    print(len(val_set))

    for index, row in training_set.iterrows():
        image_path = row['path']
        img = Image.open(image_path)
        img_name = image_path[16::]
        label = row['class_label']
        img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
        make_dir_if_not_exists(
            f'./modhsa_original/train_fold{fold}/{label}')
        img.save(
            f'./modhsa_original/train_fold{fold}/{label}/{img_name}', 'PNG')

    save_train_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, large_asym_bulge_concepts,
                            training_set, fold)

    for index, row in val_set.iterrows():
        image_path = row['path']
        img = Image.open(image_path)
        img_name = image_path[16::]
        label = row['class_label']
        img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
        make_dir_if_not_exists(
            f'./modhsa_original/val_fold{fold}/{label}')
        img.save(
            f'./modhsa_original/val_fold{fold}/{label}/{img_name}', 'PNG')

# original test set
test_data_tables = data_tables.loc[data_tables['set'] == 'test']
save_test_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, large_asym_bulge_concepts,
                       test_data_tables)

# %%
binary_concepts_of_interest = ['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA']
non_binary_concepts_of_interest = [('gap_start', 10), ('base_pairs_wobbles_in_stem', 0.9),
                                   ('large_asymmetric_bulge', 15)]

# create a subset of the dataframe with the definitions of the specific concepts
target_data = data_tables[['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA',
                           'gap_start', 'base_pairs_wobbles_in_stem', 'large_asymmetric_bulge']]

# %%
# convert nans to Falses
target_data['UGU'] = target_data['UGU'].fillna(False)
target_data['AU_pair_begin_maturemiRNA'] = target_data['AU_pair_begin_maturemiRNA'].fillna(False)
target_data['presence_terminal_loop'] = target_data['presence_terminal_loop'].fillna(False)
target_data['gap_start'] = target_data['gap_start'].fillna(0)

# %%
concept_correlation = target_data.corr()
import seaborn as sns
import matplotlib.pyplot as plt

xticklabels = ['Terminal\nloop', 'U-G-U\nmotif', 'A-U\nmotif',
               'Asymmetric\nbulge\ninstead\nof\nterminal\n loop', 'Frequency\npairs\nand\nwobbles\nin\nstem',
               'High\nasymmetric\nbulge']
yticklabels = ['Terminal loop', 'U-G-U motif', 'A-U motif',
               'Asymmetric\nbulge instead\nof terminal loop', 'Frequency\nbase pairs and\nwobbles in stem',
               'High\nasymmetric\nbulge']
sns.set(font_scale=1)
sns.heatmap(concept_correlation, vmin=-1, vmax=1, xticklabels=xticklabels,
            yticklabels=yticklabels, annot=True, cmap='Oranges', annot_kws={"fontsize": 12})
plt.xticks(rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.title('Correlation between preliminary concepts', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('corr_matrix_preliminaryconcepts.svg', format='svg')
plt.show()

# %%
# create a subset of the dataframe with the definitions of the specific concepts
target_data_v2 = data_tables[['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA',
                              'gap_start', 'base_pairs_wobbles_in_stem', 'large_asymmetric_bulge',
                              'loop_length', 'loop_width']]
target_data_v2[["large_terminal_loop", "newlarge_gapcol2", "high_freq_base_pairs",
                'very_large_asym_bulge']] = None

target_data_v3 = pd.DataFrame(columns=['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA',
                              'gap_start', 'base_pairs_wobbles_in_stem', 'large_asymmetric_bulge',
                              'loop_length', 'loop_width', "large_terminal_loop", "large_gap",
                                       "high_freq_base_pairs", "very_large_asym_bulge"])
new_rows = []
for index, row in target_data_v2.iterrows():
    if row['loop_length'] >= 21 and row['loop_width'] >= 12:
        row['large_terminal_loop'] = True
    else:
        row['large_terminal_loop'] = False

    if row['gap_start'] >= 10:
        row['large_gap'] = True
    else:
        row['large_gap'] = False

    if row['base_pairs_wobbles_in_stem'] >= 0.9:
        row['high_freq_base_pairs'] = True
    else:
        row['high_freq_base_pairs'] = False

    if row['large_asymmetric_bulge'] >= 15:
        row['very_large_asym_bulge'] = True
    else:
        row['very_large_asym_bulge'] = False

    new_rows.append(row)
target_data_v3 = target_data_v3.append(pd.DataFrame(new_rows, columns=target_data_v3.columns)).reset_index()
#%%
target_data_v3 = target_data_v3[['large_terminal_loop','UGU', 'AU_pair_begin_maturemiRNA',
                              'large_gap', 'high_freq_base_pairs', 'very_large_asym_bulge']]
target_data_v3['UGU'] = target_data_v3['UGU'] .fillna(False)

#%%
# convert all Trues to 1 and all Falses to 0
target_data_v3 = target_data_v3.astype(int)

#%%
concept_correlation = round(target_data_v3.corr(), 2)
xticklabels = ['Large\nterminal\nloop', 'U-G-U\nmotif',
               'A-U\npairs\nmotif',
               'Large\nasymmetric\nbulge\ninstead\nof terminal\nloop',
               'At least\n90% base\npairs and\nwobbles in\nstem',
               'Large\nasymmetric\nbulge']
yticklabels = ['Large terminal loop', 'U-G-U motif', 'A-U pairs motif',
               'Large asymmetric\nbulge instead\nof terminal loop',
               'At least 90%\nbase pairs and\nwobbles in stem',
               'Large asymmetric bulge']
sns.set(font_scale=1)
sns.heatmap(concept_correlation, vmin=-1, vmax=1, xticklabels=xticklabels,
            yticklabels=yticklabels, annot=True, cmap='Oranges', annot_kws={"fontsize": 12})
plt.xticks(rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.title('Correlation between pre-miRNA concepts', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('corr_matrix_specificconcepts.svg', format='svg')
plt.show()
