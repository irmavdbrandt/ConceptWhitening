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
            symmetric_bulges, asymmetric_bulges, symmetric_bulge_info, asymmetric_bulge_info = bulges(data, base_pairs,
                                                                                                      loop_start_pixel,
                                                                                                      stem_begin)
            palindrome_score, palindrome_loop, largest_bulge = palindrome(data, loop_start_pixel)
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
            # # check the class label, if 0, start adding information on the bulges to the class0 list
            # if (dirname[39:40] == '0') or (dirname[38:39] == '0'):
            #     # in case the length of the asymmetric bulge list is >= 1, there is at least 1 asymmetric bulge present
            #     if len(asymmetric_bulge_info) >= 1:
            #         entry['asymmetric_bulge_presence'] = True
            #         entry['asymmetric_bulges_count'] = len(asymmetric_bulge_info)  # add the number of asymmetric bulges
            #         for i in range(len(asymmetric_bulge_info)):
            #             asymmetric_bulge_tables_class0.append([('index', filename),
            #                                                    ('width', asymmetric_bulge_info[i][3]),
            #                                                    ('height', asymmetric_bulge_info[i][4]),
            #                                                    ('width_height', str(asymmetric_bulge_info[i][3]) + '_'
            #                                                     + str(asymmetric_bulge_info[i][4]))])
            #             if (asymmetric_bulge_info[i][4] > loop_length) or (asymmetric_bulge_info[i][4] == loop_length):
            #                 if (asymmetric_bulge_info[i][3] > loop_width) or (
            #                         asymmetric_bulge_info[i][3] == loop_width):
            #                     entry['large_asymmetric_bulge'] = 'True'
            #                 else:
            #                     # only if the width is not like above, give the high value
            #                     entry['high_asymmetric_bulge'] = 'True'
            #
            #             if (asymmetric_bulge_info[i][3] > loop_width) or (asymmetric_bulge_info[i][3] == loop_width):
            #                 if (asymmetric_bulge_info[i][4] > loop_length) or \
            #                         (asymmetric_bulge_info[i][4] == loop_length):
            #                     entry['large_asymmetric_bulge'] = 'True'
            #                 else:
            #                     entry['wide_asymmetric_bulge'] = 'True'
            #     else:
            #         # in case the length of the asymmetric bulge list is 0, there is no asymmetric bulge present
            #         entry['asymmetric_bulge_presence'] = False
            #         entry['high_asymmetric_bulge'] = np.nan
            #         entry['wide_asymmetric_bulge'] = np.nan
            #         entry['large_asymmetric_bulge'] = np.nan
            #     # in case the length of the symmetric bulge list is >= 1, there is at least 1 symmetric bulge present
            #     if len(symmetric_bulge_info) >= 1:
            #         entry['symmetric_bulge_presence'] = True
            #         entry['symmetric_bulges_count'] = len(symmetric_bulge_info)
            #         for i in range(len(symmetric_bulge_info)):
            #             symmetric_bulge_tables_class0.append([('index', filename),
            #                                                   ('width', symmetric_bulge_info[i][3]),
            #                                                   ('height', symmetric_bulge_info[i][4]),
            #                                                   ('width_height', str(symmetric_bulge_info[i][3]) + '_'
            #                                                    + str(symmetric_bulge_info[i][4]))])
            #     else:
            #         # in case the length of the symmetric bulge list is 0, there is no symmetric bulge present
            #         entry['symmetric_bulge_presence'] = False
            # else:
            #     if len(asymmetric_bulge_info) >= 1:
            #         entry['asymmetric_bulge_presence'] = True
            #         entry['asymmetric_bulges_count'] = len(asymmetric_bulge_info)  # add the number of asymmetric bulges
            #         for i in range(len(asymmetric_bulge_info)):
            #             asymmetric_bulge_tables_class1.append([('index', filename),
            #                                                    ('width', asymmetric_bulge_info[i][3]),
            #                                                    ('height', asymmetric_bulge_info[i][4]),
            #                                                    ('width_height', str(asymmetric_bulge_info[i][3])
            #                                                     + '_' + str(asymmetric_bulge_info[i][4]))])
            #             if (asymmetric_bulge_info[i][4] > loop_length) or (asymmetric_bulge_info[i][4] == loop_length):
            #                 if (asymmetric_bulge_info[i][3] > loop_width) or (
            #                         asymmetric_bulge_info[i][3] == loop_width):
            #                     entry['large_asymmetric_bulge'] = 'True'
            #                 else:
            #                     # only if the width is not like above, give the high value
            #                     entry['high_asymmetric_bulge'] = 'True'
            #
            #             if (asymmetric_bulge_info[i][3] > loop_width) or (asymmetric_bulge_info[i][3] == loop_width):
            #                 if (asymmetric_bulge_info[i][4] > loop_length) or \
            #                         (asymmetric_bulge_info[i][4] == loop_length):
            #                     entry['large_asymmetric_bulge'] = 'True'
            #                 else:
            #                     entry['wide_asymmetric_bulge'] = 'True'
            #     # check the class label, if 1, start adding information on the bulges to the class1 list
            #     else:
            #         # in case the length of the asymmetric bulge list is 0, there is no asymmetric bulge present
            #         entry['asymmetric_bulge_presence'] = False
            #         entry['high_asymmetric_bulge'] = np.nan
            #         entry['wide_asymmetric_bulge'] = np.nan
            #         entry['large_asymmetric_bulge'] = np.nan
            #     # in case the length of the symmetric bulge list is >= 1, there is at least 1 symmetric bulge present
            #     if len(symmetric_bulge_info) >= 1:
            #         entry['symmetric_bulge_presence'] = True
            #         entry['symmetric_bulges_count'] = len(symmetric_bulge_info)
            #         for i in range(len(symmetric_bulge_info)):
            #             symmetric_bulge_tables_class1.append([('index', filename),
            #                                                   ('width', symmetric_bulge_info[i][3]),
            #                                                   ('height', symmetric_bulge_info[i][4]),
            #                                                   ('width_height', str(symmetric_bulge_info[i][3]) +
            #                                                    '_' + str(symmetric_bulge_info[i][4]))])
            #     else:
            #         # in case the length of the symmetric bulge list is 0, there is no symmetric bulge present
            #         entry['symmetric_bulge_presence'] = False
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

            # base pairing propensity: # base pairs / stem length
            entry['base_pairs_in_stem'] = (base_pairs.count(1) + base_pairs.count(2)) / (stem_begin - loop_start_pixel)
            entry['base_pairs_wobbles_in_stem'] = (base_pairs.count(1) + base_pairs.count(2) + base_pairs.count(3)) / \
                                                  (stem_begin - loop_start_pixel)

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

#%%
data_posclass_noloop = data_tables.loc[data_tables['class_label'] == '1']
data_posclass_noloop = data_posclass_noloop.loc[data_posclass_noloop['presence_terminal_loop'] == False]

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
# asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric_bulge_presence')
# asymmetric_bulge_count_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric_bulges_count')
# high_asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'high_asymmetric_bulge')
# wide_asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'wide_asymmetric_bulge')
# large_asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'large_asymmetric_bulge')
#
# widths = []
# heights = []
# width_height_combi = []
# for bulge in asymmetric_bulge_tables_class0:
#     widths.append(bulge[1][1])
#     heights.append(bulge[2][1])
#     width_height_combi.append(bulge[3][1])
# widths_values, width_counts = np.unique(widths, return_counts=True)
# heights_values, heights_counts = np.unique(heights, return_counts=True)
# width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
# print('stats asymmetric bulge widths class 0: ', widths_values, width_counts)
# print('stats asymmetric bulge heights class 0: ', heights_values, heights_counts)
# print('stats asymmetric bulge width_height class 0: ', width_height_values, width_height_counts)
#
# widths = []
# heights = []
# width_height_combi = []
# for bulge in asymmetric_bulge_tables_class1:
#     widths.append(bulge[1][1])
#     heights.append(bulge[2][1])
#     width_height_combi.append(bulge[3][1])
# widths_values, width_counts = np.unique(widths, return_counts=True)
# heights_values, heights_counts = np.unique(heights, return_counts=True)
# width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
# print('stats asymmetric bulge widths class 1: ', widths_values, width_counts)
# print('stats asymmetric bulge heights class 1: ', heights_values, heights_counts)
# print('stats asymmetric bulge width_height class 1: ', width_height_values, width_height_counts)


# %%
def make_dir_if_not_exists(directory):
    if not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)


def save_train_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, dataframe, fold):
    for i in range(len(binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[binary_concepts_of_interest[i]] == True]

        if binary_concepts_of_interest[i] == 'presence_terminal_loop':
            target_data = target_data.loc[target_data['loop_length'] >= 21]
            target_data = target_data.loc[target_data['loop_width'] >= 7]

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


def save_test_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, dataframe):
    for i in range(len(binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[binary_concepts_of_interest[i]] == True]

        if binary_concepts_of_interest[i] == 'presence_terminal_loop':
            target_data = target_data.loc[target_data['loop_length'] >= 21]
            target_data = target_data.loc[target_data['loop_width'] >= 7]

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


binary_concepts_of_interest = ['presence_terminal_loop', 'UGU', 'AU_pair_begin_maturemiRNA']
non_binary_concepts_of_interest = [('gap_start', 8), ('base_pairs_wobbles_in_stem', 0.9),
                                   ('large_asymmetric_bulge', 15)]
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

    save_train_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, training_set, fold)

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
save_test_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, test_data_tables)

# # %%
# # save concept images in the subdirectories
# def save_train_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, dataframe):
#     binary_target_df_names = [f"target_data_{binary_concepts_of_interest[i]}" for i in
#                               range(len(binary_concepts_of_interest))]
#     non_binary_target_df_names = [f"target_data_{non_binary_concepts_of_interest[i][0]}" for i in
#                                   range(len(non_binary_concepts_of_interest))]
#     target_df_names = binary_target_df_names + non_binary_target_df_names
#
#     targets = {}
#     for i in range(len(binary_concepts_of_interest)):
#         target_df = dataframe.loc[(dataframe[binary_concepts_of_interest[i]] == True) |
#                                   (dataframe[binary_concepts_of_interest[i]] == 'True')]
#         targets[target_df_names[i]] = target_df
#
#         print(len(target_df), binary_concepts_of_interest[i])
#
#     for i in range(len(non_binary_concepts_of_interest)):
#         target_df = dataframe.loc[dataframe[non_binary_concepts_of_interest[i][0]] >= non_binary_concepts_of_interest[i][1]]
#         targets[target_df_names[i + len(binary_concepts_of_interest)]] = target_df
#
#         print(len(target_df), non_binary_concepts_of_interest[i][0])
#
#     # order the range in targets and target_df_names based on the length of the target dfs (small to large)
#     targets = {k: v for k, v in sorted(targets.items(), key=lambda item: len(item[1]))}
#
#     # then take out 80 of the smallest target_df, save those 80 as concepts and their paths in a list
#     # these paths should then be removed from the other target_dfs
#
#
#     # subset the dataframes such that they are mutually exclusive
#     shared_img = []
#     new_targets = {}
#     for key, value in targets.items():
#         if len(shared_img) > 0:
#             value_noshared = value[~value['path'].isin(shared_img)]
#             new_target = value_noshared.sample(n=80, random_state=2)
#             shared_img.append(new_target['path'])
#             new_targets[key] = new_target
#         else:
#             new_target = value.sample(n=80, random_state=2)
#             shared_img.extend(new_target['path'])
#             new_targets[key] = new_target
#
#     for key, value in new_targets.items():
#         for index, row in value.iterrows():
#             image_path = row['path']
#             img = Image.open(image_path)
#             img_name = image_path[16::]
#             img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
#             name = key[12:]
#             make_dir_if_not_exists(
#                 f'./modhsa_original/concept_train/{name}/{name}')
#             img.save(
#                 f'./modhsa_original/concept_train/{name}/{name}/{img_name}', 'PNG')
#
#
# def save_test_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, dataframe):
#     binary_target_df_names = [f"target_data_{binary_concepts_of_interest[i]}" for i in
#                               range(len(binary_concepts_of_interest))]
#     non_binary_target_df_names = [f"target_data_{non_binary_concepts_of_interest[i][0]}" for i in
#                                   range(len(non_binary_concepts_of_interest))]
#     target_df_names = binary_target_df_names + non_binary_target_df_names
#
#     targets = {}
#     for i in range(len(binary_concepts_of_interest)):
#         target_df = dataframe.loc[(dataframe[binary_concepts_of_interest[i]] == True) |
#                                   (dataframe[binary_concepts_of_interest[i]] == 'True')]
#         targets[target_df_names[i]] = target_df
#
#         print(len(target_df), binary_concepts_of_interest[i])
#
#     for i in range(len(non_binary_concepts_of_interest)):
#         target_df = dataframe.loc[
#             dataframe[non_binary_concepts_of_interest[i][0]] >= non_binary_concepts_of_interest[i][1]]
#         targets[target_df_names[i + len(binary_concepts_of_interest)]] = target_df
#
#         print(len(target_df), non_binary_concepts_of_interest[i][0])
#
#     # order the range in targets and target_df_names based on the length of the target dfs (small to large)
#     targets = {k: v for k, v in sorted(targets.items(), key=lambda item: len(item[1]))}
#
#     # then take out 80 of the smallest target_df, save those 80 as concepts and their paths in a list
#     # these paths should then be removed from the other target_dfs
#
#     # subset the dataframes such that they are mutually exclusive
#     shared_img = []
#     new_targets = {}
#     for key, value in targets.items():
#         if len(shared_img) > 0:
#             value_noshared = value[~value['path'].isin(shared_img)]
#             new_target = value_noshared.sample(n=80, random_state=2)
#             shared_img.append(new_target['path'])
#             new_targets[key] = new_target
#         else:
#             new_target = value.sample(n=80, random_state=2)
#             shared_img.extend(new_target['path'])
#             new_targets[key] = new_target
#
#     for key, value in new_targets.items():
#         for index, row in value.iterrows():
#             image_path = row['path']
#             img = Image.open(image_path)
#             img_name = image_path[16::]
#             img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
#             name = key[12:]
#             make_dir_if_not_exists(
#                 f'./modhsa_original/concept_test/{name}')
#             img.save(
#                 f'./modhsa_original/concept_test/{name}/{img_name}', 'PNG')

# %%
print('train', np.mean([94.24, 91.46, 93.61, 94.29, 93.22]))
print('train', np.std([94.24, 91.46, 93.61, 94.29, 93.22]))
print('val', np.mean([97.86, 94.74, 97.27, 97.07, 95.9]))
print('val', np.std([97.86, 94.74, 97.27, 97.07, 95.9]))
print(np.mean([0.12, 0.12, 0.12, 0.13, 0.12]))
print(np.std([0.12, 0.12, 0.12, 0.13, 0.12]))
