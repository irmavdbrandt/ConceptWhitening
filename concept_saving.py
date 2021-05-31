from PIL import Image
import os
import numpy as np
import pandas as pd
from concept_detection import loop_concepts, ugu_motif, base_pairs_stem, bulges, palindrome, AU_pairs_begin_maturemiRNA
import tensorflow as tf

# STEP 1: collect the data and concept annotations

tables = []
symmetric_bulge_tables_class0 = []
symmetric_bulge_tables_class1 = []
asymmetric_bulge_tables_class0 = []
asymmetric_bulge_tables_class1 = []

for dirname, _, filenames in os.walk('modhsa_original/'):
    for filename in filenames:
        if filename == ".DS_Store":
            continue
        else:
            entry = pd.DataFrame([os.path.join(dirname, filename)], columns=['path'])
            if dirname[16:21] == "train":
                entry['class_label'] = dirname[22:23]
                entry['set'] = dirname[16:21]
            else:
                entry['class_label'] = dirname[21:22]
                entry['set'] = dirname[16:20]
            # add info on concepts
            image = Image.open(os.path.join(dirname, filename))
            # convert image to numpy array
            data = np.array(image)
            # generate all the concept information
            base_pairs, stem_begin = base_pairs_stem(data)
            terminal_loop, loop_start_row, loop_start_pixel, loop_highest_row, loop_highest_pixel, loop_length, \
            loop_width = loop_concepts(data, base_pairs)
            ugu_combined = ugu_motif(data, terminal_loop, loop_highest_pixel, loop_start_pixel)
            symmetric_bulges, asymmetric_bulges, symmetric_bulge_info, asymmetric_bulge_info = bulges(data, base_pairs,
                                                                                                      loop_start_pixel,
                                                                                                      stem_begin)
            palindrome_score, palindrome_loop = palindrome(data, loop_start_pixel)
            au_pair = AU_pairs_begin_maturemiRNA(base_pairs)
            # start adding the concept information to the entry of the dataframe
            entry['presence_terminal_loop'] = terminal_loop
            entry['start_loop_upperhalf_row'] = loop_start_row
            entry['start_loop_upperhalf_col'] = loop_start_pixel
            entry['highest_point_loop_upperhalf_row'] = loop_highest_row
            entry['highest_point_loop_upperhalf_col'] = loop_highest_pixel
            entry['loop_length'] = loop_length
            entry['loop_width'] = loop_width
            entry['stem_begin'] = stem_begin
            # check the class label, if 0, start adding information on the bulges to the class0 list
            if (dirname[22:23] == '0') or (dirname[21:22] == '0'):
                # in case the length of the asymmetric bulge list is >= 1, there is at least 1 asymmetric bulge present
                if len(asymmetric_bulge_info) >= 1:
                    entry['asymmetric_bulge_presence'] = True
                    entry['asymmetric_bulges_count'] = len(asymmetric_bulge_info)  # add the number of asymmetric bulges
                    for i in range(len(asymmetric_bulge_info)):
                        asymmetric_bulge_tables_class0.append([('index', filename),
                                                               ('width', asymmetric_bulge_info[i][3]),
                                                               ('height', asymmetric_bulge_info[i][4]),
                                                               ('width_height', str(asymmetric_bulge_info[i][3]) + '_'
                                                                + str(asymmetric_bulge_info[i][4]))])
                else:
                    # in case the length of the asymmetric bulge list is 0, there is no asymmetric bulge present
                    entry['asymmetric_bulge_presence'] = False
                # in case the length of the symmetric bulge list is >= 1, there is at least 1 symmetric bulge present
                if len(symmetric_bulge_info) >= 1:
                    entry['symmetric_bulge_presence'] = True
                    entry['symmetric_bulges_count'] = len(symmetric_bulge_info)
                    for i in range(len(symmetric_bulge_info)):
                        symmetric_bulge_tables_class0.append([('index', filename),
                                                                  ('width', symmetric_bulge_info[i][3]),
                                                                  ('height', symmetric_bulge_info[i][4]),
                                                                  ('width_height', str(symmetric_bulge_info[i][3]) + '_'
                                                                   + str(symmetric_bulge_info[i][4]))])
                else:
                    # in case the length of the symmetric bulge list is 0, there is no symmetric bulge present
                    entry['symmetric_bulge_presence'] = False
            else:
                if len(asymmetric_bulge_info) >= 1:
                    entry['asymmetric_bulge_presence'] = True
                    entry['asymmetric_bulges_count'] = len(asymmetric_bulge_info)  # add the number of asymmetric bulges
                    for i in range(len(asymmetric_bulge_info)):
                        asymmetric_bulge_tables_class1.append([('index', filename),
                                                               ('width', asymmetric_bulge_info[i][3]),
                                                               ('height', asymmetric_bulge_info[i][4]),
                                                               ('width_height', str(asymmetric_bulge_info[i][3])
                                                                + '_' + str(asymmetric_bulge_info[i][4]))])
                # check the class label, if 1, start adding information on the bulges to the class1 list
                else:
                    # in case the length of the asymmetric bulge list is 0, there is no asymmetric bulge present
                    entry['asymmetric_bulge_presence'] = False
                # in case the length of the symmetric bulge list is >= 1, there is at least 1 symmetric bulge present
                if len(symmetric_bulge_info) >= 1:
                    entry['symmetric_bulge_presence'] = True
                    entry['symmetric_bulges_count'] = len(symmetric_bulge_info)
                    for i in range(len(symmetric_bulge_info)):
                        symmetric_bulge_tables_class1.append([('index', filename),
                                                                  ('width', symmetric_bulge_info[i][3]),
                                                                  ('height', symmetric_bulge_info[i][4]),
                                                                  ('width_height', str(symmetric_bulge_info[i][3]) +
                                                                   '_' + str(symmetric_bulge_info[i][4]))])
                else:
                    # in case the length of the symmetric bulge list is 0, there is no symmetric bulge present
                    entry['symmetric_bulge_presence'] = False
            entry['palindrome_True'] = palindrome_score[1]
            entry['palindrome_loop'] = palindrome_loop
            entry['AU_pair_begin_maturemiRNA'] = au_pair
            if np.isnan(loop_start_pixel):
                entry['stem_end'] = 0
                entry['stem_length'] = stem_begin
                entry['total_width'] = stem_begin + 1
            else:
                entry['stem_end'] = loop_start_pixel + 1
                entry['stem_length'] = stem_begin - loop_width
                entry['total_width'] = stem_begin + 1
            if (base_pairs[0:4].count(1)) + (base_pairs[0:4].count(2)) == 4:
                entry['base_beginstem_4nt_clean'] = True
            else:
                entry['base_beginstem_4nt_clean'] = False
            if (base_pairs[0:4].count(1)) + (base_pairs[0:4].count(2)) == 3:
                entry['base_beginstem_4nt_1error'] = True
            else:
                entry['base_beginstem_4nt_1error'] = False
            if base_pairs[0:4].count(0) == 0:
                entry['base_wobble_beginstem_4nt_clean'] = True
            else:
                entry['base_wobble_beginstem_4nt_clean'] = False
            if (base_pairs[4:8].count(1)) + (base_pairs[4:8].count(2)) == 4:
                entry['base_beginstem_48nt_clean'] = True
            else:
                entry['base_beginstem_48nt_clean'] = False
            if (base_pairs[0:8].count(1)) + (base_pairs[0:8].count(2)) == 8:
                entry['base_beginstem_8nt_clean'] = True
            else:
                entry['base_beginstem_8nt_clean'] = False
            if (base_pairs[0:8].count(1)) + (base_pairs[0:8].count(2)) == 7:
                entry['base_beginstem_8nt_1error'] = True
            else:
                entry['base_beginstem_8nt_1error'] = False
            if base_pairs[0:8].count(0) == 0:
                entry['base_wobble_beginstem_8nt_clean'] = True
            else:
                entry['base_wobble_beginstem_8nt_clean'] = False
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
            entry['base_pair_successions'] = base_pair_successions

            # UGU-motif
            # we want the first occurrence of the combined nucleotides, so the motif with the smallest pixel as starting
            # point is assumed to be the motif of interest
            ugu_start = 100
            for i in range(len(ugu_combined)):
                if len(ugu_combined[i]) > 1:
                    if ugu_combined[i][1] < ugu_start:
                        entry['UGU'] = True
                        entry['UGU_start'] = ugu_combined[i][1]
                        entry['UGU_end'] = ugu_combined[i][2]
                        ugu_start = ugu_combined[i][1]
                    else:
                        print('motif but not first')
                else:
                    continue
            # if after the check the UGU-related cols have not been added, there is not motif and the cols
            # with default values should be added
            if len(entry.columns) < 12:
                entry['UGU'] = False
                entry['UGU_start'] = np.nan
                entry['UGU_end'] = np.nan
            tables.append(entry)

# collect the data and concept information
data_tables = pd.concat(tables, ignore_index=True)  # create dataframe from list of tables and reset index
print(data_tables['class_label'].value_counts())


# %%
def calculate_summary_stats(original_data, label_col, col_of_interest):
    data_stats = original_data[[label_col, col_of_interest]]
    data_stats = data_stats.groupby(label_col).describe().unstack(1).reset_index().pivot(index=label_col, values=0,
                                                                                         columns='level_1')
    print(f'Summary stats for {col_of_interest}: ', data_stats)
    return data_stats


print("General info")
width_stats = calculate_summary_stats(data_tables, 'class_label', 'total_width')
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

print("---------------")
print("---------------")
print("---------------")
print("Motif info")
UGU_stats = calculate_summary_stats(data_tables, 'class_label', 'UGU')
AU_pair_stats = calculate_summary_stats(data_tables, 'class_label', 'AU_pair_begin_maturemiRNA')
palindrome_stats = calculate_summary_stats(data_tables, 'class_label', 'palindrome_True')

print("---------------")
print("---------------")
print("---------------")
print("Base pairing info")
base_pair_succ_pres_stats = calculate_summary_stats(data_tables, 'class_label', 'base_pair_successions_presence')
base_pair_succ_data = calculate_summary_stats(data_tables, 'class_label', 'base_pair_successions')
clean_4nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'base_beginstem_4nt_clean')
error_4nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'base_beginstem_4nt_1error')
base_wobble_4nt_stats = calculate_summary_stats(data_tables, 'class_label', 'base_wobble_beginstem_4nt_clean')
base_48nt_stats = calculate_summary_stats(data_tables, 'class_label', 'base_beginstem_48nt_clean')
clean_8nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'base_beginstem_8nt_clean')
error_8nt_base_stats = calculate_summary_stats(data_tables, 'class_label', 'base_beginstem_8nt_1error')
base_wobble_8nt_stats = calculate_summary_stats(data_tables, 'class_label', 'base_wobble_beginstem_8nt_clean')

print("---------------")
print("---------------")
print("---------------")
print("Bulge info: symmetric")
symmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'symmetric_bulge_presence')
symmetric_bulge_count_stats = calculate_summary_stats(data_tables, 'class_label', 'symmetric_bulges_count')
widths = []
heights = []
width_height_combi = []
for bulge in symmetric_bulge_tables_class0:
    widths.append(bulge[1][1])
    heights.append(bulge[2][1])
    width_height_combi.append(bulge[3][1])
widths_values, width_counts = np.unique(widths, return_counts=True)
heights_values, heights_counts = np.unique(heights, return_counts=True)
width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
print('stats symmetric bulge widths class 0: ', widths_values, width_counts)
print('stats symmetric bulge heights class 0: ', heights_values, heights_counts)
print('stats symmetric bulge width_height class 0: ', width_height_values, width_height_counts)

widths = []
heights = []
width_height_combi = []
for bulge in symmetric_bulge_tables_class1:
    widths.append(bulge[1][1])
    heights.append(bulge[2][1])
    width_height_combi.append(bulge[3][1])
widths_values, width_counts = np.unique(widths, return_counts=True)
heights_values, heights_counts = np.unique(heights, return_counts=True)
width_height_values, width_height_counts = np.unique(width_height_combi, return_counts=True)
print('stats symmetric bulge widths class 1: ', widths_values, width_counts)
print('stats symmetric bulge heights class 1: ', heights_values, heights_counts)
print('stats symmetric bulge width_height class 1: ', width_height_values, width_height_counts)

print("---------------")
print("---------------")
print("---------------")
print("Bulge info: asymmetric")
asymmetric_bulge_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric_bulge_presence')
asymmetric_bulge_count_stats = calculate_summary_stats(data_tables, 'class_label', 'asymmetric_bulges_count')

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
train_data_tables = data_tables.loc[data_tables['set'] == 'train']
test_data_tables = data_tables.loc[data_tables['set'] == 'test']


def make_dir_if_not_exists(directory):
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)

 # %%
# save concept images in the subdirectories
# first concepts that are chosen: terminal loop presence, no mismatches/gaps in 1-4nt, sequence of base pairs,
# symmetric bulge presence, palindrome structure, AU pairs

def save_train_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, dataframe):
    for i in range(len(binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[binary_concepts_of_interest[i]] == True]

        print(len(target_data), binary_concepts_of_interest[i])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(f'./modhsa_original/concept_train/{binary_concepts_of_interest[i]}/{binary_concepts_of_interest[i]}')
            img.save(f'./modhsa_original/concept_train/{binary_concepts_of_interest[i]}/{binary_concepts_of_interest[i]}/{img_name}', 'PNG')

    for i in range(len(non_binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[non_binary_concepts_of_interest[i][0]] >= non_binary_concepts_of_interest[i][1]]

        print(len(target_data), non_binary_concepts_of_interest[i])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(f'./modhsa_original/concept_train/{binary_concepts_of_interest[i]}/{binary_concepts_of_interest[i]}')
            img.save(f'./modhsa_original/concept_train/{binary_concepts_of_interest[i]}/{binary_concepts_of_interest[i]}/{img_name}', 'PNG')


def save_test_concepts_img(binary_concepts_of_interest, non_binary_concepts_of_interest, dataframe):
    for i in range(len(binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[binary_concepts_of_interest[i]] == True]

        print(len(target_data), binary_concepts_of_interest[i])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(f'./modhsa_original/concept_test/{binary_concepts_of_interest[i]}')
            img.save(f'./modhsa_original/concept_test/{binary_concepts_of_interest[i]}/{img_name}', 'PNG')

    for i in range(len(non_binary_concepts_of_interest)):
        target_data = dataframe.loc[dataframe[non_binary_concepts_of_interest[i][0]] >= non_binary_concepts_of_interest[i][1]]

        print(len(target_data), non_binary_concepts_of_interest[i])

        for index, row in target_data.iterrows():
            image_path = row['path']
            img = Image.open(image_path)
            img_name = image_path[16::]
            img_name = img_name.replace("/", "__")  # replace / to prevent confusion when reading the path
            make_dir_if_not_exists(f'./modhsa_original/concept_test/{binary_concepts_of_interest[i]}')
            img.save(f'./modhsa_original/concept_test/{binary_concepts_of_interest[i]}/{img_name}', 'PNG')


binary_concepts_of_interest_train = ['presence_terminal_loop', 'AU_pair_begin_maturemiRNA',
                                     'base_pair_successions_presence', 'symmetric_bulge_presence',
                                     'base_beginstem_4nt_clean']
non_binary_concepts_of_interest_train = [('palindrome_True', 0.85)]  # for now, take the mean of the true class as
# threshold
save_train_concepts_img(binary_concepts_of_interest_train, non_binary_concepts_of_interest_train, train_data_tables)

binary_concepts_of_interest_test = ['presence_terminal_loop', 'AU_pair_begin_maturemiRNA',
                                     'base_pair_successions_presence', 'symmetric_bulge_presence',
                                     'base_beginstem_4nt_clean']
non_binary_concepts_of_interest_test = [('palindrome_True', 0.85)]  # for now, take the mean of the true class as
# threshold
save_test_concepts_img(binary_concepts_of_interest_test, non_binary_concepts_of_interest_test, test_data_tables)