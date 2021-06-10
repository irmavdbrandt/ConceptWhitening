#%%
import warnings
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')


def base_pairs_stem(data):
    """
    :param data: image data as array
    :return: pixel where stem begins (counting from right to left) and a list with integers [0, 3] and length equal to
    # the length of the pre-miRNA corresponding to the pairing in the stem. (0: gap or mismatch, 1: CG/GC, 2: AU/UA,
    3: GU/UG wobble)
    """
    # find the beginning pixel of the stem by looking for the first colored pixel in the row before the middle of the
    # image and going from right to left
    begin = 0
    row_before_middle = 12
    # iterate over the pixels from right to left (-1)
    for pixel_index in range(99, -1, -1):
        if (data[row_before_middle][pixel_index][0] == 255) and (
                data[row_before_middle][pixel_index][1] == 255) \
                and (data[row_before_middle][pixel_index][2] == 255):
            pixel_index += 1
        else:
            begin = pixel_index
            break

    # initialize an empty list where all the pairing integers will be stored
    pairs = []
    # iterate over the pixels from right to left starting for the beginning of the stem (-1)
    for pixel_index in range(begin, -1, -1):
        # C-G (blue-red with bar length 2 for both)
        if ((data[row_before_middle - 1][pixel_index][0] == 0) and
            (data[row_before_middle - 1][pixel_index][1] == 0)
            and (data[row_before_middle - 1][pixel_index][2] == 255)) and \
                ((data[row_before_middle - 2][pixel_index][0] == 255) and
                 (data[row_before_middle - 2][pixel_index][1] == 255)
                 and (data[row_before_middle - 2][pixel_index][2] == 255)) and \
                ((data[row_before_middle + 2][pixel_index][0] == 255) and (
                        data[row_before_middle + 2][pixel_index][1] == 0)
                 and (data[row_before_middle + 2][pixel_index][2] == 0)) and \
                ((data[row_before_middle + 3][pixel_index][0] == 255) and (
                        data[row_before_middle + 3][pixel_index][1] == 255)
                 and (data[row_before_middle + 3][pixel_index][2] == 255)):
            pairs.append(1)  # CG is referenced as 1
        else:
            # G-C (red-blue with bar length 2 for both)
            if ((data[row_before_middle - 1][pixel_index][0] == 255) and (
                    data[row_before_middle - 1][pixel_index][1] == 0)
                and (data[row_before_middle - 1][pixel_index][2] == 0)) and \
                    ((data[row_before_middle - 2][pixel_index][0] == 255) and (
                            data[row_before_middle - 2][pixel_index][1] == 255)
                     and (data[row_before_middle - 2][pixel_index][2] == 255)) and \
                    ((data[row_before_middle + 2][pixel_index][0] == 0) and (
                            data[row_before_middle + 2][pixel_index][1] == 0)
                     and (data[row_before_middle + 2][pixel_index][2] == 255)) and \
                    ((data[row_before_middle + 3][pixel_index][0] == 255) and (
                            data[row_before_middle + 3][pixel_index][1] == 255)
                     and (data[row_before_middle + 3][pixel_index][2] == 255)):
                pairs.append(1)  # CG is referenced as 1
            else:
                # U-A (green-yellow with bar length 3 for both)
                if ((data[row_before_middle - 2][pixel_index][0] == 0) and (
                        data[row_before_middle - 2][pixel_index][1] == 255)
                    and (data[row_before_middle - 3][pixel_index][2] == 255)) and \
                        ((data[row_before_middle - 3][pixel_index][0] == 255) and (
                                data[row_before_middle - 3][pixel_index][1] == 255)
                         and (data[row_before_middle - 2][pixel_index][2] == 0)) and \
                        ((data[row_before_middle + 3][pixel_index][0] == 255) and (
                                data[row_before_middle + 3][pixel_index][1] == 255)
                         and (data[row_before_middle + 3][pixel_index][2] == 0)) and \
                        ((data[row_before_middle + 4][pixel_index][0] == 255) and (
                                data[row_before_middle + 4][pixel_index][1] == 255)
                         and (data[row_before_middle + 4][pixel_index][2] == 255)):
                    pairs.append(2)  # UA is referenced as 2
                else:
                    # A-U (yellow-green with bars of length 3)
                    if ((data[row_before_middle - 2][pixel_index][0] == 255) and (
                            data[row_before_middle - 2][pixel_index][1] == 255)
                        and (data[row_before_middle - 2][pixel_index][2] == 0)) and \
                            ((data[row_before_middle - 3][pixel_index][0] == 255) and (
                                    data[row_before_middle - 3][pixel_index][1] == 255)
                             and (data[row_before_middle - 3][pixel_index][2] == 255)) and \
                            ((data[row_before_middle + 3][pixel_index][0] == 0) and (
                                    data[row_before_middle + 3][pixel_index][1] == 255)
                             and (data[row_before_middle + 3][pixel_index][2] == 0)) and \
                            ((data[row_before_middle + 4][pixel_index][0] == 255) and (
                                    data[row_before_middle + 4][pixel_index][1] == 255)
                             and (data[row_before_middle + 4][pixel_index][2] == 255)):
                        pairs.append(2)  # UA is referenced as 2
                    else:
                        # G-U wobble (red-green with length 4 for both bars)
                        if ((data[row_before_middle - 3][pixel_index][0] == 255) and (
                                data[row_before_middle - 3][pixel_index][1] == 0)
                            and (data[row_before_middle - 3][pixel_index][2] == 0)) and \
                                ((data[row_before_middle - 4][pixel_index][0] == 255) and (
                                        data[row_before_middle - 4][pixel_index][1] == 255)
                                 and (data[row_before_middle - 4][pixel_index][2] == 255)) and \
                                ((data[row_before_middle + 4][pixel_index][0] == 0) and (
                                        data[row_before_middle + 4][pixel_index][1] == 255)
                                 and (data[row_before_middle + 4][pixel_index][2] == 0)) and \
                                ((data[row_before_middle + 5][pixel_index][0] == 255) and (
                                        data[row_before_middle + 5][pixel_index][1] == 255)
                                 and (data[row_before_middle + 5][pixel_index][2] == 255)):
                            pairs.append(3)  # GU is referenced as 3
                        else:
                            # U-G wobble (green-red with length 4 for both bars)
                            if ((data[row_before_middle - 3][pixel_index][0] == 0) and (
                                    data[row_before_middle - 3][pixel_index][1] == 255)
                                and (data[row_before_middle - 3][pixel_index][2] == 0)) and \
                                    ((data[row_before_middle - 4][pixel_index][0] == 255) and (
                                            data[row_before_middle - 4][pixel_index][1] == 255)
                                     and (data[row_before_middle - 4][pixel_index][2] == 255)) and \
                                    ((data[row_before_middle + 4][pixel_index][0] == 255) and (
                                            data[row_before_middle + 4][pixel_index][1] == 0)
                                     and (data[row_before_middle + 4][pixel_index][2] == 0)) and \
                                    ((data[row_before_middle + 5][pixel_index][0] == 255) and (
                                            data[row_before_middle + 5][pixel_index][1] == 255)
                                     and (data[row_before_middle + 5][pixel_index][2] == 255)):
                                pairs.append(3)  # UG is referenced as 3
                            else:
                                pairs.append(0)  # all remaining pairs (gaps and mismatches) are referenced as 0

    return pairs, begin


def loop_concepts(data, pairs):
    """
    :param data: image data as array
    :param pairs: list of references to types of pairs in the stem
    :return: concepts related to the terminal loop: - presence of the loop
                                                    - starting pixel and row (counting from left to right, and in
                                                            the upper half of the image)
                                                    - pixel and row of highest point (from bottom to top) of the loop
                                                    - the length and width of the loop
    """

    # Terminal loop presence: is there a gap in the first pair of the pre-miRNA (from left to right) in either the upper
    # or lower half. This can be checked by going over the 12th and 13th row of the image and check for a black pixel
    # as first pixel in either of these rows.
    row_before_middle = 12

    terminal_loop_present = None
    for row_index in range(row_before_middle, row_before_middle + 2):
        if (data[row_index][0][0] == 0) and (data[row_index][0][1] == 0) \
                and (data[row_index][0][2] == 0):
            terminal_loop_present = False
            break
        else:
            terminal_loop_present = True

    # If present, estimate the terminal loop width and length by estimating the starting and end point (row, pixel)
    # do this by using the pair list: check for the first occurrence of a pair that is not a gap or mismatch
    # (i.e., 0)
    # the pairs list is ordered from right to left, so first reverse the order
    pairs_reversed = pairs[::-1]

    if terminal_loop_present:
        # step 1: find lowest point in loop (i.e., the starting pixel and row (going from right to left))
        start_pixel = 0
        start_row = 0
        for pair_index in range(len(pairs_reversed)):
            # if the pair is equal to 0, it is a gap or mismatch so it is still in the loop and we should continue
            if pairs_reversed[pair_index] == 0:
                continue
            else:
                # the first non-zero element is the start of the stem
                start_pixel = pair_index - 1  # loop ends at pixel that is found - 1
                break

        # go over all rows until the middle row with the starting pixel and check when the pixel color is changed from
        # white to colored, this is the row index where the loop starts
        for row_index in range(0, row_before_middle):
            if (data[row_index][start_pixel][0] == 255) and (data[row_index][start_pixel][1] == 255) and \
                    (data[row_index][start_pixel][2] == 255):
                continue
            else:
                start_row = row_index
                break

        # step 2: find highest point in loop
        highest_pixel = start_pixel  # initial value for the pixel: starting point of the loop
        highest_row = 12  # initial value for the row: the lowest row possible in the upper half
        for row_index in range(0, row_before_middle):
            # find first occurrence of colored pixels and store this index
            # for the pixels, we need to go until the first base pair has been found
            for pixel_index in range(0, start_pixel):
                if (data[row_index][pixel_index][0] == 255) and (
                        data[row_index][pixel_index][1] == 255) \
                        and (data[row_index][pixel_index][2] == 255):
                    continue
                else:
                    if ((pixel_index < highest_pixel) and (row_index < highest_row)) or \
                            ((pixel_index > highest_pixel) and (row_index == highest_row)):
                        highest_pixel = pixel_index
                        highest_row = row_index

        # width of the loop: from pixel 0 to start_pixel
        width = start_pixel + 1  # + 1 because we are dealing with an index from 0 to 99

        # get the lowest point of the loop by going over all rows in the bottom half of the image and the pixel index
        # of the highest point in the loop
        lowest_row = 0
        for row_index in range(row_before_middle + 1, 25):
            if ((data[row_index][highest_pixel][0] == 255) and
                    (data[row_index][highest_pixel][1] == 255)
                    and (data[row_index][highest_pixel][2] == 255)):
                if ((data[row_index - 1][highest_pixel][0] == 255) and
                        (data[row_index - 1][highest_pixel][1] == 255)
                        and (data[row_index - 1][highest_pixel][2] == 255)):
                    print('white pixel proceeded')
                else:
                    lowest_row = row_index

        # loop length: loop is symmetrical
        # Case 1: lowest point is the last row of the image and the highest point the first row of the image
        # --> the loop length is equal to the whole image length
        if (lowest_row == 0) and (highest_row == 0):
            length = 25
        else:
            # Case 2: the loop goes all the way down to the lowest row of the image but not all the way up to
            # the highest row --> the loop length is equal to 25 - highest row
            if (lowest_row == 0) and (highest_row != 0):
                length = 25 - highest_row
            else:
                # Case 3: loop does not go all the way down and up
                length = 25 - ((25 - lowest_row) + highest_row)

    else:
        start_row = np.nan
        start_pixel = np.nan
        highest_row = np.nan
        highest_pixel = np.nan
        length = np.nan
        width = np.nan

    return terminal_loop_present, start_row, start_pixel, highest_row, highest_pixel, length, width


def ugu_motif(data, terminal_loop_present, highest_pixel, start_pixel):
    """
    :param data: image data as array
    :param terminal_loop_present: boolean whether terminal loop is present or not
    :param highest_pixel: highest point of the terminal loop
    :param start_pixel: starting point of the terminal loop
    :return: list of references to whether the motif was present (1) or not (0), the starting point and ending point
    of the motif
    """
    # for the motif to be present, there needs to be a terminal loop
    if terminal_loop_present:
        # check whether the UGU-motif is either in the loop (from the highest point to the start) or just
        # before the loop (what is just before?, for now take a range of 5 pixels (nt)) in the upper half of the image
        # todo: check with "just before"
        gug_motifs = []
        ugu_motifs = []
        row_before_middle = 12
        for pixel in range(highest_pixel, start_pixel + 5):
            if ((data[row_before_middle][pixel][0] == 255) and (data[row_before_middle][pixel][1] == 0) and
                    (data[row_before_middle][pixel][2] == 0)):
                if ((data[row_before_middle][pixel + 1][0] == 0) and (data[row_before_middle][pixel + 1][1] == 255) and
                        (data[row_before_middle][pixel + 1][2] == 0)):
                    if ((data[row_before_middle][pixel + 2][0] == 255) and (
                            data[row_before_middle][pixel + 2][1] == 0) and (
                            data[row_before_middle][pixel + 2][2] == 0)):
                        gug_motifs.append([1, pixel, pixel + 2])
                    else:
                        gug_motifs.append([0])
                else:
                    gug_motifs.append([0])
            else:
                gug_motifs.append([0])
        for pixel in range(highest_pixel, start_pixel + 5):
            if ((data[row_before_middle][pixel][0] == 0) and (data[row_before_middle][pixel][1] == 255) and
                    (data[row_before_middle][pixel][2] == 0)):
                if ((data[row_before_middle][pixel + 1][0] == 255) and (data[row_before_middle][pixel + 1][1] == 0) and
                        (data[row_before_middle][pixel + 1][2] == 0)):
                    if ((data[row_before_middle][pixel + 2][0] == 0) and (
                            data[row_before_middle][pixel + 2][1] == 255) and (
                            data[row_before_middle][pixel + 2][2] == 0)):
                        ugu_motifs.append([1, pixel, pixel + 2])
                    else:
                        ugu_motifs.append([0])
                else:
                    ugu_motifs.append([0])
            else:
                ugu_motifs.append([0])

    else:
        ugu_motifs = [[0]]  # if there is no loop, there cannot be a ugu-motif
        gug_motifs = [[0]]  # if there is no loop, there cannot be a ugu-motif

    # combined ugu_motifs and gug_motifs
    combined_motifs = ugu_motifs + gug_motifs

    return combined_motifs


def bulge_info(bulges_list, data):
    """
    :param bulges_list: list containing indices of all bulges in the pre-miRNA
    :param data: image data as array
    :return: information on the bulges such as width, height, shape, start and end point, nucleotide types
    """
    row_before_middle = 12

    # go over the bulges in the list of bulges add generate the information
    bulge_info_list = []
    for bulge in bulges_list:
        width = len(bulge)  # the width of the bulge is equal to the total length of the bulge

        # find the highest point (row,pixel) of the bulge by going over the rows in the upper half of the image and the
        # pixels that are part of the bulge to find which pixel is located highest and has a white pixel above it (or
        # the pixel is located in the first row and there is nothing above it)
        bulge_highest_pixel = 100
        bulge_highest_row = row_before_middle
        for row_index in range(0, row_before_middle):
            # find first occurrence of colored pixels and store this index
            for pixel_index in range(bulge[0], bulge[-1] + 1):
                if (data[row_index][pixel_index][0] == 255) and (data[row_index][pixel_index][1] == 255) \
                        and (data[row_index][pixel_index][2] == 255):
                    continue
                else:
                    if ((pixel_index < bulge_highest_pixel) and (row_index < bulge_highest_row)) or \
                            ((pixel_index > bulge_highest_pixel) and (row_index == bulge_highest_row)):
                        bulge_highest_pixel = pixel_index
                        bulge_highest_row = row_index

        # similar for the lower half of the image to find the lowest point of the bulge
        bulge_highest_row_lowerhalf = 0
        bulge_highest_pixel_lowerhalf = None
        # loop over the rows from lowest to highest
        for row_index in range(24, row_before_middle, -1):
            for pixel_index in range(bulge[0], bulge[-1] + 1):
                if ((data[row_index][pixel_index][0] == 255) and
                        (data[row_index][pixel_index][1] == 255)
                        and (data[row_index][pixel_index][2] == 255)):
                    if ((data[row_index - 1][pixel_index][0] == 255) and
                            (data[row_index - 1][pixel_index][1] == 255)
                            and (data[row_index - 1][pixel_index][2] == 255)):
                        print('white pixel proceeded')
                    else:
                        bulge_highest_row_lowerhalf = row_index
                        bulge_highest_pixel_lowerhalf = pixel_index
                # if in the lowest row there is a colored pixel, we cannot use the before if statement
                # and then the lowest point of the bulge is the current row index
                else:
                    bulge_highest_row_lowerhalf = row_index + 1  # +1 bc we are dealing with index range starting at 0
                    bulge_highest_pixel_lowerhalf = pixel_index
            # stop the loop in case we have found a colored pixel
            if bulge_highest_row_lowerhalf != 0:
                break

        # in case the bulge's highest and lowest point are the first and last row, the length is equal to 25
        if (bulge_highest_row_lowerhalf == 25) and (bulge_highest_row == 0):
            length = 25
        else:
            # in case the lowest point is the lowest row, the length is equal to 25 - highest point
            if (bulge_highest_row_lowerhalf == 25) and (bulge_highest_row != 0):
                length = 25 - bulge_highest_row
            else:
                length = 25 - ((25 - bulge_highest_row_lowerhalf) + bulge_highest_row)

        # get color of pairs by checking the pixel RGB values
        nucleotide_pairs = []
        for pixel in range(bulge[0], bulge[-1] + 1):
            pair = []
            for row in range(row_before_middle, row_before_middle + 2):
                if ((data[row][pixel][0] == 0) and (data[row][pixel][1] == 0)
                        and (data[row][pixel][2] == 0)):
                    pair.append('gap')
                else:
                    if ((data[row][pixel][0] == 255) and (data[row][pixel][1] == 0)
                            and (data[row][pixel][2] == 0)):
                        pair.append('G')
                    else:
                        if ((data[row][pixel][0] == 0) and (data[row][pixel][1] == 255)
                                and (data[row][pixel][2] == 0)):
                            pair.append('U')
                        else:
                            if ((data[row][pixel][0] == 0) and (data[row][pixel][1] == 0)
                                    and (data[row][pixel][2] == 255)):
                                pair.append('C')
                            else:
                                if ((data[row][pixel][0] == 255) and (data[row][pixel][1] == 255)
                                        and (data[row][pixel][2] == 0)):
                                    pair.append('A')
            # combine the two nucleotides so that the pair is saved as ..-..
            nucleotide_pairs.append(pair[0] + "-" + pair[1])

        # add all the generated info into a list and add this list to an overview list
        bulge_info_list.append([bulge, (bulge_highest_row, bulge_highest_pixel), (bulge_highest_row_lowerhalf,
                                bulge_highest_pixel_lowerhalf), width, length, nucleotide_pairs])

    return bulge_info_list


#
#
# def shape_asymmetric_bulge(bulge_info_list, row_before_middle, image_array):
#     for bulge_info in bulge_info_list:
#         width = bulge_info[3]  # 3rd item from the info list is the width of the bulge
#         if width == 1:
#             shape = 'rectangle'
#             bulge_info.append(shape)
#         else:
#             bulge_indices_rows_upperhalf = []
#             for bulge_index in bulge_info[0]:
#                 for row_index in range(0, row_before_middle):
#                     if (image_array[row_index][bulge_index][0] == 255) and \
#                             (image_array[row_index][bulge_index][1] == 255) and \
#                             (image_array[row_index][bulge_index][2] == 255):
#                         continue
#                     else:
#                         if (image_array[row_index - 1][bulge_index][0] == 255) and \
#                                 (image_array[row_index - 1][bulge_index][1] == 255) and \
#                                 (image_array[row_index - 1][bulge_index][2] == 255):
#                             bulge_indices_rows_upperhalf.append((row_index, bulge_index))
#             bulge_indices_rows_lowerhalf = []
#             for bulge_index in bulge_info[0]:
#                 for row_index in range(row_before_middle + 1, 25):
#                     if (image_array[row_index][bulge_index][0] == 255) and \
#                             (image_array[row_index][bulge_index][1] == 255) and \
#                             (image_array[row_index][bulge_index][2] == 255):
#                         continue
#                     else:
#                         if (image_array[row_index + 1][bulge_index][0] == 255) and \
#                                 (image_array[row_index + 1][bulge_index][1] == 255) and \
#                                 (image_array[row_index + 1][bulge_index][2] == 255):
#                             bulge_indices_rows_lowerhalf.append((row_index, bulge_index))
#
#             print(bulge_indices_rows_upperhalf)
#             # get the shape of the upper half of the bulge
#             left_triangle_upper = []
#             right_triangle_upper = []
#             for i in range(len(bulge_indices_rows_upperhalf) - 1):
#                 if (bulge_indices_rows_upperhalf[i][0] > bulge_indices_rows_upperhalf[i + 1][0]) and \
#                         (bulge_indices_rows_upperhalf[i][1] < bulge_indices_rows_upperhalf[i + 1][1]):
#                     left_triangle_upper.append(True)
#                     right_triangle_upper.append(False)
#
#                 else:
#                     if bulge_indices_rows_upperhalf[i][0] == 0:
#                         break
#                     else:
#                         left_triangle_upper.append(False)
#                         if (bulge_indices_rows_upperhalf[i][0] < bulge_indices_rows_upperhalf[i + 1][0]) and \
#                                 (bulge_indices_rows_upperhalf[i][1] < bulge_indices_rows_upperhalf[i + 1][1]):
#                             right_triangle_upper.append(True)
#                             if bulge_indices_rows_upperhalf[i][0] == 0:
#                                 break
#                         else:
#                             left_triangle_upper.append(False)
#                             right_triangle_upper.append(False)
#                             print('no clean triangle, next steps...')
#
#             print(bulge_indices_rows_lowerhalf)
#             # get the shape of the upper half of the bulge
#             left_triangle_lower = []
#             right_triangle_lower = []
#             for i in range(len(bulge_indices_rows_lowerhalf) - 1):
#                 print(bulge_indices_rows_lowerhalf[i][0])
#                 if (bulge_indices_rows_lowerhalf[i][0] > bulge_indices_rows_lowerhalf[i + 1][0]) and \
#                         (bulge_indices_rows_lowerhalf[i][1] < bulge_indices_rows_lowerhalf[i + 1][1]):
#                     left_triangle_lower.append(True)
#                     right_triangle_lower.append(False)
#
#                 else:
#                     if bulge_indices_rows_lowerhalf[i][0] == 24:
#                         break
#                     else:
#                         left_triangle_lower.append(False)
#                         if (bulge_indices_rows_lowerhalf[i][0] < bulge_indices_rows_lowerhalf[i + 1][0]) and \
#                                 (bulge_indices_rows_lowerhalf[i][1] < bulge_indices_rows_lowerhalf[i + 1][1]):
#                             right_triangle_lower.append(True)
#                             if bulge_indices_rows_lowerhalf[i][0] == 24:
#                                 break
#                         else:
#                             left_triangle_lower.append(False)
#                             right_triangle_lower.append(False)
#                             print('no clean triangle, next steps...')
#
#             print(left_triangle_upper)
#             print(right_triangle_upper)
#             print(left_triangle_lower)
#             print(right_triangle_lower)
#
#             if all(left_triangle_upper):
#                 shape_upper = 'left-triangle-clean-upper'
#                 if all(left_triangle_lower):
#                     shape_lower = 'left-triangle-clean-lower'
#                     shape = shape_upper + "__" + shape_lower
#                     print(shape)
#                     bulge_info.append(shape)
#                 else:
#                     if all(right_triangle_lower):
#                         shape_lower = 'right-triangle-clean-lower'
#                         shape = shape_upper + "__" + shape_lower
#                         print(shape)
#                         bulge_info.append(shape)
#                     else:
#                         shape_lower = 'None'
#                         shape = shape_upper + "__" + shape_lower
#                         print(shape)
#                         bulge_info.append(shape)
#             else:
#                 print('going to this else')
#                 if all(right_triangle_upper):
#                     shape_upper = 'right-triangle-clean-upper'
#                     if all(left_triangle_lower):
#                         shape_lower = 'left-triangle-clean-lower'
#                         shape = shape_upper + "__" + shape_lower
#                         print(shape)
#                         bulge_info.append(shape)
#                     else:
#                         if all(right_triangle_lower):
#                             shape_lower = 'right-triangle-clean-lower'
#                             shape = shape_upper + "__" + shape_lower
#                             print(shape)
#                             bulge_info.append(shape)
#                         else:
#                             shape_lower = 'None'
#                             shape = shape_upper + "__" + shape_lower
#                             print(shape)
#                             bulge_info.append(shape)
#                 else:
#                     shape_upper = 'None'
#                     if all(left_triangle_lower):
#                         shape_lower = 'left-triangle-clean-lower'
#                         shape = shape_upper + "__" + shape_lower
#                         print(shape)
#                         bulge_info.append(shape)
#                     else:
#                         if all(right_triangle_lower):
#                             shape_lower = 'right-triangle-clean-lower'
#                             shape = shape_upper + "__" + shape_lower
#                             print(shape)
#                             bulge_info.append(shape)
#                         else:
#                             shape_lower = 'None'
#                             shape = shape_upper + "__" + shape_lower
#                             print(shape)
#                             bulge_info.append(shape)
#
#     return bulge_info_list
#
#
# def shape_symmetric_bulge(bulge_info_list, row_before_middle, image_array):
#     for bulge_info in bulge_info_list:
#         width = bulge_info[3]  # 3rd item from the info list is the width of the bulge
#         if width == 1:
#             shape = 'rectangle'
#             bulge_info.append(shape)
#         else:
#             # if the width is more than 1, it can be a rectangle or triangle (left/right) or loop
#             # check where the lowest point of the bulge is to define a rectangle
#             # NOTE: it can also be that it is a loop --> two low points
#             # best thing would be to safe all the rows and indices of the loop
#             bulge_indices_rows_upperhalf = []
#             for bulge_index in bulge_info[0]:
#                 for row_index in range(0, row_before_middle):
#                     if (image_array[row_index][bulge_index][0] == 255) and \
#                             (image_array[row_index][bulge_index][1] == 255) and \
#                             (image_array[row_index][bulge_index][2] == 255):
#                         continue
#                     else:
#                         if (image_array[row_index - 1][bulge_index][0] == 255) and \
#                                 (image_array[row_index - 1][bulge_index][1] == 255) and \
#                                 (image_array[row_index - 1][bulge_index][2] == 255):
#                             bulge_indices_rows_upperhalf.append((row_index, bulge_index))
#             bulge_indices_rows_lowerhalf = []
#             for bulge_index in bulge_info[0]:
#                 for row_index in range(row_before_middle + 1, 25):
#                     if (image_array[row_index][bulge_index][0] == 255) and \
#                             (image_array[row_index][bulge_index][1] == 255) and \
#                             (image_array[row_index][bulge_index][2] == 255):
#                         continue
#                     else:
#                         if (image_array[row_index + 1][bulge_index][0] == 255) and \
#                                 (image_array[row_index + 1][bulge_index][1] == 255) and \
#                                 (image_array[row_index + 1][bulge_index][2] == 255):
#                             bulge_indices_rows_lowerhalf.append((row_index, bulge_index))
#
#             left_triangle = []
#             right_triangle = []
#             for i in range(len(bulge_indices_rows_upperhalf) - 1):
#                 if (bulge_indices_rows_upperhalf[i][0] > bulge_indices_rows_upperhalf[i + 1][0]) and \
#                         (bulge_indices_rows_upperhalf[i][1] < bulge_indices_rows_upperhalf[i + 1][1]):
#                     left_triangle.append(True)
#                     right_triangle.append(False)
#
#                 else:
#                     if bulge_indices_rows_upperhalf[i][0] == 0:
#                         break
#                     else:
#                         left_triangle.append(False)
#                         if (bulge_indices_rows_upperhalf[i][0] < bulge_indices_rows_upperhalf[i + 1][0]) and \
#                                 (bulge_indices_rows_upperhalf[i][1] < bulge_indices_rows_upperhalf[i + 1][1]):
#                             right_triangle.append(True)
#                             if bulge_indices_rows_upperhalf[i][0] == 0:
#                                 break
#                         else:
#                             left_triangle.append(False)
#                             right_triangle.append(False)
#                             shape = "Loop or rectangle"
#                             bulge_info.append(shape)
#
#             if False in left_triangle:
#                 continue
#             else:
#                 shape = 'left-triangle-clean'
#                 bulge_info.append(shape)
#
#             if False in right_triangle:
#                 continue
#             else:
#                 shape = 'right-triangle-clean'
#                 bulge_info.append(shape)
#
#     return bulge_info_list
#

def bulges(data, pairs, start_pixel, begin):
    """
    :param data: image data as array
    :param pairs: list of references to pairs inside pre-miRNA
    :param start_pixel: pixel index of starting point of loop (from right to left)
    :param begin: beginning index of stem (from right to left)
    :return: list of symmetric and asymmetric bulges, as well as more detailed list containing information on the size,
    height, nucleotide pairs, shape, etc. inside the bulge
    """
    # go over pairs list and for every part between base pairs (1/2) and/or wobbles (3), check if it is a symmetric or
    # asymmetric bulge. The assumption is that there cannot be a succession of symmetric and asymmetric
    # bulges (there should always be base pairs in between)
    row_before_middle = 12

    # we need to look for bulges in the area after the loop, hence we need to slice the pairs list from the stem begin
    # until the start point of the loop. Recall that the pairs list is ordered from right to left
    # in case there is no loop, we do not have a starting point for the loop so we assign the smallest index possible
    # -1 as the starting pixel (to have a correct range that does not stop too early)
    if np.isnan(start_pixel):
        start_pixel = -1
    pairs_until_loop = pairs[0:len(pairs) - int(start_pixel) - 1]  # -1 bc we should stop collecting
    # pairs one before the loop starts, otherwise pairs from the loop are also added

    # get all the indices where the pair is part of a bulge (value 0). Since the base_pairs list is in reverse order
    # compared to the original image pixel order, we do stem begin - index of the list item
    bulge_indices = [begin - i for i, e in enumerate(pairs_until_loop) if e == 0]

    # create lists of lists from items bulge_indices so that successive indices are in one list
    bulge_indices_grouped = []
    bulge_list = []  # initialize an empty list where the indices of one bulge will be stored
    for i in range(len(bulge_indices) - 1):
        # check whether the bulge consists of only 1 element by checking whether the bulge indices are more than 1 away
        # from each other
        if (bulge_indices[i] + 1 != bulge_indices[i - 1]) and (bulge_indices[i] - 1 != bulge_indices[i + 1]):
            single_bulge = [bulge_indices[i]]
            bulge_indices_grouped.append(single_bulge)
        else:
            # in case the bulge consists of multiple pairs, check whether the indices succeed each other
            if bulge_indices[i] - 1 == bulge_indices[i + 1]:
                bulge_list.append(bulge_indices[i])
                # if the index is equal to the final index in the range of the for loop (len(bulge_indices) - 2), add
                # the next index to the bulge list (as this is the final index which is still part of the bulge but will
                # not be included in the for loop
                if i == (len(bulge_indices) - 2):
                    bulge_list.append(bulge_indices[i + 1])

                    bulge_indices_grouped.append(bulge_list)
            else:
                # in case the next item in the bulge list is no longer a successive element of the current index, we
                # have reached the final index of a bulge. This index should be added to the bulge list
                bulge_list.append(bulge_indices[i])
                # now the list containing all indices of one bulge is complete, so it can be added the overall bulge
                # list
                bulge_indices_grouped.append(bulge_list)
                bulge_list = []  # re-initialize the empty list for bulge storage

    symmetric_bulges_list = []
    asymmetric_bulges_list = []
    # go over all bulge lists in the bulge_indices_grouped and check whether one of the pairs in the list contains
    # a gap --> if so, the bulge is asymmetric, if not, it is symmetric
    for bulge_list in bulge_indices_grouped:
        gap_test = []
        for pixel_index in bulge_list:
            if ((data[row_before_middle - 1][pixel_index][0] == 0) and (
                    data[row_before_middle - 1][pixel_index][1] == 0)
                    and (data[row_before_middle - 1][pixel_index][2] == 0)):
                gap_test.append(True)
            else:
                if ((data[row_before_middle + 2][pixel_index][0] == 0) and (
                        data[row_before_middle + 2][pixel_index][1] == 0)
                        and (data[row_before_middle + 2][pixel_index][2] == 0)):
                    gap_test.append(True)
                else:
                    gap_test.append(False)
        # if there is a True in gap_test, there is a gap in the bulge and we are dealing with an asymmetric bulge
        # also, reverse the order in the list so that the bulge is ordered from left to right
        if True in gap_test:
            asymmetric_bulges_list.append(bulge_list[::-1])
        else:
            symmetric_bulges_list.append(bulge_list[::-1])

    # reverse the order of the list of bulges to match with the order in the image (from left to right)
    symmetric_bulges_list = symmetric_bulges_list[::-1]
    asymmetric_bulges_list = asymmetric_bulges_list[::-1]

    # get information on the bulges such as width, height, start and end point, nucleotide types inside bulge
    symmetric_bulge_info_list = bulge_info(symmetric_bulges_list, data)
    asymmetric_bulge_info_list = bulge_info(asymmetric_bulges_list, data)

    # asymmetric_bulge_info = shape_asymmetric_bulge(asymmetric_bulge_info, row_before_middle, image_array)
    # symmetric_bulge_info = shape_symmetric_bulge(symmetric_bulge_info, row_before_middle, image_array)

    return symmetric_bulges_list, asymmetric_bulges_list, symmetric_bulge_info_list, asymmetric_bulge_info_list


def palindrome(data, start_pixel):
    """
    :param data: image data as array
    :param start_pixel: pixel that denotes the start of the loop (from right to left)
    :return: score referring to what extent the structure of the two halves of the image is symmetrical
    """
    # for both halves of the figure, count the number of colored pixels in a bar until a white pixel occurs
    # save the counts in lists
    # loop over both lists simultaneously and if the counts are equal, add True, else, add False
    # lastly, divide the amount of Trues by the length of the pre-miRNA to get the symmetrical structure score

    row_before_middle = 12

    upper_half_counts = []
    lower_half_counts = []
    for pixel_index in range(0, 100):
        upper_count = 0
        for row_index in range(row_before_middle, -1, -1):
            if (data[row_index][pixel_index][0] == 255) and (data[row_index][pixel_index][1] == 255) and \
                    (data[row_index][pixel_index][2] == 255):
                upper_half_counts.append(upper_count)
                break
            else:
                # if the colored pixels go up to the first row, there is no white pixel above that any more so we need
                # to add the current count to the list
                if row_index == 0:
                    upper_half_counts.append(upper_count)
                    break
                else:
                    upper_count += 1

        lower_count = 0
        for row_index in range(row_before_middle + 1, 25):
            if (data[row_index][pixel_index][0] == 255) and \
                    (data[row_index][pixel_index][1] == 255) and \
                    (data[row_index][pixel_index][2] == 255):
                lower_half_counts.append(lower_count)
                break
            else:
                # if the colored pixels go up to the last row, there is no white pixel below that any more so we need
                # to add the current count to the list
                if row_index == 24:
                    lower_half_counts.append(lower_count + 1)
                    break
                else:
                    lower_count += 1

    # zip the count lists and go over them to check whether the counts are equal (= symmetric structure) or not.
    # if so, add True, else, add False
    palindrome_array = []
    for pixel_upper, pixel_lower in zip(upper_half_counts, lower_half_counts):
        if pixel_upper == pixel_lower:
            palindrome_array.append(True)
        else:
            palindrome_array.append(False)

    # divide the number of Trues/Falses by the length of the pre-miRNA
    # get the index of the first occurrence of a 0-count, this is where there are only white pixels
    # this also defines the total length of the pre-miRNA
    if 0 in upper_half_counts:
        len_premiRNA = upper_half_counts.index(0)
    else:
        len_premiRNA = len(upper_half_counts)
    # use the index for the length of the pre-miRNA to slice the palindrome array so that there are True/False values
    # that reference to the pre-miRNA and not to the white pixels
    subset_palindrome_array = palindrome_array[0:len_premiRNA]

    # get the values and frequencies for the Trues and Falses in the array
    subset_values, subset_counts = np.unique(subset_palindrome_array, return_counts=True)
    # if only Trues or only Falses, then the counts will have length one so we need to account for that for
    # the next lines of code
    # if only Trues then change counts from [1.] into [0., 1.]
    if all(subset_palindrome_array):
        subset_counts = np.array([0, 1 * len_premiRNA])
    else:
        # if only Falses then change counts from [1.] into [1., 0.]
        if not any(subset_palindrome_array):
            subset_counts = np.array([1 * len_premiRNA, 0])
    # divide the counts of the Trues and Falses by the length of the pre-miRNA to get the palindrome structure score
    # the result is an array with [(False counts / len_premiRNA), (True counts / len_premiRNA)]
    palindrome_score = subset_counts / len_premiRNA

    palindrome_score_loop = None
    if np.isnan(start_pixel):
        palindrome_score_loop = False
    else:
        loop_part = subset_palindrome_array[0:int(start_pixel)]
        if all(loop_part):
            palindrome_score_loop = True
        else:
            palindrome_score_loop = False

    return palindrome_score, palindrome_score_loop


def AU_pairs_begin_maturemiRNA(pairs):
    """
    :param pairs: list of references to types of pairs in stem of pre-miRNA
    :return: boolean whether a successions of at least 2 AU/UA pairs was found in the area around 18-25 nt from the
    stem begin
    """
    # look for a succession/presence of 2+ AU/UA pairs in the area around 18-25 nt from the stem begin
    area_of_interest = pairs[18:26]

    AU_motif = None
    for i in range(len(area_of_interest) - 1):
        if (area_of_interest[i] == 2) and (area_of_interest[i + 1] == 2):
            AU_motif = True
            break
        else:
            AU_motif = False

    # if the length of the pre-miRNA < 19, it cannot be a pre-miRNA so there also cannot be a AU motif
    # todo: check this with Jens
    if len(area_of_interest) < 2:
        AU_motif = False

    return AU_motif
