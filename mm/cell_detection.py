# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
import itertools

from ..generic.otsu import threshold_otsu
from ..generic.signal import hamming_smooth, simple_baseline_correction, find_extrema_and_prominence, \
    vertical_mean, threshold_outliers, savitzky_golay_filter, horizontal_mean, normalize, normalize2, \
    add_element_to_list, multiply_list_by_element, subtraction, white_top_hat, min_and_max, normalize_specific_range

from ..debugging import DebugPlot
from ..generic.tunable import tunable


class Cell(object):
    """
    A Cell.

    :param top: coordinate of the 'top' of the cell, in channel coordinates
    :param bottom: coordinate of the 'bottom' of the cell, in channel coordinates
    :param channel: Channel object the cell belongs to
    """
    __slots__ = ['local_top', 'local_bottom', 'channel']

    def __init__(self, top, bottom, channel):
        self.local_top = float(top)
        self.local_bottom = float(bottom)

        self.channel = channel

    @property
    def top(self):
        """
        Returns the absolute (on rotated image) coordinate of the cell top.

        :return: top
        """
        return self.channel.top + self.local_top

    @property
    def bottom(self):
        """
        Returns the absolute (on rotated image) coordinate of the cell bottom.

        :return:
        """
        return self.channel.top + self.local_bottom

    @property
    def length(self):
        """
        Returns the cell length.

        :return: length
        """
        return abs(self.top - self.bottom)

    @property
    def centroid_1d(self):
        """
        Returns the (one dimensional) (absolute coordinate on rotated image) centroid.
        :return: centroid
        :rtype: float
        """
        return (self.top + self.bottom) / 2.0

    @property
    def centroid(self):
        """
        Returns the (absolute coordinate on rotated image) centroid (2D).
        :return:
        :rtype: list
        """
        return [self.channel.centroid[0], self.centroid_1d]

    @property
    def cell_image(self):
        """
        The cell image, cropped out of the channel image.

        :return: image
        :rtype: numpy.ndarray
        """
        return self.crop_out_of_channel_image(self.channel.channel_image)

    def crop_out_of_channel_image(self, channel_image):
        """
        Crops the clel out of a provided image.
        Used internally for :py:meth:`Cell.cell_image`, and to crop cells out of fluorescence channel images.

        :param channel_image:
        :type channel_image: numpy.ndarray
        :return: image
        :rtype: numpy.ndarray
        """
        return channel_image[int(self.local_top):int(self.local_bottom), :]

    def __lt__(self, other_cell):
        return self.local_top < other_cell.local_top


class Cells(object):
    """
        A Cells object, a collection of Cell objects.
    """

    __slots__ = ['cells_list', 'channel', 'nearest_tree', 'min_max_from_profile']

    cell_type = Cell

    def __init__(self, channel, bootstrap=True):

        self.cells_list = []

        self.channel = channel

        self.nearest_tree = None

        if not bootstrap:
            return

        find_cells_in_channel_ = find_cells_in_channel(self.channel.channel_image)

        cells_list = find_cells_in_channel_[0]
        self.min_max_from_profile = find_cells_in_channel_[1]

        # only keep the top cell
        if len(cells_list) > 0:
            del cells_list[1:];  # bottom
            # del cells_list[0:-1]; # top

        for b, e in cells_list:
            # ... this is the actual minimal size filtering
            if self.channel.image.mu_to_pixel(
                    tunable('cells.minimal_length.in_mu', 1.0,
                            description="The minimal allowed cell size (Smaller cells will be filtered out).")
            ) < e - b:
                self.cells_list.append(self.__class__.cell_type(b, e, self.channel))

    def __len__(self):
        return len(self.cells_list)

    def __iter__(self):
        return iter(self.cells_list)

    def clean(self):
        """
        Performs clean-up.

        """
        pass

    @property
    def centroids(self):
        """
        Returns the centroids of the cells.

        :return: centroids
        :rtype: list
        """
        return [cell.centroid for cell in self.cells_list]


def find_cells_in_channel(image):
    method = tunable('cells.detectionmethod', 'classic', description="Cell detection method to use.")
    if method == 'classic':
        return find_cells_in_channel_classic_with_top_hat(image)
    else:
        raise RuntimeError('Unsupported cell detection method passed.')


def find_cells_in_channel_classic_with_top_hat(original_image):
    """

    :param original_image:
    :return:
    """

    #################### Parameters ######################
    otsu_bias = 0.8
    smooth = 7
    cells_extrema_order = 12
    # otsu_threshold_gap = 0.6
    otsu_bright = 0.5
    threshold_prominence_for_poles = 40

    # is a cell parameters
    min_size = 9
    max_bright = 0.7
    min_prominence = 10

    smoth_binary_profile = smooth

    cut_edges_from_binary_image = True

    top_hat_transformation = True
    top_hat_tuple = [6, 5]  # this is the transpose of the tuple used in the notebook for debugging
    cut_image = False
    pixel_start, number_pixels_to_keep = 1, 6

    ############### Print Debug Parameters ##############
    debug_print_possible_poles = False
    debug_print_cell_evaluation = False

    ##################### Transform #####################

    if top_hat_transformation:
        image = white_top_hat(original_image, top_hat_tuple)
    else:
        image = original_image
    if cut_image:
        def cut_rows_from_image_array(image_, pixels_from_left, number_pixels_to_keep):
            h = pixels_from_left + number_pixels_to_keep
            cutted_image = [a[pixels_from_left:h:] for a in image_.astype(float)]
            return np.array(cutted_image)

        cutted_image = cut_rows_from_image_array(image, pixel_start, number_pixels_to_keep)
        image = cutted_image

    ####################### Binary image & cut edges of it

    threshold_image = threshold_otsu(image)

    # import numpy as np
    # print(np.min(image) ,"\t", np.max(image),"\t",  np.round(np.mean(image)), "\t", threshold_image, "\t", np.min(original_image), "\t", np.max(original_image), "\t", np.round(np.mean(original_image)),"\t",   threshold_otsu(original_image))

    binary_image = image > threshold_image * tunable('cells.otsu_bias', otsu_bias,
                                                     description="Bias factor for the cell detection Otsu image.")

    with DebugPlot('variable_otsu_bias') as p:
        p.title("Cell detection")
        p.imshow(np.transpose(image), aspect='auto', extent=(0, image.shape[0], 10 * image.shape[1], 0))
        p.show()
        p.imshow(np.transpose(binary_image), aspect='auto', extent=(0, image.shape[0], 0, -10 * image.shape[1]))
        p.show()
        # p.imshow(np.transpose(threshold_image), aspect='auto', extent=(0, image.shape[0], 0, -10 * image.shape[1]))

        # p.legend()

    complete_image = True

    if cut_edges_from_binary_image:
        # Here I will potentially remove one or both horizontal edges taking in account shadows coming
        # from being at the edge #maybe this should have been corrected in channel_detetion and not here

        profile_of_binary_image_horizontal = horizontal_mean(binary_image.astype(float))
        if profile_of_binary_image_horizontal[0] < 0.02:
            # print("antes ", len(binary_image[0]))
            binary_image_new = [a[1:] for a in binary_image.astype(float)]
            profile_of_binary_image = vertical_mean(binary_image_new)

            # print("despues", len(binary_image_new[0]))
            complete_image = False
            # print("removed: ", 0)
            # print(profile_of_binary_image_horizontal[-1])

        if profile_of_binary_image_horizontal[-1] < 0.02:
            if complete_image:
                complete_image = False
                last_row = len(profile_of_binary_image_horizontal) - 1
            else:
                last_row = len(profile_of_binary_image_horizontal) - 2
            binary_image_new = [a[:last_row] for a in binary_image.astype(float)]
            profile_of_binary_image = vertical_mean(binary_image_new)

            binary_image = binary_image_new
            # print("removed: ", last_row)

    ############# All profiles

    profile_raw = vertical_mean(original_image)
    profile_t = vertical_mean(image)
    profile_t_bl = simple_baseline_correction(profile_t)
    profile_smooth = hamming_smooth(profile_t_bl, tunable('cells.smoothing.length', smooth,
                                                          description="Length of smoothing Hamming window for cell detection.")).round(
        8)

    profile = profile_smooth

    profile_of_binary_image = vertical_mean(binary_image)
    profile_of_binary_image = normalize2(hamming_smooth(profile_of_binary_image, smoth_binary_profile))

    ############ Finding Poles
    extrema = find_extrema_and_prominence(profile_smooth, order=tunable('cells.extrema.order', cells_extrema_order,
                                                                        description="For cell detection, window width of the local extrema detector. How many points on each side to use for the comparison to consider comparator(n, n+x) to be True."))

    possible_pos_max = [_pos for _pos in extrema.maxima]

    # merge, eliminate repetitions and sort
    possible_positions = possible_pos_max
    possible_positions = list(set(possible_positions))
    possible_positions = sorted(possible_positions)

    # evaluate prominence in all of the positions
    positions = [_pos for _pos in possible_positions if extrema.prominence[_pos] > threshold_prominence_for_poles]

    # evaluate brightness in "poles"

    # extrema_original = find_extrema_and_prominence(hamming_smooth(profile_raw, smooth), order=tunable('cells.extrema.order', cells_extrema_order,
    #                                                                     description="For cell detection, window width of the local extrema detector. How many points on each side to use for the comparison to consider comparator(n, n+x) to be True."))

    threshold_for_binary_image = normalize_specific_range(extrema.prominence, 0.7, 0.4)
    fixed_threshold_binary_image = False
    if fixed_threshold_binary_image == True:
        positions = [_pos for _pos in positions if
                     profile_of_binary_image[_pos] >= otsu_bright]
    else:
        positions = [_pos for _pos in positions if
                     profile_of_binary_image[_pos] >= threshold_for_binary_image[_pos]]

    # here I am evaluating at the two neighbor positions, in case the maxima/pole its not cach in same position as in the binary profile
    # positions_1=[]
    # for _pos in positions:
    #     evaluation = profile_of_binary_image[_pos] >= otsu_bright
    #     if _pos>0:
    #         evaluation_one_before = profile_of_binary_image[_pos-1] >= otsu_bright
    #         evaluation = evaluation or evaluation_one_before
    #     # evaluation_one_after = profile_of_binary_image[_pos+1] >= otsu_bright
    #
    #     if evaluation:
    #         positions_1.append(_pos)

    positions = [0] + positions + [profile.size - 1]

    # re-evaluation poles
    positions_ends = []
    for _last_pos, _pos in zip([0] + positions, positions):
        positions_ends.append([_last_pos, _pos])

        new_positions_ends = []
    new_positions_ends = []
    for start, end in positions_ends:

        go_right = True
        i = start
        new_start = start

        while go_right and i < end - 1:
            i = i + 1
            if profile_of_binary_image[i] >= profile_of_binary_image[i - 1] and profile_of_binary_image[i] >= \
                    threshold_for_binary_image[i]:
                new_start = i
            else:
                go_right = False

        go_left = True
        i = end
        new_end = end

        while go_left and i > start:
            i = i - 1

            if profile_of_binary_image[i] >= profile_of_binary_image[i + 1] * 0.9 and profile_of_binary_image[i] >= \
                    threshold_for_binary_image[i]:
                new_end = i
            else:
                go_left = False
        new_positions_ends.append([new_start, new_end])

    # possible positions are constructed, and a cell list is generated by checking them with the is_a_cell function
    # points from maxima criteria and gap analysis from otsu profile
    ################# Evaluation cells # Evaluation to determine if what is inside the positions is a cells
    def is_a_cell(last_pos, pos):
        """

        :param last_pos:
        :param pos:
        :return:
        """
        # based on the following filter function,
        # it will be decided whether a pair of extrema marks a cell or not
        # #1# size must be larger than zero
        # #2# the cell must have a certain 'blackness' (based on the Otsu binarization)
        # #3# the cell must have a certain prominence (difference from background brightness)

        # please note, while #1# looks like the minimum size criterion as described in the paper,
        # it is just a pre-filter, the actual minimal size filtering is done in the Cells class!
        # that way, the cell detection routine here is independent of more mundane aspects like calibration,
        # and changes in cell detection routine will still profit from the size-postprocessing

        cell = False
        size_condition = pos - last_pos > min_size

        if debug_print_cell_evaluation:
            print([_last_pos, _pos])
            print("\t size ", size_condition, pos - last_pos)

        if size_condition:
            brightness = profile_of_binary_image[last_pos:pos].mean()
            brightness_condition = brightness < tunable('cells.filtering.maximum_brightness', max_bright,
                                                        description="For cell detection, maximum brightness a cell may have.")
            if debug_print_cell_evaluation:
                print("\t bright ", brightness_condition, brightness)

            if brightness_condition:
                mean_prominence = extrema.prominence[last_pos:pos].mean()
                prominence_condition = mean_prominence > \
                                       tunable('cells.filtering.minimum_prominence', min_prominence,
                                               description="For cell detection, minimum prominence a cell must have.")
                if debug_print_cell_evaluation:
                    print("\t prom ", prominence_condition, mean_prominence)

                cell = prominence_condition

        return cell

    new_cells = []
    for _last_pos, _pos in new_positions_ends:
        # print(_last_pos, _pos)
        cell = False
        if is_a_cell(_last_pos, _pos):
            cell = True
            new_cells.append([_last_pos, _pos])

    # Print and plots to debug
    if debug_print_possible_poles:

        print("possible positions \t mean prominence \t mean otsu \t threshold bi \t positions \t ")
        for _pos in possible_positions:
            print(_pos,
                  "\t", float("{0:.2f}".format(extrema.prominence[_pos])),
                  "\t", float("{0:.3f}".format(profile_of_binary_image[_pos])),
                  "\t", float("{0:.4f}".format(threshold_for_binary_image[_pos - 1])),
                  "\t", _pos in positions)

    with DebugPlot('graph profiles') as p:
        p.plot(normalize2(profile_raw), label="profile")
        p.plot(normalize2(profile_t_bl), label="transformation and baseline correction")
        p.plot(normalize2(profile_smooth), label="smooth")
        # p.show()
        # p.plot(profile_raw, label="raw")
        # p.plot(profile_t, label="t")
        # p.plot(profile_t_bl, label="t-bl")
        # p.plot(profile_smooth, label="smooth")

        p.title("normalized vertical profiles of the channel")
        p.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    with DebugPlot('graph binary image') as p:
        p.title("profile_of_binary_image")
        p.plot(profile_of_binary_image)
        p.plot(threshold_for_binary_image)

        all_ends = list(itertools.chain.from_iterable(new_positions_ends))

        p.plot([s for s in possible_pos_max], [0.8] * len(possible_pos_max), 'o', color='m', label="possible_pos_max")
        p.plot([pos for pos in all_ends], [0.85] * len(all_ends), 'o', color='b',
               label="re-evaluation ends")
        p.plot(positions, [0.75] * len(positions), 'o', color='orange', label="positions")
        p.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    with DebugPlot('cell_detection', 'channel', 'graph') as p:
        p.title("Cell detection")
        p.imshow(np.transpose(image), aspect='auto', extent=(0, image.shape[0], 10 * image.shape[1], 0))
        p.imshow(np.transpose(binary_image), aspect='auto', extent=(0, image.shape[0], 0, -10 * image.shape[1]))

        p.legend()

        new_cell_lines = [__pos for __pos in new_cells for __pos in __pos]

        p.vlines(new_cell_lines,
                 [image.shape[1] * -10] * len(new_cell_lines),
                 [image.shape[1] * 10] * len(new_cell_lines), linestyles='dashed', colors='m')

    return [new_cells, min_and_max(profile_smooth)]
