# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

import numpy as np

from .tracking_output import s_to_h
from ..generic.etc import QuickTableDumper
from enum import Enum;
from numpy.linalg import norm;
from math import floor, ceil, hypot;

try:
    # noinspection PyUnresolvedReferences
    import cPickle

    pickle = cPickle
except ImportError:
    import pickle


def interactive_ground_truth_main(args, tracked_results):
    """
    Ground truth mode entry function.

    :param args:
    :param tracked_results:
    :return: :raise SystemExit:
    """

    class Mode(Enum):
        Default = 0
        UpperPosition = 1
        LowerPosition = 2
        AllUpperPosition = 3
        AllLowerPosition = 4
        RectangleSelector = 5
        RectangleSelectorErased = 6

    pos = args.positions_to_process[0];
    assert (len(args.positions_to_process) == 1);

    mode = Mode(Mode.Default);

    acceptable_pos_chans = \
        {p: list(range(len(tracked_results[list(tracked_results.keys())[p]].channel_accumulator.keys())))
         for p in range(len(tracked_results.keys())) if
         len(tracked_results[list(tracked_results.keys())[p]].channel_accumulator.keys()) > 0}

    def plots_info():
        """
        Outputs some information about the data set.
        """

        print("Positions " + str(list(tracked_results.keys())))
        print("Acceptable channels per position " + repr(acceptable_pos_chans))

    plots_info()

    ground_truth_data = args.ground_truth

    # noinspection PyUnresolvedReferences
    try:
        with open(ground_truth_data, 'rb') as fp:
            all_envs = pickle.load(fp)
    except FileNotFoundError:
        print("File did not exist, starting anew")
        all_envs = {}

    def save_data():
        """
        Saves the ground truth data to the file specified. (pickled data)

        """
        with open(ground_truth_data, 'wb+') as inner_fp:
            pickle.dump(all_envs, inner_fp, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved data to %s" % (ground_truth_data,))

    lowest_position = min(acceptable_pos_chans.keys())
    highest_position = max(acceptable_pos_chans.keys())

    next_dataset = [lowest_position, next(iter(acceptable_pos_chans[lowest_position]))]

    # loop controller
    toContinue = {};
    toContinue['value'] = True;

    # rectangle selection recorder
    rectangle_selection_top_recorder = [False] * (len(tracked_results[pos].channel_accumulator[0]));
    rectangle_selection_bottom_recorder = [False] * (len(tracked_results[pos].channel_accumulator[0]));

    def perform_it():
        """
        Runs the ground truth mode.

        :return: :raise SystemExit:
        """
        next_pos, next_chan = next_dataset

        def empty_env():
            """
            Generates an empty environment.

            :return:
            """
            return {
                'points': np.ma.array(np.zeros((1024, 3))),
                'points_empty': np.ma.array(np.zeros((1024, 3))),
                'used': 0,
                'last_point_x': None,
                'last_point_y': None,
            }

        if (next_pos, next_chan) not in all_envs:
            all_envs[(next_pos, next_chan)] = empty_env()

        env = all_envs[(next_pos, next_chan)]

        pos = list(tracked_results.keys())[next_pos]
        tracking = tracked_results[pos]

        chan_num = list(tracking.channel_accumulator.keys())[next_chan]

        channels = tracking.channel_accumulator[chan_num]

        print("Opening position %d, channel %d" % (pos, chan_num,))

        data = np.zeros((len(channels), 6))

        n_timepoint, n_width, n_height, n_top, n_bottom, n_width_cumsum = 0, 1, 2, 3, 4, 5

        some_channel_image = None

        for n, cc in enumerate(channels):
            data[n, n_timepoint] = cc.image.timepoint
            data[n, n_width] = cc.channel_image.shape[1]
            data[n, n_height] = cc.channel_image.shape[0]
            data[n, n_top] = cc.top
            data[n, n_bottom] = cc.bottom
            some_channel_image = cc.channel_image

        data[:, n_width_cumsum] = np.cumsum(data[:, n_width])

        max_top, min_top = data[:, n_top].max(), data[:, n_top].min()
        max_bottom, min_bottom = data[:, n_bottom].max(), data[:, n_bottom].min()

        low, high = int(np.floor(min_top)), int(np.ceil(max_bottom))

        large_image = np.zeros((high - low, int(data[-1, n_width_cumsum])), dtype=some_channel_image.dtype)

        for n, cc in enumerate(channels):
            lower_border = int(np.floor(data[n, n_top] - low))
            large_image[
            lower_border:int(lower_border + data[n, n_height]),
            int(data[n, n_width_cumsum] - data[n, n_width]):int(data[n, n_width_cumsum])
            ] = cc.channel_image

        import matplotlib.pyplot as plt

        # import pdb;pdb.set_trace()
        # plt.clf();
        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(50, 5))
        ax = fig.add_axes([.1, .1, .85, .85])

        # plt.subplots_adjust(left=0.25, bottom=0.25)
        # plt.subplots_adjust(top=0.9)

        if len(tracked_results[0].channel_accumulator[0]) == len(args.timepoints_to_process):
            fig.canvas.set_window_title("Default mode: channel " + str(chan_num))
        else:
            fig.canvas.set_window_title("Certain frames were excluded from analysis: channel " + str(chan_num))

        channels_per_inch = 5.0
        # plt.rcParams['figure.figsize'] = (len(channels) / channels_per_inch, 4.0)
        plt.rcParams['figure.figsize'] = [50, 5];
        # plt.rcParams['figure.dpi'] = 600;

        # plt.rcParams['figure.subplot.top'] = 0.8
        # plt.rcParams['figure.subplot.bottom'] = 0.2
        # plt.rcParams['figure.subplot.left'] = 0.2
        # plt.rcParams['figure.subplot.right'] = 0.8

        plt.rcParams['image.cmap'] = 'gray'

        # zoom = 5;
        # ax.rcParams["figure.figsize"] = [fig.get_figwidth() * zoom, fig.get_figheight() * zoom];

        def x_coordinate_to_timepoint(x, to_integer):

            # the ordinary case: all timepoints are under processing
            timepoint = int(to_integer(x / data[0, n_width] - 0.5));
            if timepoint < len(tracked_results[pos].channel_accumulator[chan_num]):
                if data[timepoint - 1, n_width_cumsum] <= x <= data[timepoint, n_width_cumsum]:
                    return timepoint;

            timepoint = [];

            for timepoint, channel in enumerate(tracked_results[pos].channel_accumulator[chan_num]):
                if data[timepoint, n_width_cumsum] >= x:
                    return timepoint;

        def timepoint_to_x_coordinate(timepoint):
            # x = (timepoint + 0.5) * 8;
            # return x;
            return data[timepoint, n_width_cumsum] - 0.5 * data[timepoint, n_width];

        plotBottom = [];
        plotTop = [];
        plotSelection = [];

        def show_cell_bounaries():

            xPos = [None] * (len(tracked_results[pos].channel_accumulator[chan_num]));
            bottom = [None] * (len(tracked_results[pos].channel_accumulator[chan_num]));
            top = [None] * (len(tracked_results[pos].channel_accumulator[chan_num]));

            for timepoint, channel in enumerate(tracked_results[pos].channel_accumulator[chan_num]):

                xPos[timepoint] = timepoint_to_x_coordinate(timepoint);

                if channel.cells.cells_list:
                    bottom[timepoint] = channel.cells.cells_list[0].local_bottom;
                    top[timepoint] = channel.cells.cells_list[0].local_top;

            nonlocal plotBottom;
            nonlocal plotTop;
            plotBottom, = plt.plot(xPos, bottom, "r+");
            plotTop, = plt.plot(xPos, top, "b+");

        plt.imshow(large_image, aspect='auto', interpolation='none')
        show_cell_bounaries();

        # fig.tight_layout()

        o_scatter = ax.scatter([], [])
        o_lines, = plt.plot([], [])

        def refresh():
            """
            Refreshs the overlay.

            """
            o_lines.set_data(env['points'][:env['used'], 0], env['points'][:env['used'], 1])

            o_scatter.set_offsets(env['points'][:env['used'], :2])
            fig.canvas.draw()

        def show_help():
            """
            Shows a help text for the ground truth mode.

            """
            print("""
            Ground Truth Mode:
            = Mouse =====================================
            Mark division events by right click:
            First a division, then a child's division.
            = Keys ======================================
            h       show  this help
            p       print growth rates
                    (last is based on mean division time)
            d       delete last division event
            n/N     next/previous multipoint
            m/M     next/previous channel
            o       output tabular data
            w       write data
                    (to previously specified filename)
            i       start interactive python console
            q       quit ground truth mode

            t        edit upper bound of a cell; twice to exit current mode
            ctrl + t edit upper bound of all cells; twice to exit current mode
            b        edit lower bound of a cell; twice to exit current mode
            ctrl + b edit lower bound of all cells; twice to exit current mode
            ctrl + r rectangle selector mode
            """)

        refresh()

        show_help()

        def click(e):
            """

            :param e:
            :return:
            """

            x, y = e.xdata, e.ydata
            if x is None or y is None:
                return

            nonlocal plotBottom;
            nonlocal plotTop;

            if e.button == 1:
                if mode == Mode.UpperPosition:

                    timepoint = x_coordinate_to_timepoint(x, round);
                    tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[0].local_top = y;
                    plotTop.remove();
                    show_cell_bounaries();
                    return;

                elif mode == Mode.LowerPosition:

                    timepoint = x_coordinate_to_timepoint(x, round);
                    tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[0].local_bottom = y;
                    plotBottom.remove();
                    show_cell_bounaries();
                    return;

                elif mode == Mode.AllUpperPosition:

                    for timepoint, channel in enumerate(tracked_results[pos].channel_accumulator[chan_num]):
                        channel.cells.cells_list[0].local_top = y;
                    plotTop.remove();
                    show_cell_bounaries();
                    return;

                elif mode == Mode.AllLowerPosition:

                    for timepoint, channel in enumerate(tracked_results[pos].channel_accumulator[chan_num]):
                        channel.cells.cells_list[0].local_bottom = y;
                    plotBottom.remove();
                    show_cell_bounaries();
                    return;

                elif mode == Mode.RectangleSelector:

                    selectedXPositions, selectedYPositions = points_by_rectangle_selection();
                    minDistance = 1e10;
                    minPosX = [];
                    minPosY = [];
                    for selectedXPosition, selectedYPosition in zip(selectedXPositions, selectedYPositions):
                        distance = hypot(x - selectedXPosition, y - selectedYPosition);
                        if (minDistance != min(minDistance, distance)):
                            minDistance = min(minDistance, distance);
                            minPosX = selectedXPosition;
                            minPosY = selectedYPosition;

                    print(minPosX, minPosY);

                    # update this in cell boundaries
                    distanceToChange = y - selectedYPosition;
                    for selectedXPosition, selectedYPosition in zip(selectedXPositions, selectedYPositions):
                        timepoint = x_coordinate_to_timepoint(selectedXPosition, round);
                        if (tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[
                            0].local_top == selectedYPosition):
                            tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[
                                0].local_top += distanceToChange;
                        if (tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[
                            0].local_bottom == selectedYPosition):
                            tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[
                                0].local_bottom += distanceToChange;
                    plotTop.remove();
                    plotBottom.remove();
                    show_cell_bounaries();



            elif e.button == 3:
                last_point_x, last_point_y = env['last_point_x'], env['last_point_y']

                if last_point_x is not None:

                    if env['used'] + 3 >= env['points'].shape[0]:
                        oldmask = env['points'].mask[:env['used']]
                        env['points'] = np.ma.array(np.r_[env['points'], env['points_empty']])
                        env['points'].mask = np.zeros_like(env['points']).astype(np.dtype(bool))  # [:env['used']]
                        env['points'].mask[:env['used']] = oldmask

                    n_x = np.searchsorted(data[:, n_width_cumsum], x, side='right')
                    n_last_x = np.searchsorted(data[:, n_width_cumsum], last_point_x, side='right')

                    if x < last_point_x:
                        x, y, last_point_x, last_point_y = last_point_x, last_point_y, x, y
                        n_x, n_last_x = n_last_x, n_x

                    env['points'][env['used'], 0] = last_point_x
                    env['points'][env['used'], 1] = last_point_y
                    env['points'][env['used'], 2] = data[n_last_x, n_timepoint]
                    env['used'] += 1
                    env['points'][env['used'], 0] = x
                    env['points'][env['used'], 1] = y
                    env['points'][env['used'], 2] = data[n_x, n_timepoint]
                    env['used'] += 1
                    env['points'][env['used'], :] = np.ma.masked
                    env['used'] += 1

                    refresh()

                    # print(n, data[n, n_timepoint])
                    env['last_point_x'], env['last_point_y'] = None, None
                else:
                    env['last_point_x'], env['last_point_y'] = x, y

        def ModeToggleSwitch(mode, targetMode, title):
            if mode == targetMode:
                mode = Mode.Default
                if len(tracked_results[0].channel_accumulator[0]) == len(args.timepoints_to_process):
                    fig.canvas.set_window_title("Default mode: channel " + str(chan_num))
                else:
                    fig.canvas.set_window_title("Certain frames were excluded from analysis: channel " + str(chan_num))
            else:
                mode = targetMode
                fig.canvas.set_window_title(title)
            return mode;

        def key_press(event):
            """

            :param event:
            :return: :raise SystemExit:
            """
            nonlocal mode;

            def show_stats():
                """
                Shows statistics.

                """
                inner_times = env['points'][:env['used'], 2].compressed()
                inner_times = inner_times.reshape(inner_times.size // 2, 2)
                inner_deltas = inner_times[:, 1] - inner_times[:, 0]
                inner_deltas /= 60.0 * 60.0
                inner_mu = np.log(2) / inner_deltas
                print(inner_mu, np.mean(inner_mu))

            def try_new_poschan(p, c):
                """

                :param p:
                :param c:
                :return:
                """
                # import pdb;pdb.set_trace()
                next_pos, next_chan = next_dataset

                if p == 1:
                    while (next_pos + p) not in acceptable_pos_chans and (next_pos + p) < highest_position:
                        p += 1
                elif p == -1:
                    while (next_pos + p) not in acceptable_pos_chans and (next_pos + p) > lowest_position:
                        p -= 1

                if (next_pos + p) not in acceptable_pos_chans:
                    print("Position does not exist")
                    return

                if p != 0:
                    c = 0
                    next_chan = acceptable_pos_chans[next_pos + p][0]

                if c == 1:
                    while (next_chan + c) not in acceptable_pos_chans[next_pos + p] and (next_chan + c) < max(
                            acceptable_pos_chans[next_pos + p]):
                        c += 1
                elif c == -1:
                    while (next_chan + c) not in acceptable_pos_chans[next_pos + p] and (next_chan + c) > min(
                            acceptable_pos_chans[next_pos + p]):
                        c -= 1

                if (next_chan + c) not in acceptable_pos_chans[next_pos + p]:
                    print("Channel does not exist")
                    return

                next_dataset[0] = next_pos + p
                next_dataset[1] = next_chan + c

                plt.close()

            # if event.key == 'left':
            # timepoint.set_val(max(1, int(timepoint.val) - 1))
            # elif event.key == 'right':
            #     timepoint.set_val(min(tp_max, int(timepoint.val) + 1))
            # elif event.key == 'ctrl+left':
            #     timepoint.set_val(max(1, int(timepoint.val) - 10))
            # elif event.key == 'ctrl+right':
            #     timepoint.set_val(min(tp_max, int(timepoint.val) + 10))
            # elif event.key == 'down':
            #     multipoint.set_val(max(1, int(multipoint.val) - 1))
            # elif event.key == 'up':
            #     multipoint.set_val(min(mp_max, int(multipoint.val) + 1))
            if event.key == 'h':
                show_help()
            elif event.key == 'p':
                show_stats()
            elif event.key == 'd':
                env['used'] -= 3
                show_stats()
                refresh()
            # n next position, m next channel

            elif event.key == 'n':
                try_new_poschan(1, 0)
            elif event.key == 'N':
                try_new_poschan(-1, 0)
            elif event.key == 'm':
                try_new_poschan(0, 1)
            elif event.key == 'M':
                try_new_poschan(0, -1)
            elif event.key == 'o':
                # output

                out = QuickTableDumper()

                for (t_pos, t_chan), t_env in all_envs.items():
                    x_pos = list(tracked_results.keys())[t_pos]
                    x_chan = list(tracked_results[x_pos].channel_accumulator.keys())[t_chan]
                    times = t_env['points'][:t_env['used'], 2].compressed()
                    times = times.reshape(times.size // 2, 2)
                    deltas = times[:, 1] - times[:, 0]
                    mu = np.log(2) / (deltas / (60.0 * 60.0))

                    for num in range(len(mu)):
                        out.add({
                            'position': x_pos,
                            'channel': x_chan,
                            'growth_rate': mu[num],
                            'growth_rate_channel_mean': np.mean(mu),
                            'division_age': s_to_h(deltas[num]),
                            'growth_start': s_to_h(times[num, 0]),
                            'growth_end': s_to_h(times[num, 1]),
                        })

            elif event.key == 'w':
                save_data()
            elif event.key == 'i':
                import code
                code.InteractiveConsole(locals=globals()).interact()
            elif event.key == 'q':
                # exit the current loop
                toContinue['value'] = False;
                return;
            elif event.key == 't':
                mode = ModeToggleSwitch(mode, Mode.UpperPosition, "Editing cell upper bounday");
            elif event.key == 'b':
                mode = ModeToggleSwitch(mode, Mode.LowerPosition, "Editing cell bottom bounday");
            elif event.key == 'ctrl+t':
                mode = ModeToggleSwitch(mode, Mode.AllUpperPosition, "Editing ALL cells' upper bounday");
            elif event.key == 'ctrl+b':
                mode = ModeToggleSwitch(mode, Mode.AllLowerPosition, "Editing ALL cells' bottom bounday");
            elif event.key == 'ctrl+r':
                mode = ModeToggleSwitch(mode, Mode.RectangleSelector, "Rectangle Selector");

            elif event.key == 'ctrl+e':

                mode = ModeToggleSwitch(mode, Mode.RectangleSelectorErased, "Rectangle Selector Erased.");
                # reset rectangle selection recorder
                nonlocal rectangle_selection_top_recorder;
                nonlocal rectangle_selection_bottom_recorder;

                rectangle_selection_top_recorder = [False] * (len(tracked_results[pos].channel_accumulator[chan_num]));
                rectangle_selection_bottom_recorder = [False] * (
                    len(tracked_results[pos].channel_accumulator[chan_num]));

                nonlocal plotSelection;
                if plotSelection:
                    plotSelection.remove();
                    plotSelection, = plt.plot(0, 0);
                    return;

            toContinue['value'] = True;

        def test_release(erelease):
            x2, y2 = erelease.xdata, erelease.ydata
            print(x2, y2)

        def points_by_rectangle_selection():

            nonlocal rectangle_selection_top_recorder;
            nonlocal rectangle_selection_bottom_recorder;

            x = [];
            y = [];

            for timepoint, channel in enumerate(tracked_results[pos].channel_accumulator[chan_num]):
                if rectangle_selection_top_recorder[timepoint]:
                    xPos = timepoint_to_x_coordinate(timepoint);
                    x.append(xPos);
                    y.append(channel.cells.cells_list[0].local_top);
                    print('U', timepoint);
                if rectangle_selection_bottom_recorder[timepoint]:
                    xPos = timepoint_to_x_coordinate(timepoint);
                    x.append(xPos);
                    y.append(channel.cells.cells_list[0].local_bottom);
                    print('B', timepoint);
            return (x, y);

        def rectangle_selection(epress, erelease):

            x, y = epress.xdata, epress.ydata;
            x2, y2 = erelease.xdata, erelease.ydata

            nonlocal mode;
            if mode == Mode.RectangleSelector:
                print(x, y, x2, y2)

                nonlocal rectangle_selection_top_recorder;
                nonlocal rectangle_selection_bottom_recorder;

                leftBoundary = max(0, x_coordinate_to_timepoint(x, round));
                rightBoundary = min(len(tracked_results[pos].channel_accumulator[chan_num]),
                                    x_coordinate_to_timepoint(x2, ceil)) + 1;

                print(leftBoundary, rightBoundary);

                for timepoint in range(leftBoundary, rightBoundary):

                    top = tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[0].local_top;

                    if (y <= top <= y2):
                        rectangle_selection_top_recorder[timepoint] = not rectangle_selection_top_recorder[timepoint];

                    bottom = tracked_results[pos].channel_accumulator[chan_num][timepoint].cells.cells_list[
                        0].local_bottom;

                    if (y <= bottom <= y2):
                        rectangle_selection_bottom_recorder[timepoint] = not rectangle_selection_bottom_recorder[
                            timepoint];

                xPos, yPos = points_by_rectangle_selection();

                nonlocal plotSelection;
                if plotSelection:
                    plotSelection.remove();
                plotSelection, = plt.plot(xPos, yPos, 'ko', markerfacecolor='None');

        from matplotlib.widgets import RectangleSelector

        click.RS = RectangleSelector(ax, rectangle_selection,
                                     drawtype='box', useblit=True,
                                     button=[1, 3],  # don't use middle button
                                     minspanx=5, minspany=5,
                                     spancoords='pixels',
                                     interactive=True)

        fig.canvas.mpl_connect('key_press_event', key_press)
        # fig.canvas.mpl_connect('button_release_event', test_release)
        fig.canvas.mpl_connect('button_press_event', click)

        plt.show()

    while toContinue['value']:
        perform_it();

    return tracked_results;
