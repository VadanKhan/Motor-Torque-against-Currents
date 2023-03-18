# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:14:48 2023

@author: Vadan Khan
"""
# %% Import Statements
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import fmin

# %% Declaring Constants
FILENAME1 = 'sim_data.csv'
FILENAME2 = 'test_data.csv'
DELIMITER = ','
COMMENT_MARKER = '%'


# %% Function Definitions
def read_data(data_input_name, delimiter_input, comment_marker_input,
              column_number):
    """
    Reads x=column_number column data file (csv or txt), with a given delimiter
    and comment marker. Converts to numpy array, with x columns.
    Then sorts the data according to the first column. This will remove lines
    of data that are not numbers, indicated those that have been eliminated.
    Parameters
    ----------
    data_input_name : string
    delimiter_input : string
    comment_marker_input : string
    column_number: int
    Returns
    -------
    numpy_array[floats]
    """
    print("\nReading File: ", data_input_name)
    print("==================================================")
    try:
        data_intake = np.genfromtxt(
            data_input_name, delimiter=delimiter_input,
            comments=comment_marker_input)
    except (ValueError, NameError, TypeError, IOError):
        return 1

    index = 0
    eliminated_lines = 0
    initial_length = len(data_intake[:, 0])

    if column_number == 1:
        for line in data_intake:
            if np.isnan(line[0]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 2:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 3:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 7:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]) or \
                    np.isnan(line[3]) or np.isnan(line[4]) or \
                    np.isnan(line[5]) or np.isnan(line[6]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1

    if eliminated_lines == 0:
        print("[no lines eliminated]")
    print("==================================================")
    print('Initial Array Length: {0}'.format(initial_length))
    print('Final Array Length: {0}'.format(len(data_intake[:, 0])))
    print('Data Read with {0} lines removed'.format(eliminated_lines))
    print("==================================================")

    return data_intake


def read_data_raw(data_input_name, delimiter_input, comment_marker_input):
    """
    Converts file to raw array.
    ----------
    data_input_name : string
    delimiter_input : string
    comment_marker_input : string
    Returns
    -------
    numpy_array[floats]
    """
    print("\nReading File: ", data_input_name)
    print("==================================================")
    try:
        data_intake = np.genfromtxt(
            data_input_name, delimiter=delimiter_input,
            comments=comment_marker_input)
    except (ValueError, NameError, TypeError, IOError):
        return 1

    return data_intake


def filter_data(data_input, column_number):
    """
    Reads x=column_number column array.
    Then sorts the data according to the first column. This will remove lines
    of data that are not numbers, indicated those that have been eliminated.
    Parameters
    ----------
    data_input : numpy_array[floats]
    column_number: int
    Returns
    -------
    numpy_array[floats]
    """
    data_intake = data_input

    print("\nFiltering Data: ")
    print("==================================================")

    index = 0
    eliminated_lines = 0
    initial_length = len(data_input[:, 0])

    if column_number == 1:
        for line in data_intake:
            if np.isnan(line[0]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 2:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 3:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 7:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]) or \
                    np.isnan(line[3]) or np.isnan(line[4]) or \
                    np.isnan(line[5]) or np.isnan(line[6]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1
    elif column_number == 13:
        for line in data_intake:
            if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]) or \
                    np.isnan(line[3]) or np.isnan(line[4]) or \
                    np.isnan(line[5]) or np.isnan(line[6]) or \
                    np.isnan(line[7]) or np.isnan(line[8]) or \
                    np.isnan(line[9]) or np.isnan(line[10]) or \
                    np.isnan(line[11]) or np.isnan(line[12]):
                print('Deleted line {0}: '.format(
                    index + 1 + eliminated_lines))
                print(line)
                data_intake = np.delete(data_intake, index, 0)
                index -= 1
                eliminated_lines += 1
            index += 1

    if eliminated_lines == 0:
        print("[no lines eliminated]")
    print("==================================================")
    print('Initial Array Length: {0}'.format(initial_length))
    print('Final Array Length: {0}'.format(len(data_intake[:, 0])))
    print('Data Read with {0} lines removed'.format(eliminated_lines))
    print("==================================================")
    return data_intake


def select_x_data(data_input, x_value_input):
    """
    Selects all the values of an array that the first column = x =
    x_value_input

    data_input : numpy_array[floats]
        Input data as a numpy array
    x_value_input : string
        Value of the first column to select
    Returns
    -------
    numpy_array[floats]
        Selected array with x_value_input as the first column value
    """
    print("\nSeparating Array by:", x_value_input)
    print("==================================================")

    try:
        width = len(data_input[0, :])

        first_column = data_input[:, 0]
        x_index_markers = np.where(first_column == x_value_input)
        # print('Selected Lines: \n', x_index_markers[0])
        selected_array = np.empty((0, width))
        for x_position in x_index_markers:
            x_line = data_input[x_position, :]
            selected_array = np.vstack((selected_array, x_line))
    except (ValueError, NameError, TypeError, IOError):
        return 2

    size = len(x_index_markers[0])
    if size == 0:
        print("[no lines found with {0} as a first column value".format(
            x_value_input))
    print('Data Read with {0} lines selected'.format(size))
    print("==================================================")
    return selected_array


def seperate_columns_234(data_input):
    """
    Takes a 2D NumPy array as input and outputs the 2nd, 3rd, and 4th columns
    as separate NumPy arrays.

    Parameters
    ----------
    data_input : numpy.ndarray
        Input data as a 2D NumPy array.

    Returns
    -------
    numpy.ndarray
        2nd column of input data.
    numpy.ndarray
        3rd column of input data.
    numpy.ndarray
        4th column of input data.
    """
    try:
        second_column = data_input[:, 1]
        third_column = data_input[:, 2]
        fourth_column = data_input[:, 3]
    except (ValueError, NameError, TypeError, IOError):
        return 2

    return second_column, third_column, fourth_column


def seperate_columns_356(data_input):
    """
    Takes a 2D NumPy array as input and outputs the 3rd, 5th and 6th columns
    as separate NumPy arrays.

    Parameters:
    input_array (numpy.ndarray): The input 2D NumPy array.

    Returns:
    numpy.ndarray: The 3rd column of the input array.
    numpy.ndarray: The 5th column of the input array.
    numpy.ndarray: The 6th column of the input array.
    """
    try:
        column3 = data_input[:, 2]
        column5 = data_input[:, 4]
        column6 = data_input[:, 5]
    except (ValueError, NameError, TypeError, IOError):
        return 2

    return column3, column5, column6


def remove_first_4_columns(data_input):
    """
    Takes a 2D NumPy array as input and removes the first 4 columns.

    Parameters
    ----------
    data_input : numpy.ndarray
        Input data as a 2D NumPy array.

    Returns
    -------
    numpy.ndarray
        Data with the first 4 columns removed.
    """
    return data_input[:, 4:]


def _3D_plot(x_values_input, y_values_input, z_values_input, x_name, y_name,
             z_name, save_name, plot_title):
    """
    Create a 3D scatter plot using the given data and labels.

    Parameters:
    x_values_input (numpy.ndarray): The x-axis values of the plot.
    y_values_input (numpy.ndarray): The y-axis values of the plot.
    z_values_input (numpy.ndarray): The z-axis values of the plot.
    x_name (str): The label for the x-axis.
    y_name (str): The label for the y-axis.
    z_name (str): The label for the z-axis.
    plot_title (str): The title of the plot.

    Returns:
    int: 0 on success.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot data
    try:
        ax.scatter(x_values_input, y_values_input, z_values_input)
    except (ValueError, NameError, TypeError, IOError):
        return 3

    # set axis labels
    ax.set_title(plot_title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    # show plot
    plt.tight_layout()
    plt.savefig(save_name + '.png', dpi=1000)
    plt.show()
    return 0


def _3D_plot_2types(x_values_input1, y_values_input1, z_values_input1,
                    x_values_input2, y_values_input2, z_values_input2,
                    x_name, y_name, z_name, dataname1, dataname2, save_name,
                    plot_title, elevation_input, azimuthal_input, point_size):
    """
    Create a 3D scatter plot using the given data and labels. Highlights two
    different data sets with different colours. Can edit orientation of view
    to check this more clearly.

    Parameters:
    x_values_input1 (numpy.ndarray): The x-axis values of the first data type.
    y_values_input1 (numpy.ndarray): The y-axis values of the first data type.
    z_values_input1 (numpy.ndarray): The z-axis values of the first data type.
    x_values_input2 (numpy.ndarray): The x-axis values of the second data type.
    y_values_input2 (numpy.ndarray): The y-axis values of the second data type.
    z_values_input2 (numpy.ndarray): The z-axis values of the second data type.
    x_name (str): The label for the x-axis.
    y_name (str): The label for the y-axis.
    z_name (str): The label for the z-axis.
    dataname1 (str): The label for the first data type.
    dataname2 (str): The label for the second data type.
    save_name (str): The name of the file to save the plot as.
    plot_title (str): The title of the plot.
    elevation_input (float): The elevation angle of the plot view.
    azimuthal_input (float): The azimuthal angle of the plot view.
    point_size (float): The size of the plot points.

    Returns:
    int: 0 on success.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # change view
    ax.view_init(elev=elevation_input, azim=azimuthal_input)

    # plot data
    try:
        ax.scatter(x_values_input1, y_values_input1, z_values_input1,
                   s=point_size, label=dataname1)
        ax.scatter(x_values_input2, y_values_input2, z_values_input2,
                   s=point_size, label=dataname2)
    except (ValueError, NameError, TypeError, IOError):
        return 3

    # set axis labels
    ax.set_title(plot_title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    # show plot
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name + '.png', dpi=1000)
    plt.show()
    return 0


# %% Main


def main_code():
    """
    Contains all executed code.

    """
    # %% Simulation Data Plotting
    sim_data_raw = read_data(FILENAME1, DELIMITER, COMMENT_MARKER, 7)
    print(sim_data_raw)
    print("="*50)

    _1000_data = select_x_data(sim_data_raw, 1000)
    _2500_data = select_x_data(sim_data_raw, 2500)
    _5000_data = select_x_data(sim_data_raw, 5000)
    _7500_data = select_x_data(sim_data_raw, 7500)
    _9000_data = select_x_data(sim_data_raw, 9000)

    Id_1000, Iq_1000, torque_1000 = seperate_columns_234(_1000_data)
    Id_2500, Iq_2500, torque_2500 = seperate_columns_234(_2500_data)
    Id_5000, Iq_5000, torque_5000 = seperate_columns_234(_5000_data)
    Id_7500, Iq_7500, torque_7500 = seperate_columns_234(_7500_data)
    Id_9000, Iq_9000, torque_9000 = seperate_columns_234(_9000_data)
    _3D_plot(Id_1000, Iq_1000, torque_1000, '$I_d$', '$I_q$', 'Torque',
             '1000s', 'Simulation Torque against $I_d$ and $I_q$ at 1000 $ms^{-1}$')
    _3D_plot(Id_2500, Iq_2500, torque_2500, '$I_d$', '$I_q$', 'Torque',
             '2500s', 'Simulation Torque against $I_d$ and $I_q$ at 2500 $ms^{-1}$')
    _3D_plot(Id_5000, Iq_5000, torque_5000, '$I_d$', '$I_q$', 'Torque',
             '5000s', 'Simulation Torque against $I_d$ and $I_q$ at 5000 $ms^{-1}$')
    _3D_plot(Id_7500, Iq_7500, torque_7500, '$I_d$', '$I_q$', 'Torque',
             '7500s', 'Simulation Torque against $I_d$ and $I_q$ at 7500 $ms^{-1}$')
    _3D_plot(Id_9000, Iq_9000, torque_9000, '$I_d$', '$I_q$', 'Torque',
             '9000s', 'Simulation Torque against $I_d$ and $I_q$ at 9000 $ms^{-1}$')

    # %% Test Data Plotting
    print("\n\n")
    test_data_raw = read_data_raw(FILENAME2, DELIMITER, COMMENT_MARKER)
    test_data_raw = remove_first_4_columns(test_data_raw)
    test_data = filter_data(test_data_raw, len(test_data_raw[0, :]))
    print(test_data)
    print("="*50)

    _9000_data_test = select_x_data(test_data_raw, 9000)

    torque_9000_test, Id_9000_test, Iq_9000_test = \
        seperate_columns_356(_9000_data_test)

    # flip sign on torque_9000_test
    torque_9000_test = -torque_9000_test

    _3D_plot(Id_9000_test, Iq_9000_test, torque_9000_test, '$I_d$', '$I_q$',
             'Torque', '9000t',
             'Test Torque against $I_d$ and $I_q$ at 9000 $ms^{-1}$')

    # %% Comparison Plots
    print("\n\n")
    _3D_plot_2types(Id_9000, Iq_9000, torque_9000,
                    Id_9000_test, Iq_9000_test, torque_9000_test,
                    '$I_d$', '$I_q$', 'Torque', 'Simulation', 'Test', 'Comp',
                    'Comparison Plot 9000 $ms^{-1}$', 30, -60, 20)
    _3D_plot_2types(Id_9000, Iq_9000, torque_9000,
                    Id_9000_test, Iq_9000_test, torque_9000_test,
                    '$I_d$', '$I_q$', 'Torque', 'Simulation', 'Test', 'Comp',
                    'Comparison Plot 9000 $ms^{-1}$', 0, -90, 5)
    _3D_plot_2types(Id_9000, Iq_9000, torque_9000,
                    Id_9000_test, Iq_9000_test, torque_9000_test,
                    '$I_d$', '$I_q$', 'Torque', 'Simulation', 'Test', 'Comp',
                    'Comparison Plot 9000 $ms^{-1}$', 0, 0, 2)

    return 0


# %% Main Execution
def main():
    """
    Main function. Will run all main_code, and should notify and exit for
    appropriate errors.
    """
    return_code = main_code()
    if return_code == 1:
        print("Error Reading File")
        sys.exit()
        return 1
    if return_code == 2:
        print("Error Array Dimensions")
        sys.exit()
        return 2
    if return_code == 3:
        print("Plotting Error")
        sys.exit()
        return 3
    return 0


if __name__ == "__main__":
    main()
