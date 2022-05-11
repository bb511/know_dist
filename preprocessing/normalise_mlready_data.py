# Normalise numpy arrays and split them into training, validation, and testing sub
# data sets to make them ready for the machine learning algorithms.

import os
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--x_data_paths', type=str, nargs='+', required=True,
                    help='Path to the data files to process.')
parser.add_argument('--y_data_paths', type=str, nargs='+', required=True,
                    help='Paths to the target files corresponding to the data.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to the output folder.')
parser.add_argument('--output_name', type=str, required=True,
                    help='Name of the output file. Will be same for data and target.')
parser.add_argument('--norm', type=str, default='nonorm',
                    help='The type of normalisation to apply to the data.')
parser.add_argument('--max_data', type=int, default=-1,
                    help='Maximum number of events that your data set should have.')
parser.add_argument('--test_split', type=float, default=0.33,
                    help='The percentage of data to be used as validation.')

def main(args):

    print("Loading the files...\n")
    x_data = np.load(args.x_data_paths[0])
    y_data = np.load(args.y_data_paths[0])
    for x_data_path, y_data_path in zip(args.x_data_paths[1:], args.y_data_paths[1:]):
        x_data = np.concatenate((x_data, np.load(x_data_path)), axis=0)
        y_data = np.concatenate((y_data, np.load(y_data_path)), axis=0)

    x_data, y_data = equalize_classes(x_data, y_data, args.max_data)
    x_data = apply_normalisation(args.norm, x_data)
    x_data_train, x_data_test, y_data_train, y_data_test = \
        train_test_split(x_data, y_data, test_size=args.test_split, random_state=7,
                         stratify=y_data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, "x_" + args.output_name  +
            f"_n{x_data_train.shape[0]}_{args.norm}_train"), x_data_train)
    np.save(os.path.join(args.output_dir, "x_" + args.output_name +
            f"_n{x_data_test.shape[0]}_{args.norm}_test"), x_data_test)
    np.save(os.path.join(args.output_dir, "y_" + args.output_name +
            f"_n{y_data_train.shape[0]}_{args.norm}_train"), y_data_train)
    np.save(os.path.join(args.output_dir, "y_" + args.output_name +
            f"_n{y_data_test.shape[0]}_{args.norm}_test"), y_data_test)

    print('\n')
    print(tcols.HEADER + "Training data" + tcols.ENDC)
    print_jets_per_class(y_data_train)
    print('\n')
    print(tcols.HEADER + "Test data" + tcols.ENDC)
    print_jets_per_class(y_data_test)

    print("\n" + tcols.OKGREEN +
          f"Saved equalized and normalised data at {args.output_dir}" +
          " \U0001F370\U00002728" +
          tcols.ENDC)

def equalize_classes(x_data: np.ndarray, y_data: np.ndarray, max_data: int) \
    -> tuple([np.array, np.array]):
    """Equalize the number of events each class has in the data file and make sure
    they add up to the maximum amount of data given by the user. IF max_data is not
    divisible by the number of classes, then it is rounded to the nearest int.
    IF max_data is -1, use as many data points as the class with the lowest data
    points has.

    Args:
        x_data: Array containing the data to equalize.
        y_data: Corresponding onehot encoded target array.
        max_data: Maximum amount of data the total data set should have.

    Returns:
        The data with equal number of events per class and the corresp target.
        This data is not shuffled, so first max_data events will be one class, the
        next max_data events will be another class, and so on...
    """
    print("Equalizing data classes...")
    x_data_segregated, y_data_segregated = segregate_data(x_data, y_data)
    maxdata_class = check_max_data_consitency(x_data_segregated, max_data)

    x_data = x_data_segregated[0][:maxdata_class, :, :]
    y_data = y_data_segregated[0][:maxdata_class, :]
    for x_data_class, y_data_class in zip(x_data_segregated[1:], y_data_segregated[1:]):
        x_data = np.concatenate((x_data, x_data_class[:maxdata_class, :, :]), axis=0)
        y_data = np.concatenate((y_data, y_data_class[:maxdata_class, :]), axis=0)

    return x_data, y_data

def apply_normalisation(choice: str, x_data: np.array) -> np.array:
    """Choose the type of normalisation to apply to the data.

    Args:
        choice: The choice of the user with repsect to the type of norm to apply.
        x_data: Array containing the data to normalise.

    Returns:
        Normalised x_data.
    """
    if choice == 'nonorm':
        print("Skipping normalisation...")
        return x_data

    print(f"Applying {choice} normalisation...")
    switcher = {}
    x_data = switcher.get(choice, lambda: None)()

    if x_data is None:
        raise NameError("Type of normalisation does not exist! Please choose from "
                        f"the following list: {list(switcher.keys())}")

    return x_data

def segregate_data(x_data: np.array, y_data: np.array)\
     -> tuple([list[np.ndarray], list[np.ndarray]]):
    """Separates the data into separate arrays for each class.

    Args:
        x_data: Array containing the data to equalize.
        y_data: Corresponding onehot encoded target array.

    Returns:
        List of numpy arrays, each numpy array corresponding to a class of data.
        First list is for data and second list is corresponding target.
    """
    x_data_segregated = []
    y_data_segregated = []
    num_data_classes = y_data.shape[1]

    for data_class_nb in range(num_data_classes):
        class_elements_boolean = np.argmax(y_data, axis=1) == data_class_nb
        x_data_segregated.append(x_data[class_elements_boolean])
        y_data_segregated.append(y_data[class_elements_boolean])

    return x_data_segregated, y_data_segregated

def check_max_data_consitency(x_data_segregated: np.ndarray, max_data: np.ndarray):
    """Checks if the desired amount of data is consistent with the size of the data
    set that is under consideration.
    """
    num_classes = len(x_data_segregated)
    num_datapoints_per_class = [len(x_data_class) for x_data_class in x_data_segregated]

    if max_data == -1:
        desired_datapoints_per_class = min(num_datapoints_per_class)
    else:
        desired_datapoints_per_class = int(max_data/num_classes)

    if False in num_datapoints_per_class < desired_datapoints_per_class:
        raise RuntimeError("Not enough data per class to satisfy given max_data!")

    return desired_datapoints_per_class

def print_jets_per_class(y_data: np.array):
    print(f'Number of gluon jets: {np.sum(np.argmax(y_data, axis=1)==0)}')
    print(f'Number of quark jets: {np.sum(np.argmax(y_data, axis=1)==1)}')
    print(f'Number of W jets: {np.sum(np.argmax(y_data, axis=1)==2)}')
    print(f'Number of Z jets: {np.sum(np.argmax(y_data, axis=1)==3)}')
    print(f'Number of top jets: {np.sum(np.argmax(y_data, axis=1)==4)}')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
