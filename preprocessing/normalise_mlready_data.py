# Normalise numpy arrays and split them into training, validation, and testing sub
# data sets to make them ready for the machine learning algorithms.

import os
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--x_data_paths",
    type=str,
    nargs="+",
    required=True,
    help="Path to the data files to process.",
)
parser.add_argument(
    "--y_data_paths",
    type=str,
    nargs="+",
    required=True,
    help="Paths to the target files corresponding to the data.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to the output folder."
)
parser.add_argument(
    "--norm",
    type=str,
    default="nonorm",
    help="The type of normalisation to apply to the data.",
)
parser.add_argument(
    "--test_split",
    type=float,
    default=0.33,
    help="The percentage of data to be used as validation.",
)


def main(args):

    print("Loading the files...\n")
    x_data = np.load(args.x_data_paths[0])
    y_data = np.load(args.y_data_paths[0])
    for x_data_path, y_data_path in zip(args.x_data_paths[1:], args.y_data_paths[1:]):
        x_data = np.concatenate((x_data, np.load(x_data_path)), axis=0)
        y_data = np.concatenate((y_data, np.load(y_data_path)), axis=0)

    x_data, y_data = equalize_classes(x_data, y_data)
    x_data = apply_normalisation(args.norm, x_data)

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data, y_data, test_size=args.test_split, random_state=7, stratify=y_data
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = format_output_filename(args.x_data_paths[0], args.norm)
    np.save(os.path.join(args.output_dir, "x_" + output_name + "_train"), x_data_train)
    np.save(os.path.join(args.output_dir, "x_" + output_name + "_test"), x_data_test)
    np.save(os.path.join(args.output_dir, "y_" + output_name + "_train"), y_data_train)
    np.save(os.path.join(args.output_dir, "y_" + output_name + "_test"), y_data_test)

    print("\n")
    print(tcols.HEADER + "Training data" + tcols.ENDC)
    print_jets_per_class(y_data_train)
    print("\n")
    print(tcols.HEADER + "Test data" + tcols.ENDC)
    print_jets_per_class(y_data_test)

    print(
        "\n"
        + tcols.OKGREEN
        + f"Saved equalized and normalised data at {args.output_dir}"
        + " \U0001F370\U00002728"
        + tcols.ENDC
    )


def equalize_classes(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple([np.array, np.array]):
    """Equalize the number of events each class has in the data file.

    Args:
        x_data: Array containing the data to equalize.
        y_data: Corresponding onehot encoded target array.

    Returns:
        The data with equal number of events per class and the corresp target.
        This data is not shuffled.
    """
    print("Equalizing data classes...")
    x_data_segregated, y_data_segregated = segregate_data(x_data, y_data)
    maxdata_class = get_min_data_per_class(x_data_segregated)

    x_data = x_data_segregated[0][:maxdata_class, :, :]
    y_data = y_data_segregated[0][:maxdata_class, :]
    for x_data_class, y_data_class in zip(x_data_segregated[1:], y_data_segregated[1:]):
        x_data = np.concatenate((x_data, x_data_class[:maxdata_class, :, :]), axis=0)
        y_data = np.concatenate((y_data, y_data_class[:maxdata_class, :]), axis=0)

    return x_data, y_data


def apply_normalisation(choice: str, x_data: np.ndarray, feat_range: tuple = (0, 1)) -> np.ndarray:
    """Choose the type of normalisation to apply to the data.

    Args:
        choice: The choice of the user with repsect to the type of norm to apply.
        x_data: Array containing the data to normalise.

    Returns:
        Normalised x_data.
    """
    if choice == "nonorm":
        print("Skipping normalisation...")
        return x_data

    print(tcols.OKGREEN + f"Applying {choice} normalisation..." + tcols.ENDC)
    switcher = {
        "minmax": lambda : minmax(x_data, feat_range),
        "robust": lambda : robust(x_data),
        "standard": lambda: standard(x_data),
    }

    x_data = switcher.get(choice, lambda: None)()

    if x_data is None:
        raise NameError(
            "Type of normalisation does not exist! Please choose from "
            f"the following list: {list(switcher.keys())}"
        )

    return x_data


def minmax(x: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
    """Applies minmax normalisation to the data, i.e., every feature of every sample
    is divided by the maximum for that respective feature.

    Args:
        x: Data array.
        feature_range: Minimum and maximum of features after the normalisation.
    """
    min_feats = x.min(axis=0).min(axis=0)
    max_feats = x.max(axis=0).max(axis=0)
    x_std = (x - min_feats)/(max_feats - min_feats)
    x_scl = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]

    return x_scl


def robust(x: np.ndarray) -> np.ndarray:
    """Applies robust normalisation to the data, i.e., the median of every feature is
    subtracted from every respective sample belonging to that feature and then each
    feature is scaled with respect to the respective inter-quantile range between
    the 1st and 3rd quantiles.

    The median of this implementation is systematically skewed with respect to the
    median determined by the sklearn implementation. However, this should not
    be important for the use scope employed here.
    """
    x_median = []
    interquantile_range = []

    for feature_idx in range(x.shape[-1]):
        x_feature = x[:, :, feature_idx].flatten()
        x_median.append(np.nanmedian(x_feature, axis=0))
        quantile75, quantile25 = np.nanpercentile(x_feature, [75, 25])
        interquantile_range.append(quantile75 - quantile25)

    x_norm = (x - x_median)/interquantile_range

    return x_norm

def standard(x: np.ndarray) -> np.ndarray:
    """Applies standard normalisation to the data, i.e., the mean of each feature is
    subtracted from every sample belonging to the respective feature and then divided
    by the corresponding standard deviation.
    """
    x_mean = []
    x_std = []

    for feature_idx in range(x.shape[-1]):
        x_feature = x[:, :, feature_idx].flatten()
        x_mean.append(x_feature.mean(axis=0))
        x_std.append(x_feature.std(axis=0))

    x_norm = (x - x_mean)/x_std

    return x_norm


def segregate_data(
    x_data: np.array, y_data: np.array
) -> tuple([list[np.ndarray], list[np.ndarray]]):
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


def get_min_data_per_class(x_data_segregated: np.ndarray):
    """Get the amount of data the class with the lowest representation has."""
    num_classes = len(x_data_segregated)
    num_datapoints_per_class = [len(x_data_class) for x_data_class in x_data_segregated]
    desired_datapoints_per_class = min(num_datapoints_per_class)

    return desired_datapoints_per_class


def format_output_filename(input_name: str, norm_name: str) -> str:
    """Formats the name of the output file given a certain convention so the data
    loading for the ml models is easier.
    """
    input_name_separated = os.path.basename(input_name).split("_")
    input_base_name = input_name_separated[1:-1]

    output_filename = "_".join(input_base_name)

    return output_filename + "_" + norm_name


def print_jets_per_class(y_data: np.array):
    print(f"Number of gluon jets: {np.sum(np.argmax(y_data, axis=1)==0)}")
    print(f"Number of quark jets: {np.sum(np.argmax(y_data, axis=1)==1)}")
    print(f"Number of W jets: {np.sum(np.argmax(y_data, axis=1)==2)}")
    print(f"Number of Z jets: {np.sum(np.argmax(y_data, axis=1)==3)}")
    print(f"Number of top jets: {np.sum(np.argmax(y_data, axis=1)==4)}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
