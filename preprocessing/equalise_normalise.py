# Normalise numpy arrays and split them into training, validation, and testing sub
# data sets to make them ready for the machine learning algorithms.

import os
import argparse

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

import feature_selection
import standardisation
import plots
import util
from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--x_data_path_train",
    type=str,
    required=True,
    help="Path to the training data file to process.",
)
parser.add_argument(
    "--y_data_path_train",
    type=str,
    required=True,
    help="Paths to the training target file corresponding to the data.",
)
parser.add_argument(
    "--x_data_path_test",
    type=str,
    required=True,
    help="Path to the test data file to process.",
)
parser.add_argument(
    "--y_data_path_test",
    type=str,
    required=True,
    help="Paths to the test target file corresponding to the data.",
)
parser.add_argument(
    "--feats",
    type=str,
    default="andre",
    choices=["andre", "jedinet"],
    help="The type of feature selection to be employed.",
)
parser.add_argument(
    "--norm",
    type=str,
    default="nonorm",
    choices=["standard", "robust", "minmax"],
    help="The type of normalisation to apply to the data.",
)
parser.add_argument(
    "--val_split",
    type=float,
    default=0.5,
    help="The percentage of data to be used as validation from the test set.",
)
parser.add_argument(
    "--shuffle_seed",
    type=int,
    default=123,
    help="Seed for shuffling the jets in the data set."
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to the output folder."
)


def main(args):
    print("Loading the files...\n")
    x_data_train = np.load(args.x_data_path_train, "r")
    y_data_train = np.load(args.y_data_path_train, "r")
    x_data_test = np.load(args.x_data_path_test, "r")
    y_data_test = np.load(args.y_data_path_test, "r")

    x_data_train, y_data_train = util.equalise_classes(x_data_train, y_data_train)
    x_data_test, y_data_test = util.equalise_classes(x_data_test, y_data_test)

    x_data = np.concatenate((x_data_train, x_data_test), axis=0)
    y_data = np.concatenate((y_data_train, y_data_test), axis=0)
    del x_data_train
    del x_data_test
    x_data = feature_selection.get_features_numpy(x_data, args.feats)

    print("Normalising and plotting data...")
    x_data = standardisation.apply_standardisation(args.norm, x_data)
    plots_folder = format_output_filename(args.x_data_path_train, args.feats, args.norm)
    plots_path = os.path.join(args.output_dir, plots_folder)
    plots.constituent_number(plots_path, x_data)
    plots.normalised_data(plots_path, x_data, y_data)

    x_data_train = x_data[:y_data_train.shape[0]]
    x_data_test = x_data[y_data_train.shape[0]:]
    del x_data

    y_category = np.argwhere(y_data_test == 1)[:, 1]
    val_idx, test_idx = next(
        StratifiedKFold(n_splits=int(1/args.val_split)).split(x_data_test, y_category)
    )
    x_data_val, y_data_val = x_data_test[val_idx], y_data_test[val_idx],
    x_data_test, y_data_test = x_data_test[test_idx], y_data_test[test_idx]

    print("Shuffling jets...")
    ntrain_events = x_data_train.shape[0]
    nvalid_events = x_data_val.shape[0]
    ntest_events = x_data_test.shape[0]

    shuffle = np.random.RandomState(seed=args.shuffle_seed).permutation(ntrain_events)
    x_data_train, y_data_train = x_data_train[shuffle], y_data_train[shuffle]

    shuffle = np.random.RandomState(seed=args.shuffle_seed).permutation(nvalid_events)
    x_data_val, y_data_val = x_data_val[shuffle], y_data_val[shuffle]

    shuffle = np.random.RandomState(seed=args.shuffle_seed).permutation(ntest_events)
    x_data_test, y_data_test = x_data_test[shuffle], y_data_test[shuffle]

    print("Shuffling constituents...")
    rng = np.random.default_rng(args.shuffle_seed)
    tr_seeds = rng.integers(low=0, high=10000, size=ntrain_events)
    va_seeds = rng.integers(low=0, high=10000, size=nvalid_events)
    te_seeds = rng.integers(low=0, high=10000, size=ntest_events)
    
    x_data_train = shuffle_constituents(x_data_train, tr_seeds)
    x_data_val = shuffle_constituents(x_data_val, va_seeds)
    x_data_test = shuffle_constituents(x_data_test, te_seeds)

    print(tcols.OKGREEN + f"Shuffling done! \U0001F0CF" + tcols.ENDC)

    print("\n")
    print(tcols.HEADER + "Training data" + tcols.ENDC)
    print_jets_per_class(y_data_train)
    print("\n")
    print(tcols.HEADER + "Validation data" + tcols.ENDC)
    print_jets_per_class(y_data_val)
    print("\n")
    print(tcols.HEADER + "Test data" + tcols.ENDC)
    print_jets_per_class(y_data_test)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = format_output_filename(args.x_data_path_train, args.feats, args.norm)

    np.save(os.path.join(args.output_dir, "x_" + output_name + "_train"), x_data_train)
    np.save(os.path.join(args.output_dir, "x_" + output_name + "_val"), x_data_val)
    np.save(os.path.join(args.output_dir, "x_" + output_name + "_test"), x_data_test)
    np.save(os.path.join(args.output_dir, "y_" + output_name + "_train"), y_data_train)
    np.save(os.path.join(args.output_dir, "y_" + output_name + "_val"), y_data_val)
    np.save(os.path.join(args.output_dir, "y_" + output_name + "_test"), y_data_test)

    print(tcols.OKGREEN + f"\nSaved data at {args.output_dir}." + tcols.ENDC)


# def write_to_tfrecord(output_file_path: str, x: np.ndarray, y: np.ndarray, feats: str):
#     print(f"\nConverting to TFrecord and saving to: {output_file_path}.tfrecord")
#     with tf.io.TFRecordWriter(output_file_path + ".tfrecord") as writer:
#         for jet_idx in range(x.shape[0]):
#             features = tf.train.Features(
#                 feature=feature_selection.get_features(x[jet_idx], y[jet_idx], feats)
#             )
#             example = tf.train.Example(features=features)
#             writer.write(example.SerializeToString())
#             print(f"\rProgress: {jet_idx/x.shape[0]:.1%}", end='')


def shuffle_constituents(data: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Shuffles the constituents based on an array of seeds.

    Note that each jet's constituents is shuffled with respect to a seed that is fixed.
    This seed is different for each jet.
    """

    for jet_idx, seed in enumerate(seeds):
        shuffling = np.random.RandomState(seed=seed).permutation(data.shape[1])
        data[jet_idx, :] = data[jet_idx, shuffling]

    return data

def format_output_filename(input_name: str, feats_sel: str, norm_name: str) -> str:
    """Formats the name of the output file given a certain convention so the data
    loading for the ml models is easier.
    """
    input_name_separated = os.path.basename(input_name).split("_")
    input_base_name = input_name_separated[1:-1]

    output_filename = "_".join(input_base_name)

    return output_filename + "_" + feats_sel + "_" + norm_name


def print_jets_per_class(y_data: np.array):
    print(f"Number of gluon jets: {np.sum(np.argmax(y_data, axis=1)==0)}")
    print(f"Number of quark jets: {np.sum(np.argmax(y_data, axis=1)==1)}")
    print(f"Number of W jets: {np.sum(np.argmax(y_data, axis=1)==2)}")
    print(f"Number of Z jets: {np.sum(np.argmax(y_data, axis=1)==3)}")
    print(f"Number of top jets: {np.sum(np.argmax(y_data, axis=1)==4)}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
