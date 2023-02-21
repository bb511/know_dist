# Handles the data importing and small preprocessing for the interaction network.

import os
import numpy as np

from .terminal_colors import tcols


class Data:
    """Data class to store the data to be used in learning for the interaction network.

    Attributes:
        fpath: Path to where the data files are located.
        fname: Name of the data to import, without train, test, val flags.
        train_events: Number of events for the training data, -1 to use all.
        val_events: Number of events for the validaiton data, -1 to use all.
        test_events: Number of events for the testing data, -1 to use all.
        jet_seed: Seed used in shuffling the jets.
        seed: The seed used in any shuffling that is done to the data.
    """

    def __init__(
        self,
        fpath: str,
        fname: str,
        train_events: int = -1,
        val_events: int = -1,
        test_events: int = -1,
        jet_seed: int = None,
        seed: int = None,
    ):

        self._fpath = fpath
        self._fname = fname

        self.jet_seed = jet_seed
        self.seed = seed

        self.tr_data, self.tr_target = self._load_data("train", train_events)
        self.va_data, self.va_target = self._load_data("val", val_events)
        self.te_data, self.te_target = self._load_data("test", test_events)

        self.ntrain_jets = self.tr_data.shape[0]
        self.nval_jets = self.va_data.shape[0]
        self.ntest_jets = self.te_data.shape[0]
        self.ncons = self.tr_data.shape[1]
        self.nfeat = self.tr_data.shape[2]

        self._success_message()

    @classmethod
    def shuffled(
        cls,
        fpath: str,
        fname: str,
        train_events: int = -1,
        val_events: int = -1,
        test_events: int = -1,
        jet_seed: int = None,
        seed: int = None,
    ):
        """Shuffles the constituents. The jets are shuffled regardless."""
        data = cls(fpath, fname, train_events, val_events, test_events, jet_seed, seed)

        print("Shuffling constituents...")
        rng = np.random.default_rng(seed)
        tr_seeds = rng.integers(low=0, high=10000, size=data.ntrain_jets)
        va_seeds = rng.integers(low=0, high=10000, size=data.nval_jets)
        te_seeds = rng.integers(low=0, high=10000, size=data.ntest_jets)

        cls._shuffle_constituents(data.tr_data, tr_seeds, "training")
        cls._shuffle_constituents(data.va_data, va_seeds, "validation")
        cls._shuffle_constituents(data.te_data, te_seeds, "testing")

        return data

    @classmethod
    def _shuffle_constituents(cls, data: np.ndarray, seeds: np.ndarray, dtype: str):
        """Shuffle the constituents of a jet given an array of seeds.

        The number of seeds coincides with the number of jets.

        Args:
            data: Array containing the jet, constituents, and features.
            seeds: Array containing the seeds, equal in number to the jets.
            dtype: The type of data to shuffle, training or testing.

        Returns:
            Shuffled (at constituent level) data.
        """

        if data.shape[0] == 0:
            return data

        for jet_idx, seed in enumerate(seeds):
            shuffling = np.random.RandomState(seed=seed).permutation(data.shape[1])
            data[jet_idx, :] = data[jet_idx, shuffling]

        print(tcols.OKGREEN + f"Shuffled the {dtype} data! \U0001F0CF\n" + tcols.ENDC)

        return data

    def _load_data(self, data_type: str, nevents: int) -> (np.ndarray, np.ndarray):
        """Load data from the data files generated by the pre-processing scripts.

        Args:
            data_type: The type of data that you want to load: val, train or test.
            nevents: Number of total events to trim the data to.

        Returns:
            Two numpy arrays with loaded data and the corresponding target.
        """
        datafile_name = "x_" + self._fname + "__" + data_type + ".npy"
        datafile_path = os.path.join(self._fpath, datafile_name)

        targetfile_name = "y_" + self._fname + "__" + data_type + ".npy"
        targetfile_path = os.path.join(self._fpath, targetfile_name)

        x = np.load(datafile_path, "r+")
        y = np.load(targetfile_path, "r+")

        x, y = self._trim_data(x, y, nevents, self.jet_seed)

        return x, y

    def _trim_data(self, x: np.ndarray, y: np.ndarray, maxdata: int, seed: int):
        """Cut the imported data and target and form a smaller data set.

        The number of events per class remains equal.

        Args:
            x: Numpy array containing the data.
            y: Numpy array containing the corresponding target (one-hot).
            maxdata: Maximum number of jets to load.
            seed: Seed to used in shuffling the jets after trimming.

        Returns:
            Two numpy arrays, one with data and one with target, containing an equal
            number of events per each class.
        """

        if maxdata < 0:
            shuffling = np.random.RandomState(seed=seed).permutation(x.shape[0])
            return x[shuffling], y[shuffling]

        num_classes = y.shape[1]
        maxdata_class = int(int(maxdata) / num_classes)

        x_segregated, y_segregated = self._segregate_data(x, y)

        x = x_segregated[0][:maxdata_class, :, :]
        y = y_segregated[0][:maxdata_class, :]

        for x_class, y_class in zip(x_segregated[1:], y_segregated[1:]):
            x = np.concatenate((x, x_class[:maxdata_class, :, :]), axis=0)
            y = np.concatenate((y, y_class[:maxdata_class, :]), axis=0)

        shuffling = np.random.RandomState(seed=seed).permutation(x.shape[0])

        return x[shuffling], y[shuffling]

    def _segregate_data(self, x_data: np.array, y_data: np.array):
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

    def _success_message(self):
        # Display success message for loading data when called.
        print("\n----------------")
        print(tcols.OKGREEN + "Data loading complete:" + tcols.ENDC)
        print(f"File name: {self._fname}")
        print(f"Training data size: {self.ntrain_jets:,}")
        print(f"Validation data size: {self.nval_jets:,}")
        print(f"Test data size: {self.ntest_jets:,}")
        print(f"Number of constituents: {self.ncons:,}")
        print(f"Number of features: {self.nfeat:,}")
        print(f"Constituent seed: {self.seed:,}")
        print(f"Jet seed: {self.jet_seed:,}")
        print("----------------\n")