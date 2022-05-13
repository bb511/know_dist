# Handles the data importing and small preprocessing for the interaction network.

import os
import numpy as np

from .terminal_colors import tcols

class Data:
    def __init__(
        self,
        data_folder: str,
        hyperparams: dict,
        norm_name: str,
        train_events: int,
        test_events: int,
        seed: int = None,
    ):

        self.data_folder = data_folder
        self.norm_name = norm_name
        self.hyperparams = hyperparams

        self.train_events = train_events
        self.test_events = test_events

        self.seed = seed

        self.tr_data, self.tr_target = self._load_data("train")
        self.te_data, self.te_target = self._load_data("test")

        self.ntrain_jets = self.tr_data.shape[0]
        self.ntest_jets = self.te_data.shape[0]
        self.ncons = self.tr_data.shape[1]
        self.nfeat = self.tr_data.shape[2]

        self._success_message()

    def _load_data(self, data_type: str) -> tuple([np.ndarray, np.ndarray]):
        """Load data from the data files generated by the pre-processing scripts.

        Args:
            data_type: The type of data that you want to load, either train or test.

        Returns:
            Two numpy arrays with loaded data and the corresponding target.
        """
        datafile_name = self._get_data_filename(data_type)
        datafile_path = os.path.join(self.data_folder, datafile_name)

        targetfile_name = self._get_target_filename(data_type)
        targetfile_path = os.path.join(self.data_folder, targetfile_name)

        x = np.load(datafile_path)
        y = np.load(targetfile_path)

        nevents = self._get_nevents(data_type)
        x, y = self._trim_data(x, y, nevents, self.seed)

        return x, y

    def _get_nevents(self, data_type: str) -> int:
        if data_type == "train":
            return self.train_events
        elif data_type == "test":
            return self.test_events
        else:
            raise TypeError("Nevents error: type of data provided does not exist! "
                            "Choose either 'train' or 'test'.")

    def _get_data_filename(self, data_type) -> str:
        return (
            "x_jet_images" + "_c" + self.hyperparams["nconstituents"] +
            "_pt" + self.hyperparams["pt_min"] + "_" +
            self.norm_name + "_" + data_type + ".npy"
        )

    def _get_target_filename(self, data_type) -> str:
        return (
            "y_jet_images" + "_c" + self.hyperparams["nconstituents"] +
            "_pt" + self.hyperparams["pt_min"] + "_" +
            self.norm_name + "_" + data_type + ".npy"
        )

    @staticmethod
    def _segregate_data(x_data: np.array, y_data: np.array)\
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

    def _trim_data(self, x: np.ndarray, y: np.ndarray, maxdata: int, seed: int) \
        -> tuple([np.ndarray, np.ndarray]):
        """Cut the imported data and target and form new data sets with
        equal numbers of events per each class.

        Args:
            @x: Numpy array containing the data.
            @y: Numpy array containing the corresponding target (one-hot).

        Returns:
            Two numpy arrays, one with data and one with target,
            containing an equal number of events per each class.
        """
        if maxdata < 0:
            return x, y

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

    def _success_message(self):
        # Display success message for loading data when called.
        print("\n----------------")
        print(tcols.OKGREEN + "Data loading complete:" + tcols.ENDC)
        print(f"Training data size: {self.tr_data.shape[0]:.2e}")
        print(f"Test data size: {self.te_data.shape[0]:.2e}")
        print("----------------\n")

    def shuffle_constituents(self, data_type: str) -> np.ndarray:
        """Shuffle the jet constituents for a particular data type. Each jet is
        shuffled with a different seed.
        """
        print("Shuffling constituents...")
        rng = np.random.default_rng(self.seed)
        if data_type == "train":
            jet_seeds = rng.integers(low=0, high=100, size=self.ntrain_jets)
            data = self.tr_data
        elif data_type == "test":
            jet_seeds = rng.integers(low=0, high=100, size=self.ntest_jets)
            data = self.te_data
        else:
            raise TypeError("Shuffling error: type of data provided does not exist! "
                            "Choose either 'train' or 'test'.")

        for jet_idx, seed in enumerate(jet_seeds):
            shuffling = np.random.RandomState(seed=seed).permutation(self.ncons)
            data[jet_idx, :, :] = data[jet_idx, shuffling, :]

        print(tcols.OKGREEN + "Shuffle done! \U0001F0CF" + tcols.ENDC)