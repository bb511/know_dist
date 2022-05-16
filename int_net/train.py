# Training of the model defined in the model.py file.

import os

from . import data

def main(args):
    seed = 123

    outdir = "./trained_intnets/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    jet_data = data.Data(args["data_folder"], args["data_hyperparams"], args["norm"],
                         args["train_events"], 0, seed=seed)
    jet_data = data.Data.shuffled(args["data_folder"], args["data_hyperparams"],
                                  args["norm"], args["train_events"], 0, seed=seed)






