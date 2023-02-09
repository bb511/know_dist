# Distiller hypermodel class within the keras tuner network.

import keras_tuner as kt


class DistillerHypermodel(kt.HyperModel):
    """Keras tuner hypermodel for the distiller class implemented in distiller.py."""

    def __init__(self, distiller, distiller_hyperparams):
        self.distiller = distiller
        self.distiller_hyperparams = distiller_hyperparams

    def build(self, hp):
        trial_distiller_hyperparams = self.distiller_hyperparams.copy()
        temperature = hp.Int(
            "temperature",
            min_value=self.distiller_hyperparams["temperature"][0],
            max_value=self.distiller_hyperparams["temperature"][1],
            step=self.distiller_hyperparams["temperature"][2],
        )
        alpha = hp.Float(
            "alpha",
            min_value=self.distiller_hyperparams["alpha"][0],
            max_value=self.distiller_hyperparams["alpha"][1],
            step=self.distiller_hyperparams["alpha"][2],
        )

        trial_distiller_hyperparams["alpha"] = alpha
        trial_distiller_hyperparams["temperature"] = temperature
        self.distiller.compile(**trial_distiller_hyperparams)

        return self.distiller
