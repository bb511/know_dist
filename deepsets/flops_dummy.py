# FLOPs dummy calculation

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL

import flops

def main():
    deepsets_input = KL.Input(shape=(8, 3), name="input_layer")

    # Phi network.
    x_phi = KL.Dense(32)(deepsets_input)
    x_phi = KL.Activation('relu')(x_phi)
    x_phi = KL.Dense(32)(x_phi)
    x_phi = KL.Activation('relu')(x_phi)
    x_phi = KL.Dense(32)(x_phi)
    phi_output = KL.Activation('relu')(x_phi)

    # Invariant operation
    # inv_operation_output = KL.GlobalMaxPooling1D()(phi_output)
    inv_operation_output = KL.GlobalAveragePooling1D()(phi_output)

    # Rho network.
    x_rho = KL.Dense(32)(inv_operation_output)
    x_rho = KL.Activation('relu')(x_rho)

    deepsets_output = KL.Dense(5)(x_rho)
    deepsets = keras.Model(deepsets_input, deepsets_output, name="deepsets_invariant")
    deepsets.compile(optimizer=tf.keras.optimizers.Adam(), loss='crossentropy')

    deepsets.save('../bin/trained_deepsets/test/deepsets_flops.h5')
    print(f"FLOPs using tfgraph: {get_flops_tfgraph()}")


def get_flops_tfgraph() -> int:
    """Computes the number of FLOPs using the tensorflow profiler."""
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(
                '../bin/trained_deepsets/test/deepsets_flops.h5', compile=False
            )

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd='op', options=opts
            )

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops


if __name__ == '__main__':
    main()

