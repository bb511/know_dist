import os
import numpy as np
import tensorflow as tf
import hls4ml

np.random.seed(12)
tf.random.set_seed(12)

from tensorflow import keras
import tensorflow.keras.layers as KL

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def get_deepsets_invariant_hls4ml(input_shape=(8, 3)):
    deepsets_input = keras.Input(shape=input_shape, name="input_layer")

    x_phi = KL.Dense(32, name="phi1")(deepsets_input)
    x_phi = KL.Activation('relu')(x_phi)

    x_phi = KL.Dense(32, name="phi2")(x_phi)
    x_phi = KL.Activation('relu')(x_phi)

    x_phi = KL.Dense(32, name="phi3")(x_phi)
    phi_output = KL.Activation('relu')(x_phi)

    # Invariant operation
    sum_output = KL.GlobalMaxPooling1D(name="gp")(phi_output) 

    x_rho = KL.Dense(16, name="rho")(sum_output)
    x_rho = KL.Activation('relu')(x_rho)
    deepsets_output = KL.Dense(5)(x_rho)
    deepsets_network = keras.Model(deepsets_input, deepsets_output, name="deepsets_invariant")

    return deepsets_network


if __name__ == "__main__":

    nconst = 8
    nfeats = 3
    input = np.random.rand(1,nconst,nfeats)
    input_shape = input.shape[1:]

    model = get_deepsets_invariant_hls4ml(input_shape=input_shape)
    outname = '/data/hlssynt-users/thaarres/ds_invariant'
    
    model.summary()
    # model.predict(input)

    # # # Check some layers and intermediate outputs
    # from keras import backend as K
    # layer = model.layers[11]
    # get_layer_output = K.function([model.layers[0].input],[layer.output])
    # layer_output = get_layer_output([input])[0]
    # print(f"For {layer.name} :\n {layer_output} \n {layer.shape}")


    # hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    # config['Model']['Strategy']= 'Resource'

    config['LayerName']['phi1']['ParallelizationFactor']= 8
    config['LayerName']['phi2']['ParallelizationFactor']= 8
    config['LayerName']['phi3']['ParallelizationFactor']= 8  

    config['LayerName']['phi1']['ReuseFactor']= 2
    config['LayerName']['phi2']['ReuseFactor']= 2
    config['LayerName']['phi3']['ReuseFactor']= 2

    # config['LayerName']['gp']['Strategy']= 'Resource'


    print(print_dict(config))
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=outname
    )
    hls_model.compile()
    # hls_model.write()
    hls_model.build()

    # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)    