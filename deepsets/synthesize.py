import sys
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


# Patricks models --> Needs to be rewritten for HLS4ML
class PermutationEquivariantMax(KL.Layer):
    """Permutation equivariant neural network layer with max operation."""

    def __init__(self, dim):
        super(PermutationEquivariantMax, self).__init__()
        self.gamma = KL.Dense(dim)
        self.lambd = KL.Dense(dim, use_bias=False)

    def call(self, inputs: np.ndarray, **kwargs):
        x_maximum = tf.reduce_max(inputs, axis=1, keepdims=True)
        x_maximum = self.lambd(x_maximum)
        x = self.gamma(inputs)
        x = x - x_maximum

        return x

class PermutationEquivariantMean(KL.Layer):
    """Permutation equivariant neural network layer with mean operation."""

    def __init__(self, dim):
        super(PermutationEquivariantMean, self).__init__()
        self.gamma = KL.Dense(dim)
        self.lambd = KL.Dense(dim, use_bias=False)

    def call(self, inputs: np.ndarray, **kwargs):
        x_mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x_mean = self.lambd(x_mean)
        x = self.gamma(inputs)
        x = x - x_mean

        return x
    
class DeepSets_Equiv(keras.Model):
    """Deep sets permutation equivariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        nnodes_phi: Number of nodes in the layers of the phi neural network.
        nnodes_rho: Number of nodes in the layers of the rho neural network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(self, nnodes_phi: int = 32, nnodes_rho: int = 16, activ: str = "elu"):
        super(DeepSets_Equiv, self).__init__(name="DeepSets_Equiv")
        self.nclasses = 5

        self.phi = keras.Sequential(
            [
                PermutationEquivariantMean(nnodes_phi),
                KL.Activation(activ),
                PermutationEquivariantMean(nnodes_phi),
                KL.Activation(activ),
                PermutationEquivariantMean(nnodes_phi),
                KL.Activation(activ),
            ]
        )

        self.rho = keras.Sequential(
            [KL.Dense(nnodes_rho, activation=activ), KL.Dense(self.nclasses)]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output
    
class DeepSets_Inv(keras.Model):
    """Deep sets permutation invariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        nnodes_phi: Number of nodes in the layers of the phi neural network.
        nnodes_rho: Number of nodes in the layers of the rho neural network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(self, nnodes_phi: int = 32, nnodes_rho: int = 16, activ: str = "elu"):
        super(DeepSets_Inv, self).__init__(name="DeepSets_Inv")
        self.nclasses = 5

        self.phi = keras.Sequential(
            [
                KL.Dense(nnodes_phi),
                KL.Activation(activ),
                KL.Dense(nnodes_phi),
                KL.Activation(activ),
                KL.Dense(nnodes_phi),
                KL.Activation(activ),
            ]
        )

        self.rho = keras.Sequential(
            [KL.Dense(nnodes_rho, activation=activ), KL.Dense(self.nclasses)]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output
        
def get_deepsets_invariant_hls4ml(input_shape=(16, 16)):
    deepsets_input = keras.Input(shape=input_shape, name="input_layer")

    x_phi = KL.Dense(32, name="phi1")(deepsets_input)
    x_phi = KL.Activation('elu')(x_phi)

    x_phi = KL.Dense(32, name="phi2")(x_phi)
    x_phi = KL.Activation('elu')(x_phi)

    x_phi = KL.Dense(32, name="phi3")(x_phi)
    phi_output = KL.Activation('elu')(x_phi)

    # Invariant operation
    sum_output = KL.GlobalMaxPooling1D(keepdims=True, name="gp")(phi_output) 

    x_rho = KL.Dense(16, name="rho")(sum_output)
    x_rho = KL.Activation('elu')(x_rho)
    deepsets_output = KL.Dense(5)(x_rho)
    deepsets_network = keras.Model(deepsets_input, deepsets_output, name="deepsets_invariant")

    return deepsets_network

def get_deepsets_equivariant_hls4ml(input_shape=(16, 16)):

    deepsets_input = keras.Input(shape=input_shape, name="input_layer")

    # Permutation Equivariant Layer
    x_max = KL.GlobalMaxPooling1D(keepdims=True)(deepsets_input) 
    x_lambd = KL.Dense(32, use_bias=False, name="lambda")(x_max)
    x_gamma = KL.Dense(32)(deepsets_input)
    x = KL.Subtract()([x_gamma, x_lambd])
    x = KL.Activation('elu')(x)

    # # Permutation Equivariant Layer
    # x_max = KL.GlobalMaxPooling1D(keepdims=True)(x)
    # x_lambd = KL.Dense(32, use_bias=False)(x_max)
    # x_gamma = KL.Dense(32)(x)
    # x = KL.Subtract()([x_gamma, x_lambd])
    # x = KL.Activation('elu')(x)
    #
    # # Permutation Equivariant Layer
    # x_max = KL.GlobalMaxPooling1D(keepdims=True)(x)
    # x_lambd = KL.Dense(32, use_bias=False)(x_max)
    # x_gamma = KL.Dense(32)(x)
    # x = KL.Subtract()([x_gamma, x_lambd])
    # x = KL.Activation('elu')(x)
 
    x_max = KL.GlobalMaxPooling1D()(x)
    x = KL.Dense(16)(x)
    x = KL.Activation('elu')(x)
    deepsets_output = KL.Dense(5)(x)
    deepsets_network = keras.Model(deepsets_input, deepsets_output, name="deepsets_equivariant")
    return deepsets_network

if __name__ == "__main__":

    doEquivariant = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nconst = 16
    nfeats = 16
    input = np.random.rand(1,nconst,nfeats)
    input_shape = input.shape[1:]

    # model = get_deepsets_equivariant_hls4ml(input_shape=input_shape)
    if doEquivariant:
        model = get_deepsets_equivariant_hls4ml(input_shape=input_shape)
        outname = '/data/hlssynt-users/thaarres/'
        # model.load_weights("model_weights_equivariant.h5") #can't be bothered layer matching these for hlsm4l model, skip for now
    else:
        model = get_deepsets_invariant_hls4ml(input_shape=input_shape)
        outname = '/data/hlssynt-users/thaarres/'
        # model.load_weights("model_weights_invariant.h5") #can't be bothered layer matching these for hlsm4l model, skip for now

    # ------------ Patricks model (not compatible with hls4ml! ------------
    # model = DeepSets_Inv()
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.build((None, nconst, nfeats))
    
    model.summary()
    model.predict(input)

    # # Check some layers and intermediate outputs
    #from keras import backend as K
    # layer = model.layers[11]
    # get_layer_output = K.function([model.layers[0].input],[layer.output])
    # layer_output = get_layer_output([input])[0]
    # print(f"For {layer.name} :\n {layer_output} \n {layer.shape}")


    # hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    if doEquivariant:
      config['LayerName']['lambda']['ParallelizationFactor']= 16
      
    else:
      config['LayerName']['phi1']['ParallelizationFactor']= 16
      config['LayerName']['phi2']['ParallelizationFactor']= 16
      config['LayerName']['phi3']['ParallelizationFactor']= 16    
    # config['LayerName']['dense_1']['ReuseFactor']= 2
    print("-----------------------------------")
    print("Configuration")
    # plotting.print_dict(config)
    print("-----------------------------------")
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config,output_dir=outname)
    hls_model.compile()
    # hls_model.write()
    hls_model.build()

    # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

    #Other: One dense is being rewritten as pointwise conv1D giving 
    #ValueError: 
    # axes don't match array
    # Solved in 
    #https://github.com/fastmachinelearning/hls4ml/pull/716 and 
    #https://github.com/fastmachinelearning/hls4ml/pull/715
