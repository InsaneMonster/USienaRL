#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import logging
import tensorflow
import enum


# Define layer types for tensorflow neural networks definition

class LayerType(enum.Enum):
    """
    Layer types for layer definition.

    Each layer type follows the naming scheme of the relative tensorflow layer types.
    Choosing a layer type actually will make the model use the relative tensorflow layer.
    """
    average_pooling_1D = 1
    average_pooling_2D = 2
    average_pooling_3D = 3
    batch_normalization = 4
    convolution_1D = 5
    convolution_2D = 6
    convolution_2D_transpose = 7
    convolution_3D = 8
    convolution_3D_transpose = 9
    dense = 10
    dropout = 11
    flatten = 12
    max_pooling_1D = 13
    max_pooling_2D = 14
    max_pooling_3D = 15
    separable_convolution_1D = 16
    separable_convolution_2D = 17


class Config:
    """
    Wrapper class used to define neural networks.

    Parts (like hidden layers) of a tensorflow neural network model can be defined by it.
    """

    def __init__(self):
        self._hidden_layers_config: [] = []

    def add_hidden_layer(self,
                         layer_type, layer_parameters: [], layer_name: str = None):
        """
        Add an hidden layer to the configuration using a tensorflow layer type as defined in the LayerType enum or as
        a custom defined string and a list of ordered layer parameters according to its specification.

        :param layer_type: the LayerType enum of the layer (following tensorflow names) or a custom string
        :param layer_name: the name of the layer, required especially when using reuse
        :param layer_parameters: the ordered list of parameters of the layer
        """
        # Add the layer type to this hidden layer definition list
        hidden_layer_config: [] = [layer_type, layer_name]
        # Add each parameter to this hidden layer definition list
        for layer_parameter in layer_parameters:
            hidden_layer_config.append(layer_parameter)
        # Add this hidden layer definition list to the hidden layers definition list
        self._hidden_layers_config.append(hidden_layer_config)

    def apply_hidden_layers(self,
                            input_layer, reuse=tensorflow.AUTO_REUSE):
        """
        Apply the hidden layer to the network on the given input layer.

        :param input_layer: the input layer of the network
        :param reuse: flag defining if a layer should be reuse of not, if at all possible (use it like in default tensorflow)
        :return: the output of the last hidden layer, usually connected to the neural network outputs.
        """
        # Initialize a list of hidden layers (to compute input/outputs)
        hidden_layers = []
        # Define all layers according to defined hidden layers type and parameters
        for index, hidden_layer_config in enumerate(self._hidden_layers_config):
            # Use layer type and parameters to define each layer
            layer_type = hidden_layer_config[0]
            layer_name = hidden_layer_config[1]
            layer_parameters: [] = hidden_layer_config[2:]
            # Define the input for the current layer
            hidden_layer_input = input_layer
            if isinstance(layer_type, str):
                # Custom hidden layer of name layer type
                hidden_layer = self._define_custom_hidden_layers(layer_type, hidden_layer_input, layer_parameters, layer_name, reuse)
            else:
                if index > 0:
                    hidden_layer_input = hidden_layers[index - 1]
                # Define the current layer depending on the specified type and parameters
                if layer_type == LayerType.average_pooling_1D:
                    hidden_layer = tensorflow.layers.average_pooling1d(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.average_pooling_2D:
                    hidden_layer = tensorflow.layers.average_pooling2d(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.average_pooling_3D:
                    hidden_layer = tensorflow.layers.average_pooling3d(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.batch_normalization:
                    hidden_layer = tensorflow.layers.batch_normalization(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.convolution_1D:
                    hidden_layer = tensorflow.layers.conv1d(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.convolution_2D:
                    hidden_layer = tensorflow.layers.conv2d(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.convolution_2D_transpose:
                    hidden_layer = tensorflow.layers.conv2d_transpose(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.convolution_3D:
                    hidden_layer = tensorflow.layers.conv3d(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.convolution_3D_transpose:
                    hidden_layer = tensorflow.layers.conv3d_transpose(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.separable_convolution_1D:
                    hidden_layer = tensorflow.layers.separable_conv1d(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.separable_convolution_2D:
                    hidden_layer = tensorflow.layers.separable_conv2d(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.dense:
                    hidden_layer = tensorflow.layers.dense(hidden_layer_input, *layer_parameters, name=layer_name, reuse=reuse)
                elif layer_type == LayerType.dropout:
                    hidden_layer = tensorflow.layers.dropout(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.flatten:
                    hidden_layer = tensorflow.layers.flatten(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.max_pooling_1D:
                    hidden_layer = tensorflow.layers.max_pooling1d(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.max_pooling_2D:
                    hidden_layer = tensorflow.layers.max_pooling2d(hidden_layer_input, *layer_parameters, name=layer_name)
                elif layer_type == LayerType.max_pooling_3D:
                    hidden_layer = tensorflow.layers.max_pooling3d(hidden_layer_input, *layer_parameters, name=layer_name)
                # An error occurred, break
                else:
                    return
            # Add the last defined layer in the layers list
            hidden_layers.append(hidden_layer)
        # Return the last layer (to compute the output)
        return hidden_layers[-1]

    def _define_custom_hidden_layers(self, layer_type_name: str, layer_input, layer_parameters: [], layer_name: str = None, reuse=None):
        """
        Define custom hidden layers indexed by the given layer name string. These layers take as input the given
        layer input, usually a tensor, and as parameters the given ordered list of parameters.

        :param layer_type_name: the string name of the layers
        :param layer_input: the input of the hidden layer (usually when applied the output of the layer before)
        :param layer_parameters: the ordered list of parameters of the layer (the same of add_custom_hidden_layer)
        :param layer_name: the name of the layer, required especially when using reuse (the same of add_custom_hidden_layer)
        :param reuse: flag defining if a layer should be reuse of not, if at all possible
        :return the output of the custom hidden layer (usually when applied what is fed to the following layer)
        """
        raise NotImplementedError()

