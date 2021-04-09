import re

from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

from src.pruning.utils import copy_layer, get_name_of_layer_from_tensor

is_decoder_re = re.compile('^decoder_.*')


def apply_city_hack(model):
    """
    Apply changes to decoder and returns a copy.

    Parameters
    ----------
    model: Unet model in which some structural changes will be applied to decoder part.
    The passed 'model' will be unchanged (immutability).

    Returns
    -------
        Unet with changed decoder.
    """
    model_input = Input(
        shape=model.layers[0].input.shape.as_list()[1:],
        name=model.layers[0].name,
        dtype=model.layers[0].dtype
    )

    flow = None

    layers_dict = {
        model_input.name: model_input,
    }

    city_layer = Conv2D(256, 1, padding='same')

    for index, layer in enumerate(model.layers[1:]):
        n_inputs = 1
        if isinstance(layer.input, list):
            n_inputs = len(layer.input)

        if n_inputs == 1:  # just one input from previous layer
            if layer.name == 'decoder_stage0a_conv':
                flow = city_layer(flow)
                layer_ = copy_layer(layer)
                flow = layer_(flow if index else model_input)
                layers_dict[layer_.name] = layer_
            else:
                layer_ = copy_layer(layer)
                flow = layer_(flow if index else model_input)
                layers_dict[layer_.name] = layer_

        else:  # many inputs
            inputs = []
            for input_ in layer.input:
                from_layer_name = get_name_of_layer_from_tensor(input_)
                inputs.append(layers_dict[from_layer_name].output)

            layer_ = copy_layer(layer)
            flow = layer_(inputs)
            layers_dict[layer_.name] = layer_

    city_model = Model(model_input, flow)

    for layer in city_model.layers:
        if is_decoder_re.match(layer.name):
            continue

        if layer.name not in layers_dict:
            continue

        layer.set_weights(
            model.get_layer(layer.name).get_weights()
        )

    return city_model
