import re
from typing import Dict, List

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.pruning.shrinkage_manager import ShrinkageManager
from src.pruning.utils import copy_layer, get_layer_mapping, get_name_of_layer_from_tensor


is_decoder_re = re.compile('^decoder_.*')


DECODER_AFFECTED = {
    'decoder_stage0a_conv': 'block_13_expand',
    'decoder_stage1a_conv': 'block_6_expand',
    'decoder_stage2a_conv': 'block_3_expand',
    'decoder_stage3a_conv': 'block_1_expand',
}


DECODER_PRUNING_LAYERS = [
    'decoder_stage0a_conv',
    'decoder_stage0b_conv',
    'decoder_stage1a_conv',
    'decoder_stage1b_conv',
    'decoder_stage2a_conv',
    'decoder_stage2b_conv',
    'decoder_stage3a_conv',
    'decoder_stage3b_conv',
    'decoder_stage4a_conv',
    'decoder_stage4b_conv',
]

ALLOWED_DECODER_PRUNING_LAYERS = [
    'decoder_stage0a_conv',
    'decoder_stage1a_conv',
    'decoder_stage2a_conv',
    'decoder_stage3a_conv',
    'decoder_stage4a_conv'
]


def prune_expand(
    model,
    expands_dict: Dict[str, List[int]],
):
    """
    Parameters
    ----------
    model: unet model
    expands_dict: expands layer -> indices of filters to remove, e.g 'block_5_expand': [1, 15]

    Returns
    -------
        model with pruned filters
    """

    mapping = get_layer_mapping(model)

    model_input = Input(
        shape=model.layers[0].input.shape.as_list()[1:],
        name=model.layers[0].name,
        dtype=model.layers[0].dtype
    )

    flow = None

    layers_dict = {
        model_input.name: model_input,
    }
    layers_with_updated_weights = {}

    for index, layer in enumerate(model.layers[1:]):
        if layer.name in layers_dict:
            continue

        n_inputs = 1
        if isinstance(layer.input, list):
            n_inputs = len(layer.input)

        if n_inputs == 1:  # just one input from previous layer
            if layer.name in expands_dict:
                flow, block_dict = ShrinkageManager.shrink_block(
                    model=model,
                    conv2d=layer,
                    indices_to_remove=expands_dict[layer.name],
                    flow=flow,
                    mapping=mapping,
                )
                layers_dict.update(block_dict)
                layers_with_updated_weights.update(block_dict)
            elif (
                    layer.name in DECODER_AFFECTED and
                    DECODER_AFFECTED[layer.name] in layers_with_updated_weights
            ):
                layer_ = ShrinkageManager.shrink_decoder_layer(
                    layer,
                    expands_dict[DECODER_AFFECTED[layer.name]],
                    model.get_layer(DECODER_AFFECTED[layer.name]),
                    from_zero=False,
                )
                flow = layer_(flow if index else model_input)
                layers_dict[layer_.name] = layer_
                layers_with_updated_weights[layer_.name] = layer_
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

    thin_model = Model(model_input, flow)
    for layer in thin_model.layers:
        if layer.name in layers_with_updated_weights:
            continue
        layer.set_weights(
            model.get_layer(layer.name).get_weights()
        )

    return thin_model


def prune_conv1(model, indices_to_remove: List[int]):
    #assert len(indices_to_remove)

    mapping = get_layer_mapping(model)

    model_input = Input(
        shape=model.layers[0].input.shape.as_list()[1:],
        name=model.layers[0].name,
        dtype=model.layers[0].dtype
    )

    flow = None

    layers_dict = {
        model_input.name: model_input,
    }

    layers_with_updated_weights = {}

    for index, layer in enumerate(model.layers[1:]):
        if layer.name in layers_dict:
            continue

        n_inputs = 1
        if isinstance(layer.input, list):
            n_inputs = len(layer.input)

        if n_inputs == 1:  # just one input from previous layer
            if layer.name == 'Conv_1':
                flow, conv1_after_dict = ShrinkageManager.shrink_conv1(
                    model=model,
                    conv2d=layer,
                    indices_to_remove=indices_to_remove,
                    flow=flow,
                    mapping=mapping,
                )
                layers_dict.update(conv1_after_dict)
                layers_with_updated_weights.update(conv1_after_dict)
            elif layer.name == 'decoder_stage0a_conv':
                layer_ = ShrinkageManager.shrink_decoder_layer(
                    layer,
                    indices_to_remove,
                    conv2d=model.get_layer('Conv_1'),
                    from_zero=False,
                )
                flow = layer_(flow)
                layers_dict[layer_.name] = layer_
                layers_with_updated_weights[layer_.name] = layer_
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

    thin_model = Model(model_input, flow)
    for layer in thin_model.layers:
        if layer.name in layers_with_updated_weights:
            continue
        layer.set_weights(
            model.get_layer(layer.name).get_weights()
        )

    return thin_model


def prune_decoder(model, decoder_dict: Dict[str, List[int]]):
    """
    Parameters
    ----------
    model
    decoder_dict

    Returns
    -------
    """
    for layer_name in decoder_dict:
        assert layer_name in ALLOWED_DECODER_PRUNING_LAYERS

    mapping = get_layer_mapping(model)

    model_input = Input(
        shape=model.layers[0].input.shape.as_list()[1:],
        name=model.layers[0].name,
        dtype=model.layers[0].dtype
    )

    flow = None

    layers_dict = {
        model_input.name: model_input,
    }
    layers_with_updated_weights = {}

    for index, layer in enumerate(model.layers[1:]):
        if layer.name in layers_dict:
            continue

        n_inputs = 1
        if isinstance(layer.input, list):
            n_inputs = len(layer.input)

        if n_inputs == 1:  # just one input from previous layer
            if layer.name in decoder_dict:
                flow, block_dict = ShrinkageManager.shrink_decoder(
                    model=model,
                    conv2d=layer,
                    indices_to_remove=decoder_dict[layer.name],
                    flow=flow,
                    mapping=mapping,
                )
                layers_dict.update(block_dict)
                layers_with_updated_weights.update(block_dict)
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

    thin_model = Model(model_input, flow)
    for layer in thin_model.layers:
        if layer.name in layers_with_updated_weights:
            continue
        layer.set_weights(
                model.get_layer(layer.name).get_weights()
        )

    return thin_model
