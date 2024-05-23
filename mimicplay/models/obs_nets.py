"""
Contains torch Modules that help deal with inputs consisting of multiple
modalities. This is extremely common when networks must deal with one or 
more observation dictionaries, where each input dictionary can have
observation keys of a certain modality and shape.

As an example, an observation could consist of a flat "robot0_eef_pos" observation key,
and a 3-channel RGB "agentview_image" observation key.
包含了一些用于处理由多个模态组成的输入的torch模块。当网络需要处理一个或多个观测字典时，这是非常常见的情况，其中每个输入字典可以具有特定模态和形状的观测键。

例如，一个观测可以由一个平的"robot0_eef_pos"观测键和一个3通道RGB的"agentview_image"观测键组成。
"""
import sys
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import Module, Sequential, MLP, RNN_Base, ResNet18Conv, SpatialSoftmax, \
    FeatureAggregator
from robomimic.models.obs_core import VisualCore, Randomizer
from robomimic.models.transformers import PositionalEncoding, GPT_Backbone


def obs_encoder_factory(
        obs_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.
    从配置中的kwargs创建一个@ObservationEncoder的实用函数。

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.
        一个将观察键映射到观察期望形状的字典。

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.
        每个观察网络后应用的非线性，默认为ReLU。传递None表示不应用激活函数。

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
            Should be of form:
        如果为None，则应用默认的encoder_kwargs。否则，应该是一个嵌套字典，包含编码器网络的相关模态信息。
            应该具有以下格式：

            obs_modality1: dict
                feature_dimension: int
                core_class: str
                core_kwargs: dict
                    ...
                    ...
                obs_randomizer_class: str
                obs_randomizer_kwargs: dict
                    ...
                    ...
            obs_modality2: dict
                ...
    """
    enc = ObservationEncoder(feature_activation=feature_activation)
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality]) if encoder_kwargs is None else \
            deepcopy(encoder_kwargs[obs_modality])

        for obs_module, cls_mapping in zip(("core", "obs_randomizer"),
                                      (ObsUtils.OBS_ENCODER_CORES, ObsUtils.OBS_RANDOMIZERS)):
            # Sanity check for kwargs in case they don't exist / are None
            if enc_kwargs.get(f"{obs_module}_kwargs", None) is None:
                enc_kwargs[f"{obs_module}_kwargs"] = {}
            # Add in input shape info
            enc_kwargs[f"{obs_module}_kwargs"]["input_shape"] = obs_shape
            # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
            if enc_kwargs[f"{obs_module}_class"] is not None:
                enc_kwargs[f"{obs_module}_kwargs"] = extract_class_init_kwargs_from_dict(
                    cls=cls_mapping[enc_kwargs[f"{obs_module}_class"]],
                    dic=enc_kwargs[f"{obs_module}_kwargs"],
                    copy=False,
                )

        # Add in input shape info
        randomizer = None if enc_kwargs["obs_randomizer_class"] is None else \
            ObsUtils.OBS_RANDOMIZERS[enc_kwargs["obs_randomizer_class"]](**enc_kwargs["obs_randomizer_kwargs"])

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizer=randomizer,
        )

    enc.make()
    return enc


class ObservationEncoder(Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks.
    一个模块，通过观察键处理输入，然后将处理后的观察键连接在一起。每个键都是用一个编码器头网络处理的。
    调用@register_obs_key注册观察键到编码器，最后调用@make创建编码器网络。
    """
    def __init__(self, feature_activation=nn.ReLU):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.
            每个观察网络后应用的非线性，默认为ReLU。传递None表示不应用激活函数。
        """
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self._locked = False

    def register_obs_key(
        self,
        name,
        shape,
        net_class=None,
        net_kwargs=None,
        net=None,
        randomizer=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.
        注册这个编码器应该负责的观察键。

        Args:
            name (str): modality name
            模态名
            shape (int tuple): shape of modality
            模态的形状
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            应该用来处理这个观察键的类的名称，这个类应该在base_nets.py中。传递None来直接展平并连接观察键。
            net_kwargs (dict): arguments to pass to @net_class
            传递给@net_class的参数
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            如果提供，使用这个模块来处理观察键，而不是创建一个不同的网络
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            如果提供，使用这个模块来增强传给编码器的观察键，可能也增强处理后的输出
            share_net_from (str): if provided, use the same instance of @net_class
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
                如果提供，使用@net_class的相同实例作为另一个观察键的实例。这个观察键必须已经存在于这个编码器中。
                警告：请注意，这不会共享观察键随机化器
        """
        assert not self._locked, "ObservationEncoder: @register_obs_key called after @make"
        assert name not in self.obs_shapes, "ObservationEncoder: modality {} already exists".format(name)

        if net is not None:
            assert isinstance(net, Module), "ObservationEncoder: @net must be instance of Module class"
            assert (net_class is None) and (net_kwargs is None) and (share_net_from is None), \
                "ObservationEncoder: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        if randomizer is not None:
            assert isinstance(randomizer, Randomizer)
            if net_kwargs is not None:
                # update input shape to visual core
                net_kwargs["input_shape"] = randomizer.output_shape_in(shape)

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = randomizer
        self.obs_share_mods[name] = share_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        创建编码器网络并锁定编码器，以防止添加更多的模态。
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        使用注册的模态创建所有编码器需要的网络和层。
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                # 创建处理该模态的网络
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](**self.obs_nets_kwargs[k])
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                # 确保网络与另一个模态共享
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.
        根据self.obs_shapes中的顺序，对每个模态进行处理。
        对于每个模态，首先使用随机器（如果存在），然后使用编码器网络（如果存在）进行处理，
        再次使用随机器（如果存在）进行处理，最后将处理后的模态展平并与其他处理后的模态进行拼接。

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.
            将模态映射到与self.obs_shapes相符的torch.Tensor批的字典。
                obs_dict中必须包含self.obs_shapes中的所有模态，但也可以包含其他模态。

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
            形状为[B, D]的扁平特征
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        # 确保编码器处理的所有模态都存在
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        # 按self.obs_shapes给定的顺序处理模态
        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            # 可能使用随机器处理编码器的输入
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_in(x)
            # maybe process with obs net
            # 可能使用obs网络进行处理
            if self.obs_nets[k] is not None:
                x = self.obs_nets[k](x)
                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            # 可能使用随机器处理编码器的输出
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_out(x)
            # flatten to [B, D]
            # 展平为[B, D]
            x = TensorUtils.flatten(x, begin_axis=1)
            feats.append(x)

        # concatenate all features together
        # 将所有特征拼接在一起
        return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        计算编码器的输出形状。
        """
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_out(feat_shape)
            feat_dim += int(np.prod(feat_shape))
        return [feat_dim]

    def __repr__(self):
        """
        Pretty print the encoder.
        美观打印编码器。
        """
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("randomizer={}\n".format(self.obs_randomizers[k]), indent)
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent("sharing_from={}\n".format(self.obs_share_mods[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    一个模块，可以通过模态生成观测输出。假设输入是平的（通常是从某个隐藏层输出的）。
    每个观测输出都是通过一个线性层从这些平的输入生成的。子类化这个模块以实现更复杂的方案来生成每个模态。
    """
    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.
            映射观测键到期望形状的字典。这用于从输入特征生成输出模态。

            input_feat_dim (int): flat input dimension size
            平的输入维度大小
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.obs_shapes[k]) for k in self.obs_shapes }

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    这个类允许网络将多个观测字典编码为单个扁平的连接向量表示。
    它通过为每个观测字典（观测组）分配一个@ObservationEncoder对象来实现这一点。

    该类接受一个字典的字典，@observation_group_shapes。
    每个键对应一个观测组（例如'obs'，'subgoal'，'goal'），每个OrderedDict应该是模态和
    期望的输入形状之间的映射（例如{'image'：（3，120，160）}）。
    """
    def __init__(
        self,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(ObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all([isinstance(observation_group_shapes[k], OrderedDict) for k in observation_group_shapes])

        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(inputs), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(
                self.nets[obs_group].forward(inputs[obs_group])
            )

        return torch.cat(outputs, dim=-1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.observation_group_shapes:
            msg += '\n'
            indent = ' ' * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_MLP(Module):
    """
    Extension to MLP to accept multiple observation dictionaries as input and
    to output dictionaries of tensors. Inputs are specified as a dictionary of 
    observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating 
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    扩展到MLP以接受多个观测字典作为输入，并输出张量字典。输入被指定为观测字典的字典，每个键对应一个观测组。

    该模块利用@ObservationGroupEncoder处理多个输入字典和@ObservationDecoder生成张量字典。
    默认行为是使用学习的CNN处理视觉输入并将平的编码与其他平的输入连接起来。
    生成输出的默认行为是使用线性层分支分别生成每个模态（包括视觉输出）。
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        layer_dims,
        layer_func=nn.Linear, 
        activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            layer_func: mapping per MLP layer - defaults to Linear

            activation: non-linearity per MLP layer - defaults to ReLU

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(MIMO_MLP, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        mlp_input_dim = self.nets["encoder"].output_shape()[0]

        # intermediate MLP layers
        self.nets["mlp"] = MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation, # make sure non-linearity is applied before decoder
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=layer_dims[-1],
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, return_latent=False, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes.

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes
        """
        enc_outputs = self.nets["encoder"](**inputs)
        mlp_out = self.nets["mlp"](enc_outputs)
        if return_latent:
            return self.nets["decoder"](mlp_out), enc_outputs.detach(), mlp_out.detach()
        return self.nets["decoder"](mlp_out)

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg
