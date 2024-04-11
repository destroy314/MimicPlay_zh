"""
This file contains base classes that other algorithm classes subclass.
Each algorithm file also implements a algorithm factory function that
takes in an algorithm config (`config.algo`) and returns the particular
Algo subclass that should be instantiated, along with any extra kwargs.
These factory functions are registered into a global dictionary with the
@register_algo_factory_func function decorator. This makes it easy for
@algo_factory to instantiate the correct `Algo` subclass.
此文件包含其他算法类子类化的基类。
每个算法文件还实现了一个算法工厂函数，该函数接受算法配置（`config.algo`）并返回应该实例化的特定
Algo子类，以及任何额外的kwargs。这些工厂函数使用@register_algo_factory_func函数装饰器注册到全局字典中，
这使得@algo_factory可以实例化正确的`Algo`子类。
NOTE 复制自robomimic/robomimic/algo/algo.py，因此实际被继承的不是这里定义的类。（只有RolloutPolicy在别处被实例化）
"""
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch.nn as nn

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils


# mapping from algo name to factory functions that map algo configs to algo class names
# 算法名称到工厂函数的映射，这些工厂函数将算法配置映射到算法类名称
REGISTERED_ALGO_FACTORY_FUNCS = OrderedDict()


def register_algo_factory_func(algo_name):
    """
    Function decorator to register algo factory functions that map algo configs to algo class names.
    Each algorithm implements such a function, and decorates it with this decorator.
    函数装饰器，用于注册将算法配置映射到算法类名称的算法工厂函数。
    每个算法都实现了这样一个函数，并使用此装饰器进行装饰。

    Args:
        algo_name (str): the algorithm name to register the algorithm under
        要将算法注册到的算法名称
    """
    def decorator(factory_func):
        REGISTERED_ALGO_FACTORY_FUNCS[algo_name] = factory_func
    return decorator


def algo_name_to_factory_func(algo_name):
    """
    Uses registry to retrieve algo factory function from algo name.
    使用注册表从算法名称检索算法工厂函数。

    Args:
        algo_name (str): the algorithm name
        算法名称
    """
    return REGISTERED_ALGO_FACTORY_FUNCS[algo_name]


def algo_factory(algo_name, config, obs_key_shapes, ac_dim, device):
    """
    Factory function for creating algorithms based on the algorithm name and config.
    根据算法名称和配置创建算法的工厂函数。

    Args:
        algo_name (str): the algorithm name
        算法名称

        config (BaseConfig instance): config object
        配置对象

        obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes
        将观察键映射到形状的字典

        ac_dim (int): dimension of action space
        动作空间的维度

        device (torch.Device): where the algo should live (i.e. cpu, gpu)
        算法应该存在的位置（即cpu，gpu）
    """

    # @algo_name is included as an arg to be explicit, but make sure it matches the config
    # 确保algo_name与config匹配
    assert algo_name == config.algo_name

    # use algo factory func to get algo class and kwargs from algo config
    # 使用算法工厂函数从算法配置获取算法类和kwargs
    factory_func = algo_name_to_factory_func(algo_name)
    algo_cls, algo_kwargs = factory_func(config.algo)

    # create algo instance
    # 创建算法实例
    return algo_cls(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
        **algo_kwargs
    )


class Algo(object):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API to be used by training functions such as @run_epoch in
    utils/train_utils.py.
    所有其他算法子类的基本算法类。定义了几个应该由子类重写的函数，
    以提供用于训练函数的标准API，如utils/train_utils.py中的@run_epoch
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config
            对应于配置的算法部分的Config实例

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config
            对应于配置的观察部分的Config实例

            global_config (Config object): global training config
            全局训练配置

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes
            将观察键映射到形状的字典

            ac_dim (int): dimension of action space
            动作空间的维度

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
            算法应该存在的位置（即cpu，gpu）
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.nets = nn.ModuleDict()
        self._create_shapes(obs_config.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _create_shapes(self, obs_keys, obs_key_shapes):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. Each dictionary
        maps observation key to shape.
        创建obs_shapes、goal_shapes和subgoal_shapes字典，以便该算法对象可以轻松跟踪观察键形状。
        每个字典将观察键映射到形状。

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            此训练运行所需的观察键的字典（通常由obs配置指定），例如，{"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
            obs_key_shapes (dict): 观察键形状的字典，例如，{"rgb": [3, 224, 224]}
        """
        # determine shapes
        # 确定形状
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()

        # We check across all modality groups (obs, goal, subgoal), and see if the inputted observation key exists
        # across all modalitie specified in the config. If so, we store its corresponding shape internally
        # 我们检查所有模态组（obs，goal，subgoal），并查看输入的观察键是否存在于配置中指定的所有模态中。
        # 如果是这样，我们在内部存储其相应的形状
        for k in obs_key_shapes:
            if "obs" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.obs.values() for obs_key in modality]:
                self.obs_shapes[k] = obs_key_shapes[k]
            if "goal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.goal.values() for obs_key in modality]:
                self.goal_shapes[k] = obs_key_shapes[k]
            if "subgoal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.subgoal.values() for obs_key in modality]:
                self.subgoal_shapes[k] = obs_key_shapes[k]

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        @self.nets should be a ModuleDict.
        创建网络并将其放入@self.nets中。
        @self.nets应该是一个ModuleDict。
        """
        raise NotImplementedError

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        使用@self.optim_params创建优化器并将其放入@self.optimizers中。
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            # 仅为已创建的网络创建优化器 - @optim_params可能具有更多用于未使用网络的设置
            if k in self.nets:
                if isinstance(self.nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        TorchUtils.optimizer_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        TorchUtils.lr_scheduler_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i], optimizer=self.optimizers[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                else:
                    self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k])
                    self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k], optimizer=self.optimizers[k])

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        处理从数据加载器中采样的输入批次，以过滤出相关信息并准备批次进行训练。

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
            从数据加载器中采样的包含torch.Tensors的字典

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
            处理和过滤后的批次，将用于训练
        """
        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        在单个数据批次上进行训练。

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            从数据加载器中采样的包含torch.Tensors的字典，经过@process_batch_for_training过滤

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping
            epoch编号 - 某些执行分阶段训练和提前停止的Algo需要

            validate (bool): if True, don't perform any learning updates.
            如果为True，则不执行任何学习更新。

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
            可能与日志记录相关的输入、输出和损失的字典
        """
        assert validate or self.nets.training
        return OrderedDict()

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        处理来自@train_on_batch的info字典，以总结信息以传递给tensorboard进行日志记录。

        Args:
            info (dict): dictionary of info
            info的字典

        Returns:
            loss log (dict): name -> summary statistic
            名称->摘要统计信息
        """
        log = OrderedDict()

        # record current optimizer learning rates
        # 记录当前优化器的学习率
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        在每个epoch结束时调用。
        """

        # LR scheduling updates
        # 学习率调度更新
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()

    def set_eval(self):
        """
        Prepare networks for evaluation.
        准备网络进行评估。
        """
        self.nets.eval()

    def set_train(self):
        """
        Prepare networks for training.
        准备网络进行训练。
        """
        self.nets.train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        获取当前模型参数的字典。
        """
        return self.nets.state_dict()

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.
        从检查点加载模型。

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            由self.serialize()保存的字典，其中包含与@self.network_classes相同的键
        """
        self.nets.load_state_dict(model_dict)

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        美观打印算法和网络描述。
        """
        return "{} (\n".format(self.__class__.__name__) + \
               textwrap.indent(self.nets.__repr__(), '  ') + "\n)"

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        重置算法状态，以准备环境rollout。
        """
        pass


class PolicyAlgo(Algo):
    """
    Base class for all algorithms that can be used as policies.
    所有可以用作策略的算法的基类。
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        获取策略动作输出。

        Args:
            obs_dict (dict): current observation
            当前观察
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            action (torch.Tensor): action tensor
            动作张量
        """
        raise NotImplementedError


class ValueAlgo(Algo):
    """
    Base class for all algorithms that can learn a value function.
    所有可以学习值函数的算法的基类。
    """
    def get_state_value(self, obs_dict, goal_dict=None):
        """
        获取基于状态的输出值。
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            当前观察
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            value (torch.Tensor): value tensor
            值张量
        """
        raise NotImplementedError

    def get_state_action_value(self, obs_dict, actions, goal_dict=None):
        """
        Get state-action value outputs.
        获取基于状态-动作的输出值。

        Args:
            obs_dict (dict): current observation
            当前观察
            actions (torch.Tensor): action
            动作
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            value (torch.Tensor): value tensor
            值张量
        """
        raise NotImplementedError


class PlannerAlgo(Algo):
    """
    Base class for all algorithms that can be used for planning subgoals
    conditioned on current observations and potential goal observations.
    所有可以用于基于当前观察和潜在目标观察计划子目标的算法的基类。
    """
    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get predicted subgoal outputs.
        获取预测的子目标输出。

        Args:
            obs_dict (dict): current observation
            当前观察
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        raise NotImplementedError

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        For planners that rely on sampling subgoals.
        用于依赖于采样子目标的规划器。

        Args:
            obs_dict (dict): current observation
            当前观察
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        raise NotImplementedError


class HierarchicalAlgo(Algo):
    """
    Base class for all hierarchical algorithms that consist of (1) subgoal planning
    and (2) subgoal-conditioned policy learning.
    所有分层算法的基类，包括（1）子目标规划和（2）子目标为条件策略学习。
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        获取策略动作输出。

        Args:
            obs_dict (dict): current observation
            当前观察
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            action (torch.Tensor): action tensor
            动作张量
        """
        raise NotImplementedError

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get subgoal predictions from high-level subgoal planner.
        获取高级子目标规划器的子目标预测。

        Args:
            obs_dict (dict): current observation
            当前观察
            goal_dict (dict): (optional) goal
            （可选的）目标

        Returns:
            subgoal (dict): predicted subgoal
            预测的子目标
        """
        raise NotImplementedError

    @property
    def current_subgoal(self):
        """
        Get the current subgoal for conditioning the low-level policy
        获取作为低级策略条件的当前子目标。

        Returns:
            current subgoal (dict): predicted subgoal
            预测的子目标
        """
        raise NotImplementedError


class RolloutPolicy(object):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    包装@Algo对象，以便在rollout循环中轻松运行策略。
    """
    def __init__(self, policy, obs_normalization_stats=None):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts
            准备进行rollout的Algo实例

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
                可选地传递一个字典以进行观察归一化。
                这应该将观察键映射到形状为（1，...）的字典，其中...是观察的默认形状。
        """
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        准备策略以开始新的rollout。
        """
        self.policy.set_eval()
        self.policy.reset()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.
        准备来自环境的原始观察字典以供策略使用。

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
            来自环境的单个观察字典（没有批维度，每个键的值为np.array）
        """
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        if self.obs_normalization_stats is not None:
            # ensure obs_normalization_stats are torch Tensors on proper device
            # 确保obs_normalization_stats是适当设备上的torch Tensors
            obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.policy.device))
            # limit normalization to obs keys being used, in case environment includes extra keys
            # 限制并归一化到使用的obs键，以防环境包含额外的键
            ob = { k : ob[k] for k in self.policy.global_config.all_obs_keys }
            ob = ObsUtils.normalize_obs(ob, obs_normalization_stats=obs_normalization_stats)
        return ob

    def __repr__(self):
        """Pretty print network description美观打印网络描述"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.
        从环境的原始观察字典（和可能的目标字典）中产生动作。

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
            来自环境的单个观察字典（没有批维度，每个键的值为np.array）
            goal (dict): goal observation
            目标观察
        """
        ob = self._prepare_observation(ob)
        if goal is not None:
            goal = self._prepare_observation(goal)
        ac = self.policy.get_action(obs_dict=ob, goal_dict=goal)
        return TensorUtils.to_numpy(ac[0])
