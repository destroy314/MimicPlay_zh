"""
Implementation of MimicPlay and PlayLMP baselines (formalized as BC-RNN (robomimic) and BC-trans)
实现MimicPlay和PlayLMP基线（作为BC-RNN（robomimic）和BC-transformer）
"""
from collections import OrderedDict

import copy
import h5py
import torch
import torch.nn as nn

import robomimic.models.base_nets as BaseNets
import mimicplay.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import mimicplay.utils.file_utils as FileUtils
from mimicplay.algo import register_algo_factory_func, PolicyAlgo
from mimicplay.algo.GPT import GPT_wrapper, GPT_wrapper_scratch
from robomimic.algo.bc import BC_Gaussian, BC_RNN

@register_algo_factory_func("mimicplay")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MimicPlay algo class to instantiate, along with additional algo kwargs.
    映射算法配置到MimicPlay算法类以及额外的算法kwargs来实例化。

    Args:
        algo_config (Config instance): algo config
        算法配置

    Returns:
        algo_class: subclass of Algo
        Algo的子类
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
        要传给算法的额外关键词参数的字典
    """

    if algo_config.highlevel.enabled:
        if algo_config.lowlevel.enabled:
            return Lowlevel_GPT_mimicplay, {}
        else:
            return Highlevel_GMM_pretrain, {}
    else:
        if algo_config.lowlevel.enabled:
            return Baseline_GPT_from_scratch, {}
        else:
            return BC_RNN_GMM, {}

class Highlevel_GMM_pretrain(BC_Gaussian):
    """
    MimicPlay highlevel latent planner, trained to generate 3D trajectory based on observation and goal image.
    MimicPlay高级潜空间规划器，训练以生成基于观察和目标图像的3D轨迹。
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        创建网络并将其放入@self.nets中。
        """
        assert self.algo_config.highlevel.enabled
        assert not self.algo_config.lowlevel.enabled

        # del self.obs_shapes['robot0_eef_pos_future_traj']
        del self.obs_shapes['hand_loc_future_traj']
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        处理用于训练的输入批，以过滤相关信息并准备批以进行训练。

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        # 确保obs_normalization_stats是适当设备上的torch张量
        obs_normalization_stats = TensorUtils.to_float(
            TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), self.device))

        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        # 我们将搜索嵌套的批字典以查找以下特殊批字典键，并将处理函数应用于其值（对应于观察）
        obs_keys = ["obs", "next_obs", "goal_obs"]

        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            将process_obs_dict应用于与obs_keys中的键匹配的嵌套字典d中的值。
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    # 找到键 - 停止搜索并处理观察
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_obs(d[k], obs_normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    # 搜索到字典内部
                    recurse_helper(d[k])

        recurse_helper(batch)

        # batch["goal_obs"]["agentview_image"] = batch["goal_obs"]["agentview_image"][:, 0]
        batch["goal_obs"]["front_image_1"] = batch["goal_obs"]["front_image_1"][:, 0]
        batch["goal_obs"]["front_image_2"] = batch["goal_obs"]["front_image_2"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def _get_latent_plan(self, obs, goal):
        assert 'agentview_image' in obs.keys() # only visual inputs can generate latent plans  只有视觉输入才能生成潜在计划

        if len(obs['agentview_image'].size()) == 5:
            bs, seq, c, h, w = obs['agentview_image'].size()

            for item in ['agentview_image']:
                obs[item] = obs[item].view(bs * seq, c, h, w)
                goal[item] = goal[item].view(bs * seq, c, h, w)

            obs['robot0_eef_pos'] = obs['robot0_eef_pos'].view(bs * seq, 3)

            dists, enc_out, mlp_out = self.nets["policy"].forward_train(
                obs_dict=obs,
                goal_dict=goal,
                return_latent=True
            )

            act_out_all = dists.mean
            act_out = act_out_all

            for item in ['agentview_image']:
                obs[item] = obs[item].view(bs, seq, c, h, w)
                goal[item] = goal[item].view(bs, seq, c, h, w)

            obs['robot0_eef_pos'] = obs['robot0_eef_pos'].view(bs, seq, 3)

            enc_out_feature_size = enc_out.size()[1]
            mlp_out_feature_size = mlp_out.size()[1]

            mlp_out = mlp_out.view(bs, seq, mlp_out_feature_size)
        else:
            dists, enc_out, mlp_out = self.nets["policy"].forward_train(
                obs_dict=obs,
                goal_dict=goal,
                return_latent=True
            )

            act_out_all = dists.mean
            act_out = act_out_all

        return act_out, mlp_out

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        内部辅助函数，用于BC算法类。计算前向传递并在@predictions字典中返回网络输出。

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """

        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"]
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        # 确保这是一批多变量动作分布，以便对数概率计算正确
        assert len(dists.batch_shape) == 1
        # log_probs = dists.log_prob(batch["obs"]["robot0_eef_pos_future_traj"])
        log_probs = dists.log_prob(batch["obs"]["hand_loc_future_traj"].reshape(-1,40))

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class Lowlevel_GPT_mimicplay(BC_RNN):
    """
    MimicPlay lowlevel plan-guided robot controller, trained to output 6-DoF robot end-effector actions conditioned on generated highlevel latent plans
    MimicPlay低级计划指导的机器人控制器，训练为以生成的高级潜空间计划为条件输出6-DoF机器人末端效应器动作；包含一个训练好的高级规划器
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.highlevel.enabled
        assert self.algo_config.lowlevel.enabled

        self.human_nets, _ = FileUtils.policy_from_checkpoint(ckpt_path=self.algo_config.lowlevel.trained_highlevel_planner, device=self.device, verbose=False, update_obs_dict=False)

        self.eval_goal_img_window = self.algo_config.lowlevel.eval_goal_img_window
        self.eval_max_goal_img_iter = self.algo_config.lowlevel.eval_max_goal_img_iter

        del self.obs_shapes['agentview_image']
        self.obs_shapes['latent_plan'] = [self.algo_config.highlevel.latent_plan_dim]

        self.nets = nn.ModuleDict()

        self.nets["policy"] = GPT_wrapper(self.algo_config.lowlevel.feat_dim,
                                          self.algo_config.lowlevel.n_layer,
                                          self.algo_config.lowlevel.n_head,
                                          self.algo_config.lowlevel.block_size,
                                          self.algo_config.lowlevel.gmm_modes,
                                          self.algo_config.lowlevel.action_dim,
                                          self.algo_config.lowlevel.proprio_dim,
                                          self.algo_config.lowlevel.spatial_softmax_num_kp,
                                          self.algo_config.lowlevel.gmm_min_std,
                                          self.algo_config.lowlevel.dropout,
                                          self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_height,
                                          self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_width)

        self.buffer = []
        self.current_id = 0
        self.save_count = 0
        self.zero_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(self.goal_ee_traj[current_id : (current_id + self.eval_goal_img_window)] - ee_pos, dim=1)
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, 'r')['data']['demo_1']['obs']['agentview_image'][:]
        self.goal_ee_traj = h5py.File(video_path, 'r')['data']['demo_1']['obs']['robot0_eef_pos'][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.
        self.goal_image_length = len(self.goal_image)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]

        key_list = copy.deepcopy(list(input_batch["obs"].keys()))
        for key in key_list:
            input_batch["obs"][key] = input_batch["obs"][key]

        input_batch["goal_obs"] = batch["goal_obs"]

        input_batch["actions"] = batch["actions"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        用于BC算法类的内部辅助函数。计算前向传递并在@predictions字典中返回网络输出。

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            从数据加载器中采样的包含torch.Tensors的字典，并通过@process_batch_for_training进行过滤
        Returns:
            predictions (dict): dictionary containing network outputs
            包含网络输出的字典
        """

        with torch.no_grad():
            _, mlp_feature = self.human_nets.policy._get_latent_plan(batch['obs'], batch["goal_obs"])
            batch["obs"]['latent_plan'] = mlp_feature.detach()

        dists = self.nets["policy"].forward_train(batch["obs"])

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        # 确保这是一批多变量动作分布，以便对数概率计算正确
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        obs_to_use = obs_dict

        with torch.no_grad():
            self.goal_id = min(self.current_id + self.algo_config.playdata.eval_goal_gap, self.goal_image_length - 1)
            goal_img = {'agentview_image': self.goal_image[self.goal_id:(self.goal_id+1)]}
            action, mlp_feature = self.human_nets.policy._get_latent_plan(obs_to_use, goal_img)
            obs_to_use['latent_plan'] = mlp_feature.detach()
            obs_to_use['guidance'] = action.detach()

            self.current_id = self.find_nearest_index(obs_to_use['robot0_eef_pos'], self.current_id)

        action = self.nets["policy"].forward_step(obs_to_use)

        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.nets["policy"].reset()
        self.human_nets.policy.reset()
        self.current_id = 0


class Baseline_GPT_from_scratch(BC_RNN):
    """
    BC transformer baseline (an end-to-end version of MimicPlay's lowlevel robot controller (no highlevel planner)).
    BC transformer基线（MimicPlay的低级机器人控制器的端到端版本（没有高级规划器））
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert not self.algo_config.highlevel.enabled
        assert self.algo_config.lowlevel.enabled

        self.eval_goal_img_window = self.algo_config.lowlevel.eval_goal_img_window
        self.eval_max_goal_img_iter = self.algo_config.lowlevel.eval_max_goal_img_iter

        self.nets = nn.ModuleDict()

        self.nets["policy"] = GPT_wrapper_scratch(self.algo_config.lowlevel.feat_dim,
                                          self.algo_config.lowlevel.n_layer,
                                          self.algo_config.lowlevel.n_head,
                                          self.algo_config.lowlevel.block_size,
                                          self.algo_config.lowlevel.gmm_modes,
                                          self.algo_config.lowlevel.action_dim,
                                          self.algo_config.lowlevel.proprio_dim,
                                          self.algo_config.lowlevel.spatial_softmax_num_kp,
                                          self.algo_config.lowlevel.gmm_min_std,
                                          self.algo_config.lowlevel.dropout,
                                          self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_height,
                                          self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_width)

        self.buffer = []
        self.current_id = 0
        self.save_count = 0
        self.zero_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(self.goal_ee_traj[current_id: (current_id + self.eval_goal_img_window)] - ee_pos, dim=1)
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, 'r')['data']['demo_1']['obs']['agentview_image'][:]
        self.goal_ee_traj = h5py.File(video_path, 'r')['data']['demo_1']['obs']['robot0_eef_pos'][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.
        self.goal_image_length = len(self.goal_image)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]

        key_list = copy.deepcopy(list(input_batch["obs"].keys()))
        for key in key_list:
            input_batch["obs"][key] = input_batch["obs"][key]

        input_batch["goal_obs"] = batch["goal_obs"]

        input_batch["actions"] = batch["actions"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        dists = self.nets["policy"].forward_train(batch["obs"], batch["goal_obs"])

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        obs_to_use = obs_dict

        self.goal_id = min(self.current_id + self.algo_config.playdata.eval_goal_gap, self.goal_image_length - 1)
        goal_img = {'agentview_image': self.goal_image[self.goal_id:(self.goal_id + 1)]}

        action = self.nets["policy"].forward_step(obs_to_use, goal_img)

        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.nets["policy"].reset()
        self.current_id = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC-RNN baseline (an end-to-end baseline adapted from robomimic)
    BC-RNN基线（从robomimic调整而来的端到端基线）
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert not self.algo_config.highlevel.enabled
        assert not self.algo_config.lowlevel.enabled

        self.eval_goal_img_window = self.algo_config.lowlevel.eval_goal_img_window
        self.eval_max_goal_img_iter = self.algo_config.lowlevel.eval_max_goal_img_iter

        self.nets = nn.ModuleDict()

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.buffer = []
        self.current_id = 0
        self.save_count = 0
        self.zero_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(self.goal_ee_traj[current_id: (current_id + self.eval_goal_img_window)] - ee_pos, dim=1)
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, 'r')['data']['demo_1']['obs']['agentview_image'][:]
        self.goal_ee_traj = h5py.File(video_path, 'r')['data']['demo_1']['obs']['robot0_eef_pos'][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.
        self.goal_image_length = len(self.goal_image)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]

        key_list = copy.deepcopy(list(input_batch["obs"].keys()))
        for key in key_list:
            input_batch["obs"][key] = input_batch["obs"][key]

        input_batch["goal_obs"] = batch["goal_obs"]
        input_batch["actions"] = batch["actions"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self.goal_id = min(self.current_id + self.algo_config.playdata.eval_goal_gap, self.goal_image_length - 1)
        goal_dict = {'agentview_image': self.goal_image[self.goal_id:(self.goal_id + 1)].unsqueeze(0)}

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log