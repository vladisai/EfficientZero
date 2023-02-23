import contextlib
import torch
import pickle
import torch.nn as nn
from core.model import BaseNet, renormalize

from rl_ssl.config import Config
from rl_ssl.models.model import build_model, Model
from rl_ssl.models.losses import build_criterion


def load_RL_SSL_model(ckpt: str) -> Model:

    # @contextlib.contextmanager
    # def monkey_patch_unpickler():
    #     OrigUnpickler = pickle.Unpickler

    #     class MyUnpickler(pickle.Unpickler):
    #         def find_class(self, modname: str, clsname):
    #             if modname.startswith('config.'):
    #                 modname = 'config.rl_ssl.' + modname
    #             return super().find_class(modname, clsname)

    #     pickle.Unpickler = MyUnpickler
    #     yield
    #     pickle.Unpickler = OrigUnpickler

    # with monkey_patch_unpickler():
    #     ckpt = torch.load(ckpt, map_location='cpu')
    ckpt = torch.load(ckpt, map_location='cpu')

    config: Config = ckpt['args']

    model: Model = build_model(config)
    model.load_state_dict(ckpt['model'])

    loss = build_criterion(config.pretraining)
    assert len(loss.state_dict()) == 0
    return model, loss



# class RLSSLPretrainedNet(BaseNet):
#     ckpt: str
#     def __init__(
#         self,
#         ckpt: str,
#         inverse_value_transform,
#         inverse_reward_transform,
#         lstm_hidden_size,
#     ):
#         """EfficientZero network
#         Parameters
#         ----------
#         ckpt: str
#             Saved RL_SSL checkpoint
#         inverse_value_transform: Any
#             A function that maps value supports into value scalars
#         inverse_reward_transform: Any
#             A function that maps reward supports into value scalars
#         lstm_hidden_size: int
#             dim of lstm hidden
#         """
#         super().__init__(
#             inverse_value_transform, inverse_reward_transform, lstm_hidden_size
#         )
#         self.ckpt = ckpt
#         rl_ssl_model = load_RL_SSL_model(ckpt)
#         self.backbone = rl_ssl_model.backbone
#         self.forward_model = rl_ssl_model.forward_model
#         self.proj_hid = proj_hid
#         self.proj_out = proj_out
#         self.pred_hid = pred_hid
#         self.pred_out = pred_out
#         self.init_zero = init_zero
#         self.state_norm = state_norm

#         self.action_space_size = action_space_size
#         block_output_size_reward = (
#             (
#                 reduced_channels_reward
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
#         )

#         block_output_size_value = (
#             (
#                 reduced_channels_value
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_value * observation_shape[1] * observation_shape[2])
#         )

#         block_output_size_policy = (
#             (
#                 reduced_channels_policy
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
#         )

#         self.representation_network = RepresentationNetwork(
#             observation_shape,
#             num_blocks,
#             num_channels,
#             downsample,
#             momentum=bn_mt,
#         )

#         self.dynamics_network = DynamicsNetwork(
#             num_blocks,
#             num_channels + 1,
#             reduced_channels_reward,
#             fc_reward_layers,
#             reward_support_size,
#             block_output_size_reward,
#             lstm_hidden_size=lstm_hidden_size,
#             momentum=bn_mt,
#             init_zero=self.init_zero,
#         )

#         self.prediction_network = PredictionNetwork(
#             action_space_size,
#             num_blocks,
#             num_channels,
#             reduced_channels_value,
#             reduced_channels_policy,
#             fc_value_layers,
#             fc_policy_layers,
#             value_support_size,
#             block_output_size_value,
#             block_output_size_policy,
#             momentum=bn_mt,
#             init_zero=self.init_zero,
#         )

#         # projection
#         in_dim = (
#             num_channels
#             * math.ceil(observation_shape[1] / 16)
#             * math.ceil(observation_shape[2] / 16)
#         )
#         self.porjection_in_dim = in_dim
#         self.projection = nn.Sequential(
#             nn.Linear(self.porjection_in_dim, self.proj_hid),
#             nn.BatchNorm1d(self.proj_hid),
#             nn.ReLU(),
#             nn.Linear(self.proj_hid, self.proj_hid),
#             nn.BatchNorm1d(self.proj_hid),
#             nn.ReLU(),
#             nn.Linear(self.proj_hid, self.proj_out),
#             nn.BatchNorm1d(self.proj_out),
#         )
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.proj_out, self.pred_hid),
#             nn.BatchNorm1d(self.pred_hid),
#             nn.ReLU(),
#             nn.Linear(self.pred_hid, self.pred_out),
#         )

#     def prediction(self, encoded_state):
#         policy, value = self.prediction_network(encoded_state)
#         return policy, value

#     def representation(self, observation):
#         encoded_state = self.representation_network(observation)
#         if not self.state_norm:
#             return encoded_state
#         else:
#             encoded_state_normalized = renormalize(encoded_state)
#             return encoded_state_normalized

#     def dynamics(self, encoded_state, reward_hidden, action):
#         # Stack encoded_state with a game specific one hot encoded action
#         action_one_hot = (
#             torch.ones(
#                 (
#                     encoded_state.shape[0],
#                     1,
#                     encoded_state.shape[2],
#                     encoded_state.shape[3],
#                 )
#             )
#             .to(action.device)
#             .float()
#         )
#         action_one_hot = (
#             action[:, :, None, None] * action_one_hot / self.action_space_size
#         )
#         x = torch.cat((encoded_state, action_one_hot), dim=1)
#         next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(
#             x, reward_hidden
#         )

#         if not self.state_norm:
#             return next_encoded_state, reward_hidden, value_prefix
#         else:
#             next_encoded_state_normalized = renormalize(next_encoded_state)
#             return next_encoded_state_normalized, reward_hidden, value_prefix

#     def get_params_mean(self):
#         representation_mean = self.representation_network.get_param_mean()
#         dynamic_mean = self.dynamics_network.get_dynamic_mean()
#         reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

#         return reward_w_dist, representation_mean, dynamic_mean, reward_mean

#     def project(self, hidden_state, with_grad=True):
#         # only the branch of proj + pred can share the gradients
#         hidden_state = hidden_state.view(-1, self.porjection_in_dim)
#         proj = self.projection(hidden_state)

#         # with grad, use proj_head
#         if with_grad:
#             proj = self.projection_head(proj)
#             return proj
#         else:
#             return proj.detach()

#     def extra_repr(self) -> str:
#         return f"ckpt={self.ckpt}"