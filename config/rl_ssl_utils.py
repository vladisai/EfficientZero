import contextlib
import torch
import numpy as np
import math
import torch.nn as nn
from core.model import BaseNet, renormalize

from rl_ssl.config import Config
from rl_ssl.models.model import build_model, Model
from rl_ssl.models.losses import build_criterion, PretrainCriterion
from config.atari.model import PredictionNetwork, conv3x3, ResidualBlock, mlp, BatchNorm1dLastDim


def load_RL_SSL_model(ckpt: str):
    ckpt = torch.load(ckpt, map_location='cpu')

    config: Config = ckpt['args']

    model: Model = build_model(config)
    model.load_state_dict(ckpt['model'])

    loss = build_criterion(config.pretraining)
    assert len(loss.state_dict()) == 0
    return model, loss



class ValuePrefixLSTM(nn.Module):
    def __init__(
        self,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size

        self.conv1x1_reward = nn.Conv2d(num_channels, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(
            input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size
        )
        self.bn_value_prefix = BatchNorm1dLastDim(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(
            self.lstm_hidden_size,
            fc_reward_layers,
            full_support_size,
            init_zero=init_zero,
            momentum=momentum,
        )

    def forward(self, new_state, reward_hidden):
        x = self.conv1x1_reward(new_state)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return reward_hidden, value_prefix

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean



class RLSSLPretrainedNet(BaseNet):
    ckpt: str

    def __init__(
        self,
        ckpt: str,
        freeze: bool,
        load_loss: bool,

        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        bn_mt=0.1,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False,
    ):
        """EfficientZero network
        Parameters
        ----------
        ckpt: str
            Saved RL_SSL checkpoint
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        """
        super().__init__(
            inverse_value_transform, inverse_reward_transform, lstm_hidden_size
        )
        self.ckpt = ckpt
        rl_ssl_model, pretrain_criterion = load_RL_SSL_model(ckpt)
        if freeze:
            for p in rl_ssl_model.parameters():
                p.requires_grad_(False)
        assert rl_ssl_model.train_context_length == 1, 'NYI'

        self.backbone = rl_ssl_model.backbone
        self.backbone.unfolded_input = False
        self.target_backbone = rl_ssl_model.target_backbone
        self.target_backbone.unfolded_input = False
        self.forward_model = rl_ssl_model.forward_model
        self.pretrain_criterion = pretrain_criterion
        self.freeze = freeze
        self.load_loss = load_loss
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size
        block_output_size_reward = reduced_channels_reward * self.backbone.latent_res * self.backbone.latent_res
        block_output_size_value = reduced_channels_value * self.backbone.latent_res * self.backbone.latent_res
        block_output_size_policy = reduced_channels_policy * self.backbone.latent_res * self.backbone.latent_res

        self.value_prefix_network = ValuePrefixLSTM(
            self.backbone.out_size,
            reduced_channels_reward,
            fc_reward_layers,
            reward_support_size,
            block_output_size_reward,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            block_output_size_value,
            block_output_size_policy,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        self.ssl_proj_head = rl_ssl_model.ssl_proj_head
        self.add_module('inverse_head', rl_ssl_model.inverse_head)

        if rl_ssl_model.predictor_head is not None:
            self.predictor_head = rl_ssl_model.predictor_head
        else:
            # initialize a new one (@ssnl: is this the right thing?)
            self.predictor_head = nn.Sequential(
                nn.Linear(rl_ssl_model.ssl_proj_head.output_dim, pred_hid),
                BatchNorm1dLastDim(pred_hid),
                nn.ReLU(),
                nn.Linear(pred_hid, pred_out),
            )

    def consistency_loss_unreduced(self, proj: torch.Tensor, action: torch.Tensor,
                                   target: torch.Tensor, mask: torch.Tensor):
        if self.load_loss:
            output = dict(
                ssl_proj_output=(proj, target),
                ssl_pred_output=self.pred_project(proj),
            )
            if self.inverse_head is not None:
                output.update(
                    predicted_inverse_actions=self.inverse_head(proj[:, :-1], target[:, 1:]),
                )
            assert action.shape[-1] == 1
            mask = (mask > 0)  # becomes bool
            losses = self.pretrain_criterion(output, action=action.squeeze(-1), mask=mask, reduced=False)
            total_l = 0
            for l in losses.values():
                assert l.shape == mask.shape
                total_l += l
            return total_l
        else:
            return super().consistency_loss_unreduced(proj, action, target, mask)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def postprocess_encoded_state(self, encoded_state):
        if self.state_norm:
            encoded_state = renormalize(encoded_state)
        return encoded_state

    def representation(self, observation, target: bool = False):
        # observation: [B, S * C, H, W]
        # output: [B, outC, outH, outW]
        backbone = self.target_backbone if target else self.backbone
        observation = observation.unflatten(-3, (backbone.hist_len, -1))  # [B, S, C, H, W]
        encoded_state = backbone(observation)  # [B, 1, outC, outH, outW]
        encoded_state = encoded_state.squeeze(-4)
        return self.postprocess_encoded_state(encoded_state)

    def target_embedding(self, obs):
        return self.representation(obs, target=True)

    def dynamics(self, encoded_state, reward_hidden, action):
        # encoded_state: [B, outC, outH, outW]
        # action: [B, 1]
        assert action.ndim == 2 and action.shape[-1] == 1
        assert encoded_state.ndim == 4
        # flat_encoded_state: [B, D]
        flat_encoded_state = encoded_state.flatten(-3)

        # forward model takes in
        # flat_encoded_state:  [B, CtxL        , D]
        # action: [B, CtxL + PredL]
        # returns [s0, predS1, predS2, ...], where only action[:, :-1] are used, as a [B, PredL, D] tensor

        pred_states = self.forward_model(
            flat_encoded_state[:, None, :],  # CtxL == 1
            action.expand(-1, 2),
        ).pred_states

        assert pred_states.ndim == 3 and pred_states.shape[1] == 2
        next_encoded_state = pred_states[:, -1].unflatten(-1, encoded_state.shape[-3:])

        reward_hidden, value_prefix = self.value_prefix_network(next_encoded_state, reward_hidden)
        next_encoded_state = self.postprocess_encoded_state(next_encoded_state)
        return next_encoded_state, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = np.nan
        dynamic_mean = np.nan

        reward_w_dist, reward_mean = self.value_prefix_network.get_reward_mean()
        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state):
        # hidden_state: [B, outC, outH, outW]
        projected = self.ssl_proj_head(
            hidden_state.flatten(1)[:, None, :],  # proj head takes in [B, T, D]
        )  # [B, 1, projD]
        assert projected.ndim == 3 and projected.shape[-2] == 1
        return projected.squeeze(-2)


    def pred_project(self, projected):
        return self.predictor_head(projected)

    def extra_repr(self) -> str:
        return f"ckpt={self.ckpt}, freeze={self.freeze}"