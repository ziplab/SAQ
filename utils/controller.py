import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(
        self,
        n_layers,
        bits=[2, 3, 4, 5, 6, 7, 8],
        hidden_size=64,
        batch_size=1,
        device="cpu",
    ):
        super(Controller, self).__init__()
        self.n_layers = n_layers
        self.bits = bits
        self.hidden_size = hidden_size

        self.bit_embedding = nn.Embedding(len(self.bits), self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.weight_bits_linear = nn.Linear(self.hidden_size, len(self.bits))
        self.activation_bits_linear = nn.Linear(self.hidden_size, len(self.bits))

        self.batch_size = batch_size
        self.device = device
        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros(
            (batch_size, self.hidden_size), device=self.device, requires_grad=False
        )

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def random_sample(self):
        bits_seq = []
        for layer_index in range(self.n_layers):
            for _ in ["weight", "activation"]:
                bits_seq.append(random.choice(self.bits))
        return bits_seq

    def forward(self):
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = self._zeros(self.batch_size)

        bits_seq = []
        probs_buf = []
        logp_buf = []
        entropy_buf = []

        for layer_index in range(self.n_layers):
            # get weight bit
            hx, cx = self.lstm(embed, hidden)
            hidden = (hx, cx)
            logits = self.weight_bits_linear(hx)
            probs = F.softmax(logits, dim=1)
            action, select_log_p, entropy = self._impl(probs)

            r = self.bits[action.item()]
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)
            bits_seq.append(r)
            probs_buf.append(probs)

            embed = self.bit_embedding(action)

            # get activation bit
            hx, cx = self.lstm(embed, hidden)
            hidden = (hx, cx)
            logits = self.activation_bits_linear(hx)
            probs = F.softmax(logits, dim=1)
            action, select_log_p, entropy = self._impl(probs)

            r = self.bits[action.item()]
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)
            bits_seq.append(r)
            probs_buf.append(probs)

            embed = self.bit_embedding(action)

        return bits_seq, probs_buf, sum(logp_buf), sum(entropy_buf)


class WABEController(nn.Module):
    def __init__(
        self,
        n_layers,
        bits=[2, 3, 4, 5, 6, 7, 8],
        hidden_size=64,
        batch_size=1,
        device="cpu",
    ):
        super(WABEController, self).__init__()
        self.n_layers = n_layers
        self.bits = bits
        self.hidden_size = hidden_size

        self.bit_embedding = nn.Embedding(len(self.bits), self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.weight_bits_linear = nn.Linear(self.hidden_size, len(self.bits))

        self.batch_size = batch_size
        self.device = device
        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros(
            (batch_size, self.hidden_size), device=self.device, requires_grad=False
        )

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def random_sample(self):
        bits_seq = []
        for layer_index in range(self.n_layers):
            bits_seq.append(random.choice(self.bits))
        return bits_seq

    def forward(self):
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = self._zeros(self.batch_size)

        bits_seq = []
        probs_buf = []
        logp_buf = []
        entropy_buf = []

        for layer_index in range(self.n_layers):
            # get weight bit
            hx, cx = self.lstm(embed, hidden)
            hidden = (hx, cx)
            logits = self.weight_bits_linear(hx)
            probs = F.softmax(logits, dim=1)
            action, select_log_p, entropy = self._impl(probs)

            r = self.bits[action.item()]
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)
            bits_seq.append(r)
            probs_buf.append(probs)

            embed = self.bit_embedding(action)

        return bits_seq, probs_buf, sum(logp_buf), sum(entropy_buf)


class WABEControllerDist(nn.Module):
    def __init__(
        self,
        n_layers,
        bits=[2, 3, 4, 5, 6, 7, 8],
        hidden_size=64,
        batch_size=1,
        device="cpu",
    ):
        super(WABEControllerDist, self).__init__()
        self.n_layers = n_layers
        self.bits = bits
        self.hidden_size = hidden_size

        self.bit_embedding = nn.Embedding(len(self.bits), self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.weight_bits_linear = nn.Linear(self.hidden_size, len(self.bits))

        self.batch_size = batch_size
        self.device = device
        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros(
            (batch_size, self.hidden_size), device=self.device, requires_grad=False
        )

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def random_sample(self):
        bits_seq = []
        for layer_index in range(self.n_layers):
            bits_seq.append(random.choice(self.bits))
        return bits_seq

    def get_action_list(self, actions):
        return [self.bits[action.item()] for action in actions]

    def forward(self):
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = self._zeros(self.batch_size)

        bits_seq = []
        probs_buf = []
        logp_buf = []
        entropy_buf = []

        for layer_index in range(self.n_layers):
            # get weight bit
            hx, cx = self.lstm(embed, hidden)
            hidden = (hx, cx)
            logits = self.weight_bits_linear(hx)
            probs = F.softmax(logits, dim=1)
            actions, select_log_p, entropy = self._impl(probs)

            r = self.get_action_list(actions)
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)
            bits_seq.append(r)
            probs_buf.append(probs)

            embed = self.bit_embedding(actions)

        # only use the first element
        # only_first_bits_seq = []
        # only_first_probs_buf = []
        # only_first_logp_buf = []
        # only_first_entropy_buf = []
        # for layer_index in range(len(bits_seq)):
        #     only_first_bits_seq.append(bits_seq[layer_index][0])
        #     only_first_probs_buf.append(probs_buf[layer_index][0])
        #     only_first_logp_buf.append(logp_buf[layer_index][0])
        #     only_first_entropy_buf.append(entropy_buf[layer_index][0])
        # return (
        #     only_first_bits_seq,
        #     only_first_probs_buf,
        #     sum(only_first_logp_buf),
        #     sum(only_first_entropy_buf),
        # )
        return bits_seq, probs_buf, sum(logp_buf), sum(entropy_buf)

