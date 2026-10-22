import torch
import torch.nn as nn
import numpy as np
from layers.RevIN import RevIN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


# Simple feed-forward network for feature refinement
class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()

        # Two-layer MLP
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


#Positional embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):

        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        return x + self.pos_embedding(positions)


class TimeExpert(nn.Module):
    def __init__(self, seq_len, pred_len, channels, hidden=256):
        super().__init__()

        self.embed = PositionalEncoding(
            d_model=channels
        )

        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, pred_len)
        )

    def forward(self, x):

        # x: [B,T,C]
        x = self.embed(x)

        x = x.permute(0, 2, 1)  # [B,C,T]

        x = self.net(x)  # [B,C,P]

        return x.permute(0, 2, 1)  # [B,P,C]


# Router for selecting experts
class Router(nn.Module):

    def __init__(
        self,
        channels,
        num_experts=8,
        hidden=128,
        top_k=4
    ):
        super().__init__()

        self.top_k = top_k
        # Generate one global gate from a learned seed, without using input data.
        self.gate_seed = nn.Parameter(torch.ones(1, num_experts))
        self.gate_net = nn.Sequential(
            nn.Linear(num_experts, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts)
        )
        # Start with logits [1, ..., 1], hence a uniform normalized gate.
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.ones_(self.gate_net[-1].bias)

    def forward(self, x, routing_input=None):
        # The learned gate is global: it does not depend on input samples.
        logits = self.gate_net(self.gate_seed)
        score = F.softmax(logits, dim=-1).expand(x.size(0), -1)

        # Select top-k experts
        value, index = torch.topk(
            score,
            self.top_k,
            dim=-1
        )

        mask = torch.zeros_like(score)

        mask.scatter_(
            1,
            index,
            value
        )

        gate = mask / (
            mask.sum(-1, keepdim=True)
            + 1e-8
        )


        return gate


# Frequency-domain leader based channel mixing
class LeaderSparseMix(nn.Module):

    def __init__(self, channels, seq_len, pre_len, k=5):
        super().__init__()

        self.k = k

        # Learnable leader representation
        self.W = nn.Parameter(
            torch.randn(channels, k) * 0.02
        )

        self.mlp = nn.Sequential(
            nn.Linear(k, 2*k).to(torch.cfloat),
            nn.Linear(2*k, channels).to(torch.cfloat)
        )

        self.project = nn.Linear(
            seq_len//2+1,
            pre_len//2+1
        ).to(torch.cfloat)


    def forward(self, x, selection_input=None):

        # x: [B,T,C]
        B, T, C = x.shape

        fre = torch.fft.rfft(
            x,
            dim=1
        )

        # RevIN makes per-channel temporal variance nearly constant. Select
        # leaders using the original input statistics instead.
        selection_input = x if selection_input is None else selection_input
        score = selection_input.std(dim=1)

        k = min(self.k, C)

        idx = torch.topk(
            score,
            k,
            dim=-1
        )[1]


        leaders = torch.gather(
            fre.transpose(1, 2),
            1,
            idx.unsqueeze(-1).expand(
                -1,
                -1,
                T//2+1
            )
        )


        out = self.mlp(
            leaders.permute(0,2,1)
        ).permute(0,2,1)


        out = self.project(out)

        out = torch.fft.irfft(
            out,
            dim=2
        )

        return out   # [B,T,C]


class TimeMoE(nn.Module):

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        num_experts=4,
        top_k=4
    ):
        super().__init__()

        self.num_bands = num_experts
        self.pred_len = pred_len
        self.router = Router(
            channels,
            self.num_bands,
            top_k=top_k
        )

        self.experts = nn.ModuleList([
            TimeExpert(seq_len, pred_len, channels)
            for _ in range(self.num_bands)
        ])
        # A full-input residual branch. Its zero-initialized last layer makes
        # it an exact no-op at initialization; training learns only a useful
        # correction on top of the frequency-band mixture.
        self.global_expert = TimeExpert(seq_len, pred_len, channels)
        nn.init.zeros_(self.global_expert.net[-1].weight)
        nn.init.zeros_(self.global_expert.net[-1].bias)

        F = seq_len // 2 + 1
        masks = []
        band = math.ceil(F / self.num_bands)
        overlap = min(6, max(1, band // 4))
        for i in range(self.num_bands):
            start = (i * F) // self.num_bands
            end = ((i + 1) * F) // self.num_bands
            left = max(start - overlap, 0)
            right = min(max(end + overlap, left + 1), F)
            mask = torch.zeros(F)
            mask[left:right] = 1
            masks.append(mask)

        # Save masks as fixed buffers
        self.register_buffer(
            "freq_masks",
            torch.stack(masks)
        )


    def forward(self, x, routing_input=None):

        gate = self.router(x, routing_input)

        fre = torch.fft.rfft(
            x,
            dim=1
        )
        # The current Router produces one global gate shared by the batch.
        # Select active bands once, then filter and inverse-transform them in
        # a single batched FFT call instead of issuing one call per expert.
        active_indices = torch.nonzero(gate[0] > 0, as_tuple=False).flatten()
        active_indices_list = active_indices.tolist()
        active_masks = self.freq_masks[active_indices]
        band_spectrum = fre[:, None, :, :] * active_masks[None, :, :, None]
        band_signals = torch.fft.irfft(
            band_spectrum,
            n=x.size(1),
            dim=2
        )  # [B,K,T,C]

        # Append the full-input residual expert to the active band experts,
        # then apply their independent Linear weights as batched einsums.
        experts = [self.experts[i] for i in active_indices_list]
        experts.append(self.global_expert)
        expert_inputs = torch.cat([band_signals, x.unsqueeze(1)], dim=1)
        position_embeddings = torch.stack([
            expert.embed.pos_embedding.weight[:x.size(1)]
            for expert in experts
        ])
        hidden_inputs = (expert_inputs + position_embeddings[None]).permute(
            0, 1, 3, 2
        )  # [B,N,C,T]

        fc1_weight = torch.stack([expert.net[0].weight for expert in experts])
        fc1_bias = torch.stack([expert.net[0].bias for expert in experts])
        hidden = torch.einsum(
            "bnct,nht->bnch", hidden_inputs, fc1_weight
        ) + fc1_bias[None, :, None, :]
        hidden = F.gelu(hidden)

        fc2_weight = torch.stack([expert.net[2].weight for expert in experts])
        fc2_bias = torch.stack([expert.net[2].bias for expert in experts])
        expert_outputs = torch.einsum(
            "bnch,nph->bncp", hidden, fc2_weight
        ) + fc2_bias[None, :, None, :]
        expert_outputs = expert_outputs.permute(0, 1, 3, 2)  # [B,N,P,C]

        num_active_bands = len(active_indices_list)
        active_gate = gate[0, active_indices]
        band_outputs = expert_outputs[:, :num_active_bands]
        out = (
            band_outputs * active_gate[None, :, None, None]
        ).sum(dim=1)
        global_residual = expert_outputs[:, num_active_bands]
        out = out + global_residual
        outputs = [band_outputs[:, i] for i in range(num_active_bands)]

        self.fig_out = band_outputs
        self.glo=global_residual
        self.gate = out/global_residual
        self.global_residual = global_residual


        return out



class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()


        # Basic configurations
        self.pre_length = configs.pred_len
        self.seq_length = configs.seq_len
        self.channels = configs.enc_in


        self.leader_topk = configs.leader_topk
        self.num_experts = configs.num_experts
        self.activated_experts = configs.activated_experts

        self.revin = configs.revin


        # RevIN normalization
        self.revin_layer = RevIN(
            self.channels,
            affine=True,
            subtract_last=False
        )


        # Leader Sparse Mixing
        self.leader_mix = LeaderSparseMix(
            self.channels,
            seq_len=self.seq_length,
            pre_len=self.pre_length,
            k=self.leader_topk
        )


        # Output refinement layer
        self.feed = FeedForward(
            self.pre_length,
            self.pre_length
        )


        # Frequency-aware TimeMoE module
        self.time_model = TimeMoE(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            channels=configs.enc_in,
            num_experts=self.num_experts,
            top_k=self.activated_experts
        )


    def forward(self, x):
        selection_input=x
        # Apply RevIN normalization
        if self.revin:
            x = self.revin_layer(
                x,
                'norm'
            )


        # MoE forecasting branch
        y_time = self.time_model(x)

        y_time = y_time.permute(
            0,
            2,
            1
        )


        # Leader-based residual branch
        residual = self.leader_mix(x, selection_input=None)


        # Combine two branches
        y_feed = y_time + residual


        # Final projection
        y = self.feed(y_feed).permute(
            0,
            2,
            1
        )


        # Restore original scale
        if self.revin:
            y = self.revin_layer(
                y,
                'denorm'
            )


        return y, None
