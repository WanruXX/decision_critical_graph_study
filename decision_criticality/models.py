"""Graph-conditioned action values and frozen-checkpoint inference."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "message_passing"
    hidden_dim: int = 32
    message_steps: int = 6


class GraphActionModel(nn.Module):
    def __init__(self, config=ModelConfig(), input_dim=40):
        super().__init__()
        if config.variant not in ("message_passing", "endpoint_mlp") or min(config.hidden_dim, config.message_steps, input_dim) <= 0:
            raise ValueError("known variant and positive dimensions required")
        self.encoder = nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.SiLU())
        self.message_layers = nn.ModuleList([nn.Linear(3 * config.hidden_dim, config.hidden_dim)
                                             for _ in range(config.message_steps if config.variant == "message_passing" else 0)])
        self.head = nn.Sequential(nn.Linear(5 * config.hidden_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 1))

    def forward(self, packed):
        nodes, adjacency, actions = packed
        action_count = nodes.shape[1]
        nodes, adjacency, actions = nodes.flatten(0, 1), adjacency.flatten(0, 1), actions.flatten()
        mask = nodes.abs().sum(dim=-1, keepdim=True) > 0
        hidden = self.encoder(nodes) * mask
        presence = (adjacency > 0).to(hidden.dtype)
        outgoing_degree = presence.sum(dim=-1, keepdim=True).clamp_min(1)
        incoming_degree = presence.sum(dim=-2).unsqueeze(-1).clamp_min(1)
        for layer in self.message_layers:
            outgoing = adjacency @ hidden / outgoing_degree
            incoming = adjacency.transpose(-1, -2) @ hidden / incoming_degree
            hidden = (hidden + torch.nn.functional.silu(layer(torch.cat((hidden, outgoing, incoming), dim=-1)))) * mask
        selected = hidden[torch.arange(len(actions), device=actions.device), actions]
        mean = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        context = torch.cat((hidden[:, 0], selected, hidden[:, 0], selected, mean), dim=-1)
        return self.head(context).reshape(-1, action_count)


def predict(model, packed):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(tuple(values[start:start + 64] for values in packed)).softmax(-1)
                          for start in range(0, len(packed[0]), 64)]).numpy().astype(float)


def predict_observations(observations, models, pack):
    indices, encoded, restore = {}, [], []
    for item in observations:
        packed = pack([item["view"]])
        signature = b"".join(tensor.numpy().tobytes() for tensor in packed)
        if signature not in indices:
            indices[signature] = len(encoded)
            encoded.append(tuple(tensor[0] for tensor in packed))
        restore.append(indices[signature])
    if not encoded:
        raise ValueError("at least one observation required")
    chunks = {name: [] for name in models}
    for start in range(0, len(encoded), 128):
        packed = tuple(torch.stack([row[column] for row in encoded[start:start + 128]]) for column in range(3))
        for name, model in models.items():
            chunks[name].append(predict(model, packed))
    return {name: np.concatenate(values)[restore] for name, values in chunks.items()}