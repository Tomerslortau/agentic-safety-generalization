import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPController(nn.Module):
    """
    A controller that directly maps state (and potentially system params) to action.
    """
    def __init__(
        self,
        input_dim,
        hidden_size,
        output_dim,
        num_hidden_layers
    ):
        super(MLPController, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            if i > 0:
                x = F.relu(layer(x)) + x
            else:
                x = F.relu(layer(x))
        logits = self.out_layer(x)               # ℝ
        # MAke the output between 0 and 1
        return F.tanh(logits)




class RSNorm(nn.Module):
    """
    Running-Statistics Normalization per input dimension.
    Tracks mean and variance online during training, uses frozen stats in eval.
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("M2", torch.zeros(dim))  # sum of squared diffs
        self.eps = eps

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor):
        x2d = x.reshape(-1, x.shape[-1])
        batch_count = torch.tensor([x2d.size(0)], device=x.device, dtype=x.dtype)

        delta = x2d.mean(dim=0) - self.mean
        total_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * (batch_count / total_count)

        # Update M2 (parallel algorithm)
        var_batch = x2d.var(dim=0, unbiased=False)
        M2_batch = var_batch * batch_count
        delta2 = x2d.mean(dim=0) - new_mean
        new_M2 = self.M2 + M2_batch + (delta * delta2) * (self.count * batch_count / total_count)

        self.count = total_count
        self.mean = new_mean
        self.M2 = new_M2

    def forward(self, x: torch.Tensor):
        if self.training:
            self._update_stats(x)
        count = torch.clamp(self.count, min=1.0)
        var = self.M2 / count
        return (x - self.mean) / torch.sqrt(var + self.eps)


class ResidualFFNBlock(nn.Module):
    """
    Pre-LayerNorm residual feedforward block with 4x expansion and ReLU.
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        hidden = expansion * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        h = self.ln(x)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class SimBaController(nn.Module):
    """
    SimBa MLP Controller:
      - RSNorm on inputs
      - Linear embed to hidden
      - N pre-LN residual FFN blocks (4x expansion, ReLU)
      - Post LayerNorm
      - Output head + tanh
    """
    def __init__(self, input_dim, hidden_size, output_dim, num_blocks):
        super().__init__()
        self.in_norm = RSNorm(input_dim)
        self.embed = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualFFNBlock(hidden_size, expansion=4) for _ in range(num_blocks)])
        self.post_ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.post_ln(x)
        logits = self.head(x)
        return torch.tanh(logits)


class AdverserialModel(nn.Module):
    """
    A controller that directly maps state (and potentially system params) to action.
    """
    def __init__(
        self,
        input_dim,
        hidden_size,
        output_dim,
        num_hidden_layers
    ):
        super(AdverserialModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            if i > 0:
                x = F.relu(layer(x)) + x
            else:
                x = F.relu(layer(x))
        logits = self.out_layer(x)               
        return logits


class HistoryEncoder(nn.Module):
    """
    Encodes the recent history of states and actions and
    predicts an adversarial force (wind) vector.

    Inputs:
      - history_states: [B, T, state_dim]
      - history_actions: [B, T, action_dim]

    Parameters: - action_dim: typically 4
      - history_length: T (number of steps to look back)
      - hidden_size: size of MLP hidden layers
      - emb_dim: size of the produced embedding

    Outputs:
      - wind_pred: [B, 3] (x,y,z adversarial force)
    """
    def __init__(
        self,
        history_length: int = 3,
        hidden_size: int = 128,
        wind_dim: int = 3,
        num_layers: int = 3,
    ):
        super().__init__()
        self.history_length = history_length
        in_dim = history_length * (12 + 4) #state_dim + action_dim

        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_size))
            dim = hidden_size
        self.backbone = nn.ModuleList(layers)
        self.wind_head = nn.Linear(hidden_size, wind_dim)

    def forward(self, history_states: torch.Tensor, history_actions: torch.Tensor):
        # history_states: [B, T*S], history_actions: [B, T*A]
        B = history_states.shape[0]
        hist = torch.cat([history_states, history_actions], dim=-1)  # [B, T, S+A]
        hist_flat = hist.reshape(B, -1)  # [B, T*(S+A)]

        x = hist_flat
        for layer in self.backbone:
            x = F.relu(layer(x))
        wind = self.wind_head(x)
        return wind


class EncoderConditionedController(nn.Module):
    """
    Controller that uses an encoder over history to produce an embedding
    (and wind prediction), and then outputs the next action given
    current state + target goal + embedding.

    Inputs:
      - current_state: [B, state_dim]
      - target_goal:   [B, goal_dim] (typically 3)
      - history_states:  [B, T, state_dim]
      - history_actions: [B, T, action_dim]

    Outputs:
      - action: [B, action_dim]
      - wind_pred: [B, 3]
    """
    def __init__(
        self,
        history_length: int = 3,
        enc_hidden: int = 128,
        ctrl_hidden: int = 128,
        ctrl_layers: int = 2,
        output_dim: int = 4,
        auxiliary_output: bool = False,
    ):
        super().__init__()
        self.history_length = history_length
        self.debug = debug

        # Encoder
        self.encoder = HistoryEncoder(
            history_length=history_length,
            hidden_size=enc_hidden,
            wind_dim=3,
            num_layers=2,
        )

        # Controller MLP: input = state + goal + wind
        ctrl_in = 12 + 3 + 3
        self.controller = MLPController(
            input_dim=ctrl_in,
            hidden_size=ctrl_hidden,
            output_dim=output_dim,
            num_hidden_layers=ctrl_layers
        )


    def forward(self,x):
        #divide the input to [history_states, history_actions, current_state, target_goal]
        history_states = x[..., :self.history_length*12]
        history_actions = x[..., self.history_length*12:self.history_length*12+self.history_length*4]
        current_state = x[..., -15:-3]#last 15 to 3 coordinates
        target_goal = x[..., -3:]#last 3 coordinates
        # Encode history (predict wind)
        wind = self.encoder(history_states, history_actions)
        # Build controller input
        ctrl_in = torch.cat([current_state, target_goal, wind], dim=-1)
        x = ctrl_in
        action = self.controller(x)
        if self.auxiliary_output:
            return action, wind
        else:
            return action