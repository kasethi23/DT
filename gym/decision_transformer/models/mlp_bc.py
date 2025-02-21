import numpy as np
import torch
import torch.nn as nn
import unittest

from decision_transformer.models.model import TrajectoryModel

#  this is an MLP that just predicts the appropriate action a_n given for si, ri, aj where i is [0...n] and j is [0...n-1]
#  have to test it
class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s.
    """


    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]
    

# import unittest
# import torch
# import numpy as np
# from decision_transformer.models.model import TrajectoryModel
# from decision_transformer.models.model import MLPBCModel  # Assuming it's in the same module

# class TestMLPBCModel(unittest.TestCase):

#     def setUp(self):
#         """Set up test environment with predefined model parameters."""
#         self.state_dim = 4
#         self.act_dim = 2
#         self.hidden_size = 64
#         self.n_layer = 2
#         self.max_length = 3
#         self.model = MLPBCModel(
#             state_dim=self.state_dim,
#             act_dim=self.act_dim,
#             hidden_size=self.hidden_size,
#             n_layer=self.n_layer,
#             max_length=self.max_length
#         )

#     def test_model_initialization(self):
#         """Test that model initializes with correct attributes."""
#         self.assertEqual(self.model.hidden_size, self.hidden_size)
#         self.assertEqual(self.model.max_length, self.max_length)
#         self.assertEqual(self.model.act_dim, self.act_dim)

#     def test_forward_pass(self):
#         """Test forward pass with dummy input tensors."""
#         batch_size = 5
#         states = torch.randn(batch_size, self.max_length, self.state_dim)
#         actions = torch.randn(batch_size, self.max_length, self.act_dim)
#         rewards = torch.randn(batch_size, self.max_length)

#         _, pred_actions, _ = self.model.forward(states, actions, rewards)

#         self.assertEqual(pred_actions.shape, (batch_size, 1, self.act_dim))
#         self.assertTrue(torch.is_tensor(pred_actions))

#     def test_get_action(self):
#         """Test get_action method for valid output."""
#         states = torch.randn(self.max_length, self.state_dim)
#         action = self.model.get_action(states, None, None)

#         self.assertEqual(action.shape, (self.act_dim,))
#         self.assertTrue(torch.is_tensor(action))

#     def test_forward_pass_no_nan(self):
#         """Ensure model does not output NaNs or inf values."""
#         batch_size = 10
#         states = torch.randn(batch_size, self.max_length, self.state_dim)
#         actions = torch.randn(batch_size, self.max_length, self.act_dim)
#         rewards = torch.randn(batch_size, self.max_length)

#         _, pred_actions, _ = self.model.forward(states, actions, rewards)

#         self.assertFalse(torch.isnan(pred_actions).any(), "Output contains NaNs")
#         self.assertFalse(torch.isinf(pred_actions).any(), "Output contains Infs")

if __name__ == '__main__':
    # unittest.main()
    pass
