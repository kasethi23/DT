import numpy as np
import torch

from decision_transformer.training.trainer import Trainer

class SequenceTrainer(Trainer):

    def train_step(self):
        # Get a batch of training data.
        # This returns tensors for states, actions, rewards, done flags, returns-to-go (rtg), timesteps, and attention mask.
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        
        # Create a copy of the actions to use as the target for the action prediction.
        # The model will try to predict these actions.
        action_target = torch.clone(actions)

        # Forward pass through the model.
        # The model takes in states, actions, rewards, and returns-to-go (excluding the last timestep),
        # along with timesteps and an attention mask. It returns predictions for states, actions, and rewards.
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        # Get the action dimension size (number of possible actions or action features).
        act_dim = action_preds.shape[2]
        
        # Reshape the action predictions to a 2D tensor where each row corresponds to a prediction.
        # We then select only those entries that correspond to valid tokens using the attention mask.
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        # Similarly, reshape the action targets and apply the same mask so we only compute loss on valid positions.
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # Compute the loss between the predicted actions and the target actions.
        # Here, the loss_fn is provided by the trainer (likely a cross-entropy or MSE loss).
        # The function is structured to receive multiple inputs, but we only pass the ones relevant to actions.
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # Zero out the gradients in the optimizer to avoid accumulation from previous steps.
        self.optimizer.zero_grad()
        
        # Backpropagate the loss.
        loss.backward()
        
        # Clip the gradients to avoid exploding gradients. The clip value here is 0.25.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        
        # Update the model parameters using the optimizer.
        self.optimizer.step()

        # Calculate the mean squared error between predicted and target actions for diagnostics,
        # then store it in the diagnostics dictionary for logging purposes.
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()

        # Return the loss value (converted to a Python number) for logging or further use.
        return loss.detach().cpu().item()
