import numpy as np
import torch

from decision_transformer.training.trainer import Trainer

#  dont need this we need for behaviour cloning 

class ActTrainer(Trainer):

    def train_step(self):
        # Retrieve a batch of training data.
        # The batch includes states, actions, rewards, dones flags, returns-to-go (rtg),
        # an unused value (here represented by _), and an attention mask.
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        
        # Create target copies for states, actions, and rewards.
        # These will serve as the ground truth for our predictions.
        state_target = torch.clone(states)
        action_target = torch.clone(actions)
        reward_target = torch.clone(rewards)

        # Forward pass through the model.
        # The model takes in states, actions, and rewards.
        # It also uses an attention mask and a target return (here, the first element of rtg along axis 1).
        # The model returns predictions for states, actions, and rewards.
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:, 0],
        )

        # Get the dimensionality of the action predictions.
        act_dim = action_preds.shape[2]
        
        # Reshape the predicted actions to a 2D tensor (each row is a prediction for one action token).
        action_preds = action_preds.reshape(-1, act_dim)
        
        # For the target actions, we use only the final action in the sequence (i.e., the last timestep).
        # We reshape this target similarly to match the predictions.
        action_target = action_target[:, -1].reshape(-1, act_dim)

        # Compute the loss using the provided loss function.
        # The loss function compares the predicted states, actions, and rewards with their corresponding targets.
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        
        # Zero out any existing gradients to prevent accumulation from previous training steps.
        self.optimizer.zero_grad()
        
        # Backpropagate the loss to compute gradients.
        loss.backward()
        
        # Update the model parameters using the optimizer.
        self.optimizer.step()

        # Return the loss value as a Python float for logging purposes.
        return loss.detach().cpu().item()
