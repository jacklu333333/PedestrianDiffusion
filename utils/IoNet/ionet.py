import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Convert from the original Keras implementation of IoNet to PyTorch.
Original implementation:
https://github.com/jpsml/6-DOF-Inertial-Odometry
"""


# Helper function for quaternion operations, assuming similar logic to tfquaternion
# This is a simplified version. For a full implementation, a dedicated library would be better.
def quat_conjugate(q):
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


# Loss functions in PyTorch
def quaternion_phi_4_error(y_true, y_pred):
    # Normalize y_pred along the last dimension before dot product
    y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
    # Ensure y_true is also normalized if it's not already
    y_true_normalized = F.normalize(y_true, p=2, dim=-1)

    # Batch dot product
    dot_product = torch.sum(y_true_normalized * y_pred_normalized, dim=-1)

    return 1 - torch.abs(dot_product)


def quat_mult_error(y_true, y_pred):
    # Assuming y_true and y_pred are tensors of shape [batch_size, 4]

    # Normalize the predicted quaternion
    q_pred_normalized = F.normalize(y_pred, p=2, dim=-1)

    # Get the conjugate of the true quaternion
    q_true_conjugate = quat_conjugate(F.normalize(y_true, p=2, dim=-1))

    # Multiply predicted quaternion by the conjugate of the true quaternion
    q_prod = quat_multiply(q_pred_normalized, q_true_conjugate)

    # The error is based on the vector part of the resulting quaternion
    # The real part (w) of q_prod should be close to 1 for perfect alignment, and the vector part (x, y, z) close to 0.
    # The error is defined as 2 * |v| where q_prod = [w, v]
    vec_part = q_prod[:, 1:]

    # The Keras implementation takes the absolute value of 2*vec_part.
    return torch.abs(2.0 * vec_part)


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return torch.mean(quat_mult_error(y_true, y_pred))


class IoNet(nn.Module):
    def __init__(self, window_size=200):
        super(IoNet, self).__init__()
        self.window_size = window_size

        # Branch for first input (x1)
        self.convA1 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=11)
        self.convA2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11)
        self.poolA = nn.MaxPool1d(kernel_size=3)

        # Branch for second input (x2)
        self.convB1 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=11)
        self.convB2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11)
        self.poolB = nn.MaxPool1d(kernel_size=3)

        # Note: The output length of the pooling layers needs to be calculated
        # For conv1: L_out = 200 - 11 + 1 = 190
        # For conv2: L_out = 190 - 11 + 1 = 180
        # For pool:  L_out = floor((180 - 3)/3 + 1) = floor(177/3 + 1) = 59 + 1 = 60
        lstm_input_size = 256  # 128 channels from each branch

        # Shared LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(0.25)
        self.lstm2 = nn.LSTM(
            input_size=256, hidden_size=128, batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.25)

        # Output layers
        self.y1_pred = nn.Linear(
            in_features=256, out_features=3
        )  # Output for 3D vector
        self.y2_pred = nn.Linear(
            in_features=256, out_features=4
        )  # Output for quaternion

    def forward(self, x1, x2):
        # Keras Conv1D with 'channels_last' default is equivalent to PyTorch Conv1D
        # if we permute the input dimensions from (batch, seq_len, channels) to (batch, channels, seq_len).
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        # Process first branch
        outA = F.relu(self.convA1(x1))
        outA = F.relu(self.convA2(outA))
        outA = self.poolA(outA)

        # Process second branch
        outB = F.relu(self.convB1(x2))
        outB = F.relu(self.convB2(outB))
        outB = self.poolB(outB)

        # Before concatenating, permute back to (batch, seq_len, features) for LSTM
        outA = outA.permute(0, 2, 1)
        outB = outB.permute(0, 2, 1)

        # Concatenate features
        AB = torch.cat([outA, outB], dim=2)

        # LSTM layers
        lstm_out1, _ = self.lstm1(AB)
        drop1 = self.dropout1(lstm_out1)

        # For the second LSTM, we take the full sequence from the first.
        lstm_out2, _ = self.lstm2(drop1)
        drop2 = self.dropout2(lstm_out2)

        # The Keras model takes the output from the last timestep of the last LSTM layer.
        # With batch_first=True, the output is (batch, seq_len, num_directions * hidden_size).
        # We take the output from the last time step.
        last_timestep_output = drop2[:, -1, :]

        # Dense layers
        y1_pred = self.y1_pred(last_timestep_output)
        y2_pred = self.y2_pred(last_timestep_output)

        return y1_pred, y2_pred


class CustomMultiLoss(nn.Module):
    def __init__(self, nb_outputs=2):
        super(CustomMultiLoss, self).__init__()
        self.nb_outputs = nb_outputs
        self.log_vars = nn.Parameter(torch.zeros(nb_outputs))

    def forward(self, ys_pred, ys_true):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs

        # Loss for the first output (3D vector) - Mean Absolute Error
        loss1 = (
            torch.exp(-self.log_vars[0]) * F.l1_loss(ys_pred[0], ys_true[0])
            + self.log_vars[0]
        )

        # Loss for the second output (quaternion) - Quaternion Mean Multiplicative Error
        loss2 = (
            torch.exp(-self.log_vars[1])
            * quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1])
            + self.log_vars[1]
        )

        return torch.mean(loss1 + loss2)


if __name__ == "__main__":
    # How to use the model and loss

    # Create the model
    pred_model = IoNet(window_size=200)
    print("Model Architecture:")
    print(pred_model)

    # Create the custom loss
    training_loss_function = CustomMultiLoss(nb_outputs=2)

    # Example of a training step
    optimizer = torch.optim.Adam(
        list(pred_model.parameters()) + list(training_loss_function.parameters()),
        lr=0.0001,
    )

    # Dummy data for demonstration
    batch_size = 32
    window_size = 200
    x1_dummy = torch.randn(batch_size, window_size, 3)
    x2_dummy = torch.randn(batch_size, window_size, 3)
    y1_true_dummy = torch.randn(batch_size, 3)
    y2_true_dummy = torch.randn(batch_size, 4)
    # Normalize dummy true quaternions
    y2_true_dummy = F.normalize(y2_true_dummy, p=2, dim=-1)

    # Forward pass
    pred_model.train()
    optimizer.zero_grad()
    y1_pred, y2_pred = pred_model(x1_dummy, x2_dummy)

    # Calculate loss
    loss = training_loss_function([y1_pred, y2_pred], [y1_true_dummy, y2_true_dummy])
    print(f"\nCalculated Loss: {loss.item()}")

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print("\nModel and loss function are ready for training in PyTorch.")
    print(
        f"input shape: {x1_dummy.shape}, output shapes: {y1_pred.shape}, {y2_pred.shape}"
    )
