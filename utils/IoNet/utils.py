import torch


def euler_to_quaternion(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of Euler angle time series to quaternions.

    Args:
        euler_angles: A tensor of shape (batch, 3, timeseries) where the
                      channels correspond to roll, pitch, and yaw.

    Returns:
        A tensor of shape (batch, 4, timeseries) representing the quaternions
        (w, x, y, z).
    """
    roll = euler_angles[:, 0, :]
    pitch = euler_angles[:, 1, :]
    yaw = euler_angles[:, 2, :]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # Stack along the channel dimension
    quaternions = torch.stack([qw, qx, qy, qz], dim=1)

    return quaternions


def quaternion_to_euler(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternion time series to Euler angles.

    Args:
        quaternions: A tensor of shape (batch, 4, timeseries) representing
                     the quaternions in (w, x, y, z) format.

    Returns:
        A tensor of shape (batch, 3, timeseries) where the channels
        correspond to roll, pitch, and yaw.
    """
    w, x, y, z = (
        quaternions[:, 0, :],
        quaternions[:, 1, :],
        quaternions[:, 2, :],
        quaternions[:, 3, :],
    )

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Use clamp to avoid numerical errors when sinp is slightly outside [-1, 1]
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Stack along the channel dimension
    euler_angles = torch.stack([roll, pitch, yaw], dim=1)

    return euler_angles


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two batches of quaternions.
    Args:
        q1: (..., 4) [w, x, y, z]
        q2: (..., 4) [w, x, y, z]
    Returns:
        q1 * q2: (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def angular_velocity_to_quaternion(omega: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Integrate angular velocity to find the total relative rotation (quaternion).
    Args:
        omega: (batch, 3, timeseries) - angular velocity (rad/s)
        dt: time step (s)
    Returns:
        q_final: (batch, 4) - Relative rotation quaternion [w, x, y, z]
    """
    batch_size, _, seq_len = omega.shape
    device = omega.device

    # Current accumulated rotation (Identity)
    acc_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(batch_size, 1)

    for i in range(seq_len):
        w = omega[:, :, i]  # (batch, 3)
        theta = torch.norm(w, dim=1) * dt  # (batch,)

        # Avoid division by zero
        theta_clamp = torch.clamp(theta, min=1e-8)

        half_theta = theta_clamp / 2
        sin_half = torch.sin(half_theta)
        cos_half = torch.cos(half_theta)

        # Axis
        axis = w / theta_clamp.unsqueeze(1)  # (batch, 3)

        # Step quaternion
        q_step_xyz = axis * sin_half.unsqueeze(1)
        q_step_w = cos_half.unsqueeze(1)

        # Handle zero rotation case (where theta is very small)
        mask = (theta < 1e-8).float().unsqueeze(1)
        q_step_xyz = q_step_xyz * (1 - mask)
        q_step_w = q_step_w * (1 - mask) + mask * 1.0

        q_step = torch.cat([q_step_w, q_step_xyz], dim=1)  # (batch, 4)

        # Accumulate: q_new = q_old * q_step (local/body frame integration)
        acc_quat = quaternion_multiply(acc_quat, q_step)

        # Normalize to prevent drift
        acc_quat = acc_quat / torch.norm(acc_quat, dim=1, keepdim=True)

    return acc_quat
