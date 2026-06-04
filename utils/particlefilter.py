import numpy as np
import torch


class ParticleFilter:
    def __init__(self, num_particles, device="cpu"):
        self.num_particles = num_particles
        self.device = torch.device(device)
        self.particles = torch.zeros(
            (num_particles, 3), device=self.device
        )  # X, Y, angle
        self.weights = torch.ones(num_particles, device=self.device) / num_particles

    def initialize(self, init_state):
        self.particles = init_state + torch.randn_like(self.particles)

    def predict(self, process_model, process_noise):
        self.particles = process_model(
            self.particles
        ) + process_noise * torch.randn_like(self.particles)

    def update(self, observation, observation_model, observation_noise):
        predicted_observation = observation_model(self.particles)
        errors = observation - predicted_observation
        self.weights *= torch.exp(
            -0.5 * torch.sum(errors**2, dim=1) / observation_noise**2
        )
        self.weights /= torch.sum(self.weights)

    def resample(self):
        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices]
        self.weights.fill_(1.0 / self.num_particles)

    def estimate(self):
        return torch.sum(self.particles * self.weights.unsqueeze(1), dim=0)
