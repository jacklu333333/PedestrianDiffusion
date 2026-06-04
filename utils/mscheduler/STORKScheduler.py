import math

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class STORKScheduler(SchedulerMixin, ConfigMixin):
    """
    STORK: Stabilized Taylor Orthogonal Runge-Kutta Scheduler.
    Implements STORK-2 algorithm for fast, stiff-aware sampling.
    Reference: https://arxiv.org/abs/2505.24210
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        s_stages: int = 5,  # Stiffness parameter (default 5 is good for 10-20 steps)
    ):
        self.timesteps = None
        self.history_v = []  # Stores past model outputs [v_{t}, v_{t-1}, v_{t-2}]
        self.history_t = []  # Stores past timesteps

        # Pre-compute Chebyshev coefficients for stability
        self.coeffs = self._compute_rock_coeffs(s_stages)

    def set_timesteps(self, num_inference_steps, device="cuda"):
        """Sets the discrete timesteps for the diffusion chain."""
        self.num_inference_steps = num_inference_steps

        # Standard Flow Matching Time Schedule (Linear t=1 to t=0)
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1)
        self.timesteps = torch.from_numpy(timesteps).to(device)

        # Clear history for new inference
        self.history_v = []
        self.history_t = []

    def _compute_rock_coeffs(self, s):
        """
        Computes stability coefficients (mu, nu, kappa) based on Chebyshev polynomials.
        These ensure the solver doesn't diverge for stiff equations.
        """
        # Standard damping parameter for ROCK2 methods
        eta = 0.05

        # Recurrence relation for Chebyshev polynomials T_s(w)
        w0 = 1 + eta / (s**2)

        # Calculate Chebyshev values at w0
        T = [1.0, w0]
        for i in range(2, s + 1):
            T.append(2 * w0 * T[-1] - T[-2])

        # Derivative T' values
        T_prime = [0.0, 1.0]
        for i in range(2, s + 1):
            T_prime.append(2 * T[-1] + 2 * w0 * T_prime[-1] - T_prime[-2])

        # Compute weight mu_1
        mu_1 = w0 / T_prime[s] * T_prime[s - 1]

        # Compute recurrence coefficients for internal stages
        mus, nus, kappas = [], [], []
        for i in range(2, s + 1):
            neg_mu = 2 * w0 * T[i - 1] / T[i]
            neg_nu = -T[i - 2] / T[i]
            # Standard normalization for RKC/ROCK
            mus.append(neg_mu)
            nus.append(neg_nu)
            kappas.append(1.0 - neg_mu - neg_nu)

        return {"mu1": mu_1, "mus": mus, "nus": nus, "kappas": kappas}

    def _taylor_approx(self, t_query, current_t):
        """
        Approximates velocity v(t_query) using 2nd-order Taylor expansion
        from cached history.
        """
        if len(self.history_v) < 3:
            # Fallback if not enough history (should not happen in main loop)
            return self.history_v[-1]

        v0, v1, v2 = self.history_v[-1], self.history_v[-2], self.history_v[-3]
        t0, t1, t2 = self.history_t[-1], self.history_t[-2], self.history_t[-3]

        dt1 = t0 - t1
        dt2 = t1 - t2

        # Finite difference estimates for derivatives
        dv1 = (v0 - v1) / dt1
        dv2 = (v1 - v2) / dt2
        ddv = (dv1 - dv2) / (0.5 * (dt1 + dt2))  # approx 2nd derivative

        # Taylor expansion: v(t) = v(t0) + v'(t0)dt + 0.5*v''(t0)dt^2
        dt_query = t_query - t0
        v_pred = v0 + dv1 * dt_query + 0.5 * ddv * (dt_query**2)

        return v_pred

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        One step of the ODE solver.
        """
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = self.timesteps[step_index + 1]
        dt = prev_timestep - timestep  # Negative value (time goes 1 -> 0)

        # Update history
        self.history_v.append(model_output)
        self.history_t.append(timestep)
        if len(self.history_v) > 3:
            self.history_v.pop(0)
            self.history_t.pop(0)

        # --- 1. Start-Up Phase (First 3 steps use simpler solvers) ---
        if step_index < 3:
            # Use Euler-like update for the very first steps to build history
            if step_index == 0:
                # Euler
                prev_sample = sample + dt * model_output
            else:
                # 2-step Adams-Bashforth (using available history)
                v_curr = self.history_v[-1]
                v_prev = self.history_v[-2]
                prev_sample = sample + dt * (1.5 * v_curr - 0.5 * v_prev)

            return SchedulerOutput(prev_sample=prev_sample)

        # --- 2. Main STORK-2 Phase ---
        # Initialize internal stage variables
        # We transform to Y variables for stability (ROCK structure)
        mu1 = self.coeffs["mu1"]
        mus = self.coeffs["mus"]
        nus = self.coeffs["nus"]
        kappas = self.coeffs["kappas"]

        # Stage 1 (Euler predictor with damping)
        # Note: In Flow Matching, dx/dt = v.
        # Stabilized methods usually solve dy/dt = F(y). Here F(y) = v_theta(y, t)

        # g0 = sample
        # g1 = g0 + (mu1 * dt) * v_curr
        g_prev2 = sample
        g_prev1 = sample + (mu1 * dt) * model_output

        # Current time t inside the step
        current_t = timestep.item()

        # Internal Stages j=2 to s
        for j, (mu, nu, kappa) in enumerate(zip(mus, nus, kappas)):
            # "Virtual NFE": We need v at an intermediate time.
            # STORK trick: Use Taylor expansion instead of calling the model

            # Estimate internal time for this stage (heuristic for stability)
            # Simplification: Assume linear distribution of stages or check Chebyshev roots
            # For STORK implementation, we often approximate the velocity
            # at the predicted position g_prev1 using the Taylor expansion relative to t.

            # The "Virtual Velocity" depends on where we are in time
            # For ROCK2, internal stages are effectively at different 't'.
            # We use the Taylor expansion to get v_approx at (t + c_j * dt)
            # NOTE: Exact c_j depends on stage, but for STORK-2 we can often
            # use the updated g directly if the field is state-dependent.
            # Since Taylor approximates v(t, x(t)), we project v forward.

            # Compute v_virtual using Taylor Expansion based on history
            # We assume the "time" for the virtual step progresses linearly
            virtual_t = current_t + (j + 1) / self.config.s_stages * dt.item()
            v_virtual = self._taylor_approx(virtual_t, current_t)

            # ROCK/RKC Recurrence
            # Y_j = mu_j * Y_{j-1} + nu_j * Y_{j-2} + kappa_j * Y_0 +...
            # The update rule combines stability terms with the gradient (velocity)

            term_v = dt * v_virtual  # The driving force from velocity

            # Update g_curr (Y_j)
            # Formula: Y_j = mu * Y_{j-1} + nu * Y_{j-2} + kappa * Y_0 - (damping terms)
            # Simplified STORK update:
            g_curr = (
                mu * g_prev1
                + nu * g_prev2
                + kappa * sample
                + term_v * (mu + nu + kappa)
            )

            # Shift for next stage
            g_prev2 = g_prev1
            g_prev1 = g_curr

        prev_sample = g_prev1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)
