import torch
import torch.nn as nn
from torch import sin, cos, cross, einsum
import numpy as np
from torchdyn.numerics.odeint import odeint


class DifferentiableQuadcopterEnv(nn.Module):
    """
    Differentiable Quadcopter environment using ODE integration (odeint).
    """
    STATE_DIM = 12
    CONTROL_DIM = 4
    G = 9.81
    M = 0.027
    L = 0.0397
    THRUST2WEIGHT_RATIO = 2.25
    J = torch.diag(torch.tensor([1.4e-5, 1.4e-5, 2.17e-5]))
    J_INV = torch.linalg.inv(J)
    KF = 3.16e-10
    KM = 7.94e-12
    GRAVITY = G * M
    HOVER_RPM = np.sqrt(GRAVITY / (4 * KF))
    MAX_RPM = np.sqrt((THRUST2WEIGHT_RATIO * GRAVITY) / (4 * KF))
    DELTA_RPM = MAX_RPM - HOVER_RPM
    MAX_THRUST = (4 * KF * MAX_RPM ** 2)
    MAX_XY_TORQUE = (2 * L * KF * MAX_RPM ** 2) / np.sqrt(2)
    MAX_Z_TORQUE = (2 * KM * MAX_RPM ** 2)


    # The condition for the adaptive controller is the x,y,z position of the target state
    def __init__(self, target_states: torch.Tensor, target_state_cost_coeffs: torch.Tensor = None, 
                 time_res: float = 0.02, horizon: int = 100, solver: str = 'rk4', 
                 reference_losses: torch.Tensor = None, noise: bool = False, noise_magnitude: float = 0.0,
                 return_trajectory: bool = False, adversarial_fn: callable = None, adversarial_active: bool = False, 
                 adversarial_magnitude: float = 0.0):
        super().__init__()
        self.time_res = time_res
        self.horizon = horizon
        self.solver = solver
        self.target_state = target_states  # shape [N, 12]
        self.task_params = target_states[:, :3]  # shape [N, 3]
        # target state coeffs should be 2 for x,y,z and 1 for rest
        self.target_state_cost_coeffs = torch.tensor([1,1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ,0.2, 0.2])
        #self.target_state_cost_coeffs = target_state_cost_coeffs.unsqueeze(0) if target_state_cost_coeffs is not None else torch.ones(1, 12)
        self.control_history = []
        self.state_history = []
        
        # Noise parameters
        self.noise = noise
        self.noise_magnitude = noise_magnitude
        
        # Adversarial parameters
        self.adversarial_fn = adversarial_fn
        self.adversarial_active = adversarial_active
        self.adversarial_magnitude = adversarial_magnitude

        # Trajectories for imitation learning 
        self.return_trajectory = return_trajectory
        self.trajectory_states = None
        self.trajectory_actions = None
        
        # Store reference losses for normalization
        self.reference_losses = reference_losses

        # Store wind force for each sample
        self.wind_force = None
        self.auxiliary_controller_output = None

        #no fly boxes - list of dicts with 'min', 'max', 'penalty' keys
        self.no_fly_boxes = []
        
        # Statistics for tracking box violations
        self.total_steps = 0
        self.violation_steps = 0
        if reference_losses is not None:
            coeffs = self.target_state_cost_coeffs.to(target_states.device)  # (1, 12)
            hover_state = torch.zeros_like(target_states)                    # (N, 12)

            # same weighted-L2 formula used in compute_costs
            weighted_diff = (target_states - hover_state) * coeffs           # (N, 12)
            hover_cost    = torch.sum(weighted_diff ** 2, dim=-1)            # (N,)
            self.normalization_factors = hover_cost 

            #transform references to positive

    def reset(self):
        self.control_history = []
        self.state_history = []
        self.current_perturbation = None
        # Adversarial noise stats (per forward)
        self._adv_norm_sum = 0.0
        self._adv_norm_count = 0
        self._last_adv_mean_norm = None
        self.control_history = []
        self.current_perturbation = None
        self._adv_norm_sum = 0.0
        self._adv_norm_count = 0
        self._last_adv_mean_norm = None
        self.trajectory_states = None
        self.trajectory_actions = None

    # Convenience setters for training modes
    def set_clean(self):
        """Disable both random and adversarial noise."""
        self.noise = False
        self.adversarial_active = False
        self.adversarial_fn = None
        return self

    def set_random_noise(self, magnitude: float = None):
        """Enable random noise; disable adversarial noise.

        Args:
            magnitude: Optional override for noise magnitude.
        """
        self.adversarial_active = False
        self.adversarial_fn = None
        self.noise = True
        if magnitude is not None:
            self.noise_magnitude = magnitude
        return self

    def set_adversarial_noise(self, adversarial_fn, magnitude: float = None):
        """Enable adversarial noise; disable random noise.

        Args:
            adversarial_fn: Callable taking concat([target, state]) and returning noise (12,)
            magnitude: Optional override for adversarial noise magnitude.
        """
        self.noise = False
        self.adversarial_active = True
        self.adversarial_fn = adversarial_fn
        if magnitude is not None:
            self.adversarial_magnitude = magnitude
        return self

    def add_no_fly_box(self, min_corner, max_corner, penalty=1.0, log_penalty_coeff=0.1):
        """
        Add a no-fly box that imposes additional penalty when quadcopter enters it.
        
        Args:
            min_corner: torch.Tensor of shape [3] - minimum x,y,z coordinates of box
            max_corner: torch.Tensor of shape [3] - maximum x,y,z coordinates of box  
            penalty: float - penalty multiplier when inside the box
        """
        box = {
            'min': min_corner,
            'max': max_corner, 
            'penalty': penalty,
            'log_penalty_coeff': log_penalty_coeff
        }
        self.no_fly_boxes.append(box)
        return self

    def clear_no_fly_boxes(self):
        """Remove all no-fly boxes."""
        self.no_fly_boxes = []
        return self
    
    def reset_violation_stats(self):
        """Reset violation statistics."""
        self.total_steps = 0
        self.violation_steps = 0
        return self
    
    def get_violation_ratio(self):
        """Get the ratio of steps that were inside boxes."""
        if self.total_steps == 0:
            return 0.0
        return self.violation_steps / self.total_steps
    @staticmethod
    def point_to_aabb_distance(point, box_min, box_max, return_surface=True):
        """
        Minimal distance from a point to an axis aligned box, plus the closest point.

        Args:
            point:    tensor [..., 3]
            box_min:  tensor [3]
            box_max:  tensor [3]
            return_surface: if True and the point is inside, return the closest point on the *surface*.
                            if False and inside, the closest point equals the point and distance is 0.

        Returns:
            dist:         tensor [...]  minimal distance
            closest:      tensor [..., 3] closest point on the box (surface or boundary)
            inside_mask:  tensor [...]  True if inside (including on boundary)
        """

        p  = point
        # Expand box bounds to match point shape [..., 3]
        lo = box_min.reshape(*([1] * (p.dim() - 1)), 3).expand_as(p)
        hi = box_max.reshape(*([1] * (p.dim() - 1)), 3).expand_as(p)

        # Check inside on all axes
        inside_mask = ((p >= lo) & (p <= hi)).all(dim=-1)

        # Outside case: clamp to box to get closest point
        clamped = torch.minimum(torch.maximum(p, lo), hi)
        vec_outside = p - clamped
        dist_outside = torch.linalg.norm(vec_outside, dim=-1)  # zero if inside or on surface

        if not return_surface:
            # If we accept interior as part of the box, the closest point for inside is the point itself.
            closest = torch.where(inside_mask[..., None], p, clamped)
            dist = torch.where(inside_mask, torch.zeros_like(dist_outside), dist_outside)
            return dist, closest, inside_mask

        # For inside points, project to the nearest face
        # distances to faces along each axis
        d_to_lo = (p - lo)            # >= 0 inside
        d_to_hi = (hi - p)            # >= 0 inside
        d_face  = torch.minimum(d_to_lo, d_to_hi)  # [..., 3]
        # axis of nearest face
        min_axis = torch.argmin(d_face, dim=-1)    # [... ]

        # Build closest surface point for inside
        # start from the original point, then set the chosen axis to the nearer face value
        closest_inside = p.clone()
        # gather whether nearer face is lo or hi along that axis
        choose_lo = (d_to_lo.gather(-1, min_axis.unsqueeze(-1)).squeeze(-1)
                    <= d_to_hi.gather(-1, min_axis.unsqueeze(-1)).squeeze(-1))

        # Choose face values per axis and assign along the nearest axis
        gather_axis = min_axis.unsqueeze(-1)
        face_vals = torch.where(choose_lo[..., None], lo, hi)  # [..., 3]
        chosen_face_coord = face_vals.gather(-1, gather_axis)
        closest_inside.scatter_(-1, gather_axis, chosen_face_coord)

        dist_inside = d_face.gather(-1, gather_axis).squeeze(-1)

        # Combine inside vs outside
        closest = torch.where(inside_mask[..., None], closest_inside, clamped)
        dist = torch.where(inside_mask, dist_inside, dist_outside)

        return dist, closest, inside_mask


    def compute_box_penalties(self, positions):
        """
        Compute penalties for positions inside no-fly boxes and log loss for getting close to boxes.
        
        Args:
            positions: torch.Tensor of shape [B, T, 3] - x,y,z positions over time
            
        Returns:
            torch.Tensor of shape [B, T] - penalty values (0 if outside all boxes)
        """
        if not self.no_fly_boxes:
            return torch.zeros(positions.shape[0], positions.shape[1], device=positions.device)
        
        batch_size, time_steps, _ = positions.shape
        total_penalty = torch.zeros(batch_size, time_steps, device=positions.device)
        any_violation = torch.zeros(batch_size, time_steps, dtype=torch.bool, device=positions.device)
        
        for box in self.no_fly_boxes:
            min_corner = box['min'].to(positions.device)
            max_corner = box['max'].to(positions.device)
            penalty = box['penalty']
            log_penalty_coeff = box['log_penalty_coeff']
            
            # # Check if position is inside box: min <= pos <= max for all dimensions
            # inside_x = (positions[..., 0] >= min_corner[0]) & (positions[..., 0] <= max_corner[0])
            # inside_y = (positions[..., 1] >= min_corner[1]) & (positions[..., 1] <= max_corner[1])
            # inside_z = (positions[..., 2] >= min_corner[2]) & (positions[..., 2] <= max_corner[2])
            
            # inside_box = inside_x & inside_y & inside_z
            # total_penalty += inside_box.float() * penalty
            # any_violation = any_violation | inside_box
            
            # Add log loss for getting close to the box (even when outside)
            # Compute distance to box boundaries
            dist, closest, inside_mask = self.point_to_aabb_distance(positions, min_corner, max_corner)
            #work in vactorized way
            total_penalty += inside_mask.float() * penalty #inside box penalty
            total_penalty += (~inside_mask).float() * torch.exp(-dist/log_penalty_coeff) * penalty #outside box penalty
            any_violation = any_violation | inside_mask
            
        # Update violation statistics
        self.total_steps += batch_size * time_steps
        self.violation_steps += any_violation.sum().item()
        
        return total_penalty

    #function to update the boxes penalty
    def update_boxes_penalty(self,new_penalty):
        for box in self.no_fly_boxes:
            box['penalty'] = new_penalty
        return self
    # Removed fixed perturbation generator; noise is injected in dynamics
    
    def forward(self, controller_fn, init_state, fixed, adv=False, debug=False, num_noise_samples=1):
        """
        Forward pass through the environment.
        If noise is enabled and num_noise_samples > 1, runs multiple times with different noise
        and returns the maximum loss.
        """
        self.controller_fn = controller_fn
        self.fixed = fixed
        self.adv = adv
        #set random wind force for each sample in this rollout
        # Default: no wind unless explicitly set (shape: [B, 3])
        self.wind_force = torch.zeros(init_state.shape[0], 3, device=init_state.device)

        # Rollout and compute costs 
        def rollout_and_cost(store_trajectory=False):
            self.reset()
            t_span = torch.linspace(0, self.time_res * self.horizon, self.horizon + 1, device=init_state.device)
            states = odeint(self.dynamics, init_state, t_span, solver=self.solver)[1]  # [T+1, B, D]
            states = states.permute(1, 0, 2)  # [B, T+1, D]
            controls = torch.stack(self.control_history).permute(1, 0, 2)  # [B, T, 4]
            if store_trajectory and self.return_trajectory:
                self.trajectory_states = states
                self.trajectory_actions = controls
            return self.compute_costs(states[:, 1:], controls)  # [B, 1]

        # If Random noise enabled and multiple samples requested: run multiple rollouts and take worst-case
        if self.noise and num_noise_samples > 1:
            costs_list = []
            for i in range(num_noise_samples):
                #set random wind force for each sample in this rollout, make it only in z axis
                self.wind_force = torch.randn(init_state.shape[0], 3, device=init_state.device) * self.noise_magnitude
                # self.wind_force = torch.zeros(init_state.shape[0], 3, device=init_state.device)
                # self.wind_force[:, 2] = torch.randn(init_state.shape[0], device=init_state.device) * self.noise_magnitude
                costs = rollout_and_cost(store_trajectory=(i == 0))  # store only first
                costs_list.append(costs.squeeze(-1))  # [B]
            stacked = torch.stack(costs_list, dim=0)  # [S, B]
            worst = torch.max(stacked, dim=0)[0]  # [B]
            return worst.mean()
        else:
            costs = rollout_and_cost(store_trajectory=True)
            if self._adv_norm_count > 0:
                mean_adv_norm = self._adv_norm_sum / max(1, self._adv_norm_count)
                self._last_adv_mean_norm = float(mean_adv_norm)
            return costs.mean()

    def dynamics(self, t, state):
        # Build controller input, optionally with adversarial noise added to the state part
        if self.fixed:
            state_for_control = state
            controller_input = state_for_control
        else: # adaptive controller, concat state and task_params
            # Optionally add adversarial noise that depends on [target, state]
            # if self.adversarial_active and (self.adversarial_fn is not None):
            #     adv_inp = torch.cat([self.task_params.to(state.device), state], dim=-1)
            #     adv_noise = self.adversarial_fn(adv_inp)
            #     # Clip noise to adversarial_magnitude per sample
            #     noise_norm = torch.norm(adv_noise, dim=-1, keepdim=True) + 1e-12
            #     scale = torch.clamp(self.adversarial_magnitude / noise_norm, max=1.0)
            #     adv_noise = adv_noise * scale
            #     # Accumulate mean norm of adversarial noise
            #     with torch.no_grad():
            #         try:
            #             mean_norm = torch.norm(adv_noise, dim=-1).mean().item()
            #             self._adv_norm_sum += float(mean_norm)
            #             self._adv_norm_count += 1
            #         except Exception:
            #             pass
            #     state_for_control = state + adv_noise
            # elif self.noise:
            #     # Inject random noise per step, clipped by noise_magnitude
            #     rnd = torch.randn_like(state)
            #     rnd_norm = torch.norm(rnd, dim=-1, keepdim=True) + 1e-12
            #     scale = (self.noise_magnitude * torch.rand_like(rnd_norm)) / rnd_norm
            #     rnd = rnd * torch.clamp(scale, max=1.0)
            #     state_for_control = state + rnd
            # else:
            state_for_control = state
            controller_input = torch.cat([state_for_control, self.task_params.to(state.device)], dim=-1)
            # history_length = min(len(self.state_history), 3) ####
            # batch_size = state_for_control.shape[0]
            # if len(self.control_history) > 0:
            #     stacked_states = torch.stack(self.state_history[-history_length:], dim=0)
            #     stacked_actions = torch.stack(self.control_history[-history_length:], dim=0)
            #     if history_length < 3: ####
            #         #oad with 0
            #         stacked_states = torch.cat([torch.zeros(3-history_length, batch_size, 12).to(state.device), stacked_states], dim=0).to(state.device)
            #         stacked_actions = torch.cat([torch.zeros(3-history_length, batch_size, 4).to(state.device), stacked_actions], dim=0).to(state.device)
            # else:
            #     stacked_states = torch.zeros(3, batch_size, 12).to(state.device)
            #     stacked_actions = torch.zeros(3, batch_size, 4).to(state.device)
            # #print shape of stacked_states and stacked_actions
            # history_input = torch.cat([stacked_states, stacked_actions], dim=-1).to(state.device)
            # #shape to [B, 5*12 + 5*4]
            # history_input = history_input.reshape(batch_size, -1)
            # controller_input = torch.cat([history_input,controller_input], dim=-1)
            #concat to state [1,1,1]
        control = self.controller_fn(controller_input) 
        #control is between -1 and 1 so we want rpm between 0 and MAX_RPM ant init on hover rpm
        rpm = self.HOVER_RPM + self.DELTA_RPM * control
        #rpm = control # makes it between 
        self.state_history.append(state)
        self.control_history.append(control)   
        #print rpm if no nan

        pos, rpy, vel, rpy_rates = state[..., 0:3], state[..., 3:6], state[..., 6:9], state[..., 9:12]
        forces = rpm ** 2 * self.KF
        thrust_z = torch.sum(forces, dim=-1)
        thrust = torch.zeros_like(pos)
        thrust[..., 2] = thrust_z
        rotation = euler_matrix(rpy[..., 0], rpy[..., 1], rpy[..., 2]).to(pos.device)
        thrust_world = einsum('...ij, ...j-> ...i', rotation, thrust)
        force_world = thrust_world - torch.tensor([0, 0, self.GRAVITY], device=pos.device)
        #add wind force to force_world
        force_world += self.wind_force

        z_torques = rpm ** 2 * self.KM
        z_torque = (-z_torques[..., 0] + z_torques[..., 1] - z_torques[..., 2] + z_torques[..., 3])
        x_torque = (forces[..., 0] + forces[..., 1] - forces[..., 2] - forces[..., 3]) * (self.L / np.sqrt(2))
        y_torque = (-forces[..., 0] + forces[..., 1] + forces[..., 2] - forces[..., 3]) * (self.L / np.sqrt(2))
        torques = torch.cat([x_torque[..., None], y_torque[..., None], z_torque[..., None]], -1)
        torques -= cross(rpy_rates, einsum('ij,...i->...j', self.J.to(rpy_rates.device), rpy_rates))
        rpy_rates_deriv = einsum('ij,...i->...j', self.J_INV.to(rpy_rates.device), torques)
        accs = force_world / self.M

        dx = torch.cat([vel, rpy_rates, accs, rpy_rates_deriv], dim=-1)
        return dx

    def compute_costs(self, states, controls):
        target = self.target_state.to(states.device)
        coeffs = self.target_state_cost_coeffs.to(states.device)
        if len(target.shape) == 1:
            target = target.unsqueeze(0).unsqueeze(1)
        else:
            target = target.unsqueeze(1)
        weighted_diff = (states - target) * coeffs.unsqueeze(0)
        cost = torch.sum(weighted_diff ** 2, dim=-1, keepdim=True)

        # Add box penalties
        positions = states[..., :3]  # Extract x,y,z positions [B, T, 3]
        box_penalties = self.compute_box_penalties(positions)  # [B, T]
        box_penalties = box_penalties.unsqueeze(-1)  # [B, T, 1]
        cost = cost + box_penalties

        # first we will compute the mean over the time dimension and then we will normalize
        cost = torch.mean(cost, dim=1)
        # Apply normalization if reference losses are provided
        if self.reference_losses is not None:
            # Normalize each target state's cost
            # 0 = best reference loss, 1 = distance from origin
            reference_losses = self.reference_losses.to(states.device)
            normalization_factors = self.normalization_factors.to(states.device)
            # For each batch element, normalize its cost
            normalized_costs = []
            for i in range(cost.shape[0]):  # Loop over batch dimension
                target_cost = cost[i]  # [T, 1]
                ref_loss = reference_losses[i]  # scalar
                norm_factor = normalization_factors[i]  # scalar
                 # Normalize: (cost - ref_loss) / norm_factor
                # This gives 0 when cost = ref_loss, and 1 when cost = ref_loss + norm_factor
                normalized_cost = (target_cost - ref_loss) / (norm_factor - ref_loss)
                normalized_costs.append(normalized_cost)
            
            return torch.stack(normalized_costs, dim=0)
        else:
            return cost

    def get_trajectory(self):
        """
        Returns the stored trajectory (states and actions) if return_trajectory was True.
        Re-evaluates the controller on the states to get the correct number of actions.
        Returns:
            tuple: (states, actions) where states is [B, T+1, 12] and actions is [B, T, 4]
        """
        if self.trajectory_states is None:
            raise ValueError("No trajectory stored. Make sure to call forward() with return_trajectory=True first.")
        
        states = self.trajectory_states  # [B, T+1, 12]
        batch_size, time_steps, state_dim = states.shape
        
        # Re-evaluate controller on the states to get correct number of actions
        actions = []
        for t in range(time_steps - 1):  # Exclude last state (T+1 -> T actions)
            state = states[:, t, :]  # [B, 12]
            
            # Build controller input the same way as in dynamics
            if self.fixed:
                controller_input = state
            else:  # adaptive controller
                controller_input = torch.cat([state, self.task_params.to(state.device)], dim=-1)
            
            # Get control action
            with torch.no_grad():
                control = self.controller_fn(controller_input)  # [B, 4]
            actions.append(control)
        
        # Stack actions: [T, B, 4] -> [B, T, 4]
        actions = torch.stack(actions, dim=1)
        
        return states, actions

def format_sample_info(goal):
    pos = goal[:3].tolist()
    rest = goal[3:].tolist()
    pos_str = '_'.join([f'{v:.2f}' for v in pos])
    rest_str = '_'.join([f'{v:.2f}' for v in rest])
    return f'goal={pos_str}_rest={rest_str}'

def euler_matrix(ai, aj, ak, repetition=True):
    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    i, j, k = 0, 1, 2
    M = torch.cat(3 * [torch.cat(3 * [torch.zeros(ai.shape)[..., None, None]], -1)], -2)
    if repetition:
        M[..., i, i] = cj
        M[..., i, j] = sj * si
        M[..., i, k] = sj * ci
        M[..., j, i] = sj * sk
        M[..., j, j] = -cj * ss + cc
        M[..., j, k] = -cj * cs - sc
        M[..., k, i] = -sj * ck
        M[..., k, j] = cj * sc + cs
        M[..., k, k] = cj * cc - ss
    else:
        M[..., i, i] = cj * ck
        M[..., i, j] = sj * sc - cs
        M[..., i, k] = sj * cc + ss
        M[..., j, i] = cj * sk
        M[..., j, j] = sj * ss + cc
        M[..., j, k] = sj * cs - sc
        M[..., k, i] = -sj
        M[..., k, j] = cj * si
        M[..., k, k] = cj * ci
    return M
