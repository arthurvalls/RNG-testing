import numpy as np
import hashlib
import random
import time
import os
import psutil


class FortunaRNG:
    def __init__(self, seed=None):
        cpu_utilization = psutil.cpu_percent(interval=1)
        # seed_entropy = int.from_bytes(os.urandom(4), byteorder='big')
        self.seed = cpu_utilization
        self.state = self.seed

    def random(self):
        # Improved LCG parameters
        a = 22695477
        c = 1
        m = 2**32
        self.state = (a * self.state + c) % m
        # Use integer division to avoid bias
        return self.state // (m - 1)


# Define Rosenbrock function
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# PSO algorithm
def pso(cost_func, bounds, num_particles, max_iter, rng):
    dimension = len(bounds)
    # Initialize particles
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, dimension))
    # Initialize velocities
    velocities = np.zeros((num_particles, dimension))
    # Initialize personal best positions
    personal_best = particles.copy()
    # Initialize personal best costs
    personal_best_cost = np.array([cost_func(p) for p in particles])
    # Initialize global best position
    global_best_idx = np.argmin(personal_best_cost)
    global_best = personal_best[global_best_idx]
    # Initialize global best cost
    global_best_cost = personal_best_cost[global_best_idx]

    # History of best solutions and costs
    best_solutions = [global_best]
    best_costs = [global_best_cost]

    for _ in range(max_iter):
        # Update velocities
        inertia = 0.5
        cognitive_coeff = 0.5
        social_coeff = 0.2
        r1 = rng.random()
        r2 = rng.random()
        velocities = inertia * velocities + cognitive_coeff * r1 * (personal_best - particles) + social_coeff * r2 * (global_best - particles)
        # Update positions
        particles += velocities
        # Apply bounds
        particles = np.clip(particles, bounds[:, 0], bounds[:, 1])
        # Update personal best
        current_costs = np.array([cost_func(p) for p in particles])
        update_indices = current_costs < personal_best_cost
        personal_best[update_indices] = particles[update_indices]
        personal_best_cost[update_indices] = current_costs[update_indices]
        # Update global best
        global_best_idx = np.argmin(personal_best_cost)
        if personal_best_cost[global_best_idx] < global_best_cost:
            global_best = personal_best[global_best_idx]
            global_best_cost = personal_best_cost[global_best_idx]

    return global_best_cost


def generate_seed():
    bounds = np.array([[-5, 5], [-5, 5]])
    num_particles = 30
    max_iter = 250
    fortuna_rng = FortunaRNG()
    seed = pso(rosenbrock, bounds, num_particles, max_iter, fortuna_rng)
    return seed

# Example usage
if __name__ == "__main__":
    # Define parameters
    bounds = np.array([[-5, 5], [-5, 5]])
    num_particles = 30
    max_iter = 250
    fortuna_rng = FortunaRNG()
    seed = pso(rosenbrock, bounds, num_particles, max_iter, fortuna_rng)
    # Use PSO algorithm to generate seed for PRNG
    random.seed(seed)
    
    length = 10
    for i in range(10):
        print(random.randint(0,1))
