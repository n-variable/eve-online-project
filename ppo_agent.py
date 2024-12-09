import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical
from ppo_env import ModularMiningEnv

# If you implemented the environment as above, ensure it's imported or in the same file.
# from your_module import ModularMiningEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(policy, env_fn, num_episodes=10, render=False):
    """
    Evaluate a trained policy in the given environment setup.

    Args:
        policy: The trained policy model (ActorCriticNet) returned by train_ppo.
        env_fn: A function that, when called, returns a new instance of the environment.
        num_episodes: Number of episodes to run for evaluation.
        render: If True, render the environment at each step.

    Returns:
        average_return: The mean total return over the evaluated episodes.
    """
    env = env_fn()
    returns = []

    for episode_i in range(num_episodes):
        obs, info = env.reset()
        ep_ret = 0
        done = False
        truncated = False

        while not (done or truncated):
            if render:
                env.render()

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = policy.forward(obs_tensor)
                # Use greedy action selection for evaluation (no randomness)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward

        returns.append(ep_ret)

    env.close()
    average_return = np.mean(returns)
    print(f"Evaluation over {num_episodes} episodes: Average Return = {average_return}")
    return average_return

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCriticNet, self).__init__()
        # A simple feedforward network
        hidden_size = 128
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.fc(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def act(self, obs):
        """Given an observation, sample an action and return action, log_prob and value."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs, actions):
        """Evaluate given actions: return log_probs, entropy, and values for PPO update."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy, values

class RolloutBuffer:
    def __init__(self, obs_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int64)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when a terminal state is reached.
        Uses GAE-Lambda for advantage calculation.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = self.rew_buf[path_slice]
        vals = self.val_buf[path_slice]

        # GAE advantage calculation
        adv = 0
        for i in reversed(range(len(rews))):
            delta = rews[i] + self.gamma * (last_val if i == len(rews)-1 else vals[i+1]) - vals[i]
            adv = delta + self.gamma * self.lam * adv
            self.adv_buf[self.path_start_idx + i] = adv

        # Compute returns
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Normalize advantages
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std() + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return (torch.as_tensor(self.obs_buf, dtype=torch.float32, device=device),
                torch.as_tensor(self.act_buf, dtype=torch.int64, device=device),
                torch.as_tensor(self.adv_buf, dtype=torch.float32, device=device),
                torch.as_tensor(self.ret_buf, dtype=torch.float32, device=device),
                torch.as_tensor(self.logp_buf, dtype=torch.float32, device=device))

def ppo_update(policy, optimizer, data, clip_ratio, entropy_coef, vf_coef, train_iters, batch_size):
    obs, act, adv, ret, logp_old = data
    dataset_size = len(obs)
    for i in range(train_iters):
        idxs = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            obs_b = obs[batch_idx]
            act_b = act[batch_idx]
            adv_b = adv[batch_idx]
            ret_b = ret[batch_idx]
            logp_old_b = logp_old[batch_idx]

            # Forward pass
            logp, entropy, v = policy.evaluate_actions(obs_b, act_b)
            ratio = torch.exp(logp - logp_old_b)
            # PPO loss
            surrogate1 = ratio * adv_b
            surrogate2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_b
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = ((ret_b - v)**2).mean()

            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_ppo(env_fn, epochs=50, steps_per_epoch=2000, gamma=0.99, lam=0.95,
              clip_ratio=0.2, pi_lr=3e-4, vf_lr=3e-4, train_iters=10, batch_size=64,
              entropy_coef=0.01, vf_coef=0.5, render=False):
    env = env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = ActorCriticNet(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=pi_lr)

    buf = RolloutBuffer(obs_dim, steps_per_epoch, gamma, lam)

    for epoch in range(epochs):
        obs, info = env.reset()
        ep_ret = 0
        ep_len = 0

        for t in range(steps_per_epoch):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, entropy, value = policy.act(obs_tensor)
            action = action.item()
            next_obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1

            buf.store(obs, action, reward, value.item(), logp.item())

            obs = next_obs

            terminal = done or truncated
            if terminal or (t == steps_per_epoch - 1):
                if terminal:
                    last_val = 0 if done else value.item()
                else:
                    # If time horizon ended, bootstrap value
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    _, val = policy.forward(obs_tensor)
                    last_val = val.item()
                buf.finish_path(last_val)

                if terminal:
                    obs, info = env.reset()

        # PPO update
        data = buf.get()
        ppo_update(policy, optimizer, data, clip_ratio, entropy_coef, vf_coef, train_iters, batch_size)

        print(f"Epoch {epoch+1}/{epochs}: Last Episode Return: {ep_ret}, Episode Length: {ep_len}")

        if render:
            env.render()

    env.close()
    return policy

# Example usage:
if __name__ == "__main__":
    # Create a simple Stage 1 env config
    simple_config = {
        "num_belts": 1,
        "num_stations": 1,
        "belt_config": [
            {
                "belt_name": "Belt1",
                "mineral_distribution": {
                    "Veldspar": (50, 200),
                }
            }
        ],
        "station_config": [
            {
                "station_name": "Station1",
                "mineral_prices": {"Veldspar": 12},
                "fuel_price": None
            }
        ],
        "energy_config": {
            "max_energy": 300,
            "energy_regen_rate": 1,
            "energy_costs": {
                "warp": 20,
                "mine": 15,
                "wait": 0
            },
            "allow_wait_regen": True,
        },
        "cargo_capacity": 500,
        "reward_scale": 0.001,
        "max_steps": 2000
    }

    def env_fn():
        return ModularMiningEnv(env_config=simple_config)

    # Train the PPO on the simple environment
    trained_policy = train_ppo(env_fn, epochs=50, steps_per_epoch=2000, render=False)
    evaluate_policy(trained_policy, env_fn, num_episodes=10, render=True)
