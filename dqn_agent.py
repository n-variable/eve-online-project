import pygame
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from dqn_env import SingleSystemSoloAgentEnv, ASTEROID_BELTS, STATIONS, MINERALS, draw_environment

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Screen dimensions
WIDTH = 1500
HEIGHT = 1000

# DQN Hyperparameters
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000  # Decay rate for epsilon

# Define the neural network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)  # Output Q-values for each action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY

    def select_action(self, state):
        # Epsilon-greedy action selection
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = torch.FloatTensor([s[0] for s in batch]).to(self.device)
        action_batch = torch.LongTensor([s[1] for s in batch]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        reward_batch = torch.FloatTensor([s[3] for s in batch]).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor([s[4] for s in batch]).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (1 - done_batch) * GAMMA * next_state_values

        # Compute Huber loss
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Preprocessing functions
def preprocess_state(observation):
    # Flatten observation into a feature vector
    ship_state = observation['ship_state']
    cargo = ship_state['cargo']
    position = ship_state['position']
    energy = ship_state.get('energy', 100)
    account_balance = ship_state.get('account_balance', 0)
    miners = ship_state['miners']
    local_objects = observation['local_state']['objects']

    # Encode position as one-hot vector
    positions = ASTEROID_BELTS + STATIONS
    position_enc = np.zeros(len(positions))
    position_idx = positions.index(position)
    position_enc[position_idx] = 1

    # Encode miners
    miners_enc = miners

    # Encode local objects (simplified, count of objects)
    local_objects_count = len(local_objects)

    # Concatenate all features into a state vector
    state = np.concatenate([
        cargo,
        [energy],
        [account_balance],
        miners_enc,
        position_enc,
        [local_objects_count]
    ])
    return state

def action_from_index(action_idx):
    # Extend the action space to include mining up to first 3 asteroids
    actions = [
        {'action_type': 0, 'target': 0, 'miner_id': None},  # Warp to Belt1
        {'action_type': 0, 'target': 1, 'miner_id': None},  # Warp to Belt2
        {'action_type': 0, 'target': 2, 'miner_id': None},  # Warp to Belt3
        # Mining actions for first three asteroids
        {'action_type': 1, 'target': 0, 'miner_id': 0},     # Mine asteroid 0
        {'action_type': 1, 'target': 1, 'miner_id': 0},     # Mine asteroid 1
        {'action_type': 1, 'target': 2, 'miner_id': 0},     # Mine asteroid 2
        {'action_type': 5, 'target': None, 'miner_id': None},  # Sell minerals
        {'action_type': 6, 'target': None, 'miner_id': None},  # Buy fuel
        {'action_type': 0, 'target': 3, 'miner_id': None},  # Warp to Station1
        {'action_type': 0, 'target': 4, 'miner_id': None},  # Warp to Station2
    ]
    return actions[action_idx]

def index_from_action(action):
    # Map action dict to index
    actions = [
        {'action_type': 0, 'target': 0, 'miner_id': None},  # Warp to Belt1
        {'action_type': 0, 'target': 1, 'miner_id': None},  # Warp to Belt2
        {'action_type': 0, 'target': 2, 'miner_id': None},  # Warp to Belt3
        {'action_type': 1, 'target': 0, 'miner_id': 0},     # Mine asteroid
        {'action_type': 5, 'target': None, 'miner_id': None},  # Sell minerals
        {'action_type': 6, 'target': None, 'miner_id': None},  # Buy fuel
        {'action_type': 0, 'target': 3, 'miner_id': None},  # Warp to Station1
        {'action_type': 0, 'target': 4, 'miner_id': None},  # Warp to Station2
    ]
    return actions.index(action)

def run_dqn_agent(env, num_episodes=500, render=True):
    """Train and evaluate the DQN agent in the environment."""
    if render:
        pygame.init()
        # Set up the display
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("EVE Mining Environment - DQN Agent")
        clock = pygame.time.Clock()
    else:
        screen = None

    # Get state and action sizes
    observation_space_size = preprocess_state(env.reset()[0]).shape[0]
    action_space_size = 10  # Updated from 8 to 10

    agent = DQNAgent(state_size=observation_space_size, action_size=action_space_size)

    episode_rewards = []
    account_balances_per_episode = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_state(observation)
        total_reward = 0
        done = False
        step_number = 0
        running = True

        account_balances = []

        while not done and running:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        sys.exit()

            # Select and perform an action
            action_idx = agent.select_action(state)
            action = action_from_index(action_idx)
            current_action = action

            # Take the action in the environment
            observation_next, reward, done, truncated, info = env.step(action)
            state_next = preprocess_state(observation_next)

            # Remember the transition
            agent.remember(state, action_idx, state_next, reward, done)

            # Move to the next state
            state = state_next

            # Perform one step of the optimization
            agent.optimize_model()

            # Update the target network periodically
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            total_reward += reward
            step_number += 1

            # Record the current account balance
            ship_state = observation_next['ship_state']
            account_balance = ship_state.get('account_balance', 0)
            account_balances.append(account_balance)

            # Draw the environment only if rendering is enabled
            if render:
                action_description = get_action_description(action)
                draw_environment(screen, env, step_number, action_description, episode)
                clock.tick(60)  # Control the frame rate

        episode_rewards.append(total_reward)
        account_balances_per_episode.append(account_balances)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "dqn_agent.pth")

    if render:
        pygame.quit()

    # Return the collected data for plotting
    return episode_rewards, account_balances_per_episode

def evaluate_dqn_agent(env, agent, num_episodes=10):
    """Evaluate the trained DQN agent."""
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EVE Mining Environment - DQN Agent Evaluation")

    clock = pygame.time.Clock()

    episode_rewards = []
    account_balances_per_episode = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_state(observation)
        total_reward = 0
        done = False
        step_number = 0
        running = True

        account_balances = []

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

            # Select action (without exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()

            action = action_from_index(action_idx)
            current_action = action

            # Take the action in the environment
            observation_next, reward, done, truncated, info = env.step(action)
            state_next = preprocess_state(observation_next)

            # Move to the next state
            state = state_next

            total_reward += reward
            step_number += 1

            # Record the current account balance
            ship_state = observation_next['ship_state']
            account_balance = ship_state.get('account_balance', 0)
            account_balances.append(account_balance)

            # Draw the environment
            action_description = get_action_description(action)
            draw_environment(screen, env, step_number, action_description, 1)

            # Control the frame rate
            clock.tick(60)  # Adjust the number to control the speed (frames per second)

        episode_rewards.append(total_reward)
        account_balances_per_episode.append(account_balances)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    pygame.quit()  # Ensure Pygame is properly quit after evaluation

    average_reward = sum(episode_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} Evaluation Episodes: {average_reward}")

    # Return the collected data for plotting
    return episode_rewards, account_balances_per_episode

def get_action_description(action):
    """Generate a human-readable description of the action."""
    action_names = {
        0: 'Warp',
        1: 'Mine',
        2: 'Move',
        3: 'Dock',
        4: 'Empty Cargo',
        5: 'Sell Minerals',
        6: 'Buy Fuel'
    }
    action_type = action['action_type']
    action_name = action_names.get(action_type, 'Unknown')

    target = action.get('target', 'N/A')
    miner_id = action.get('miner_id', 'N/A')

    return f"{action_name}, Target: {target}, Miner ID: {miner_id}"

if __name__ == "__main__":
    env = SingleSystemSoloAgentEnv()

    # Train the agent
    episode_rewards, account_balances_per_episode = run_dqn_agent(env, num_episodes=50)

    # Load the trained model for evaluation
    observation_space_size = preprocess_state(env.reset()[0]).shape[0]
    action_space_size = 10  # Updated from 8 to 10

    trained_agent = DQNAgent(state_size=observation_space_size, action_size=action_space_size)
    trained_agent.policy_net.load_state_dict(torch.load("dqn_agent.pth"))
    trained_agent.policy_net.eval()

    # Evaluate the agent
    evaluation_episode_rewards, evaluation_account_balances_per_episode = evaluate_dqn_agent(env, trained_agent, num_episodes=10)

    # Plotting the training and evaluation performance
    import matplotlib.pyplot as plt

    # Plot the account balance over time during training
    plt.figure(figsize=(10, 6))
    # To avoid overcrowding the plot, plot only the first 10 episodes
    for idx, balances in enumerate(account_balances_per_episode[:10]):
        plt.plot(balances, label=f'Episode {idx+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Account Balance')
    plt.title('Profit Over Time During Training (First 10 Episodes)')
    plt.legend()
    plt.show()

    # Plot the episode rewards during training
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Agent Training Performance')
    plt.show()

    # Plot the account balance over time during evaluation
    plt.figure(figsize=(10, 6))
    for idx, balances in enumerate(evaluation_account_balances_per_episode):
        plt.plot(balances, label=f'Evaluation Episode {idx+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Account Balance')
    plt.title('Profit Over Time During Evaluation')
    plt.legend()
    plt.show()

    # Plot the episode rewards during evaluation
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Agent Evaluation Performance')
    plt.show()