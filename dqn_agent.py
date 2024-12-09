import pygame
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from dqn_env import SingleSystemSoloAgentEnv, MAX_ENERGY, ACTIONS, draw_environment

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
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000

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

    def select_action(self, state, valid_actions):
        if not valid_actions:
            print("Warning: No valid actions available. Choosing default action.")
            # Choose a default action, e.g., 'Wait'
            wait_action_idx = next((idx for idx, action in enumerate(ACTIONS) if action['action_type'] == 7), None)
            if wait_action_idx is not None:
                return wait_action_idx
            else:
                # If 'Wait' is not defined, return a random action
                return random.randint(0, self.action_size - 1)
        # Epsilon-greedy action selection with action masking
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            # Randomly select a valid action
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy().squeeze()
                # Mask invalid actions
                masked_q_values = np.full_like(q_values, -np.inf)
                masked_q_values[valid_actions] = q_values[valid_actions]
                return np.argmax(masked_q_values)

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

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1})
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

        # Compute expected Q values
        expected_state_action_values = reward_batch + (1 - done_batch) * GAMMA * next_state_values

        # Compute loss
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

# Preprocessing functions
def preprocess_state(observation):
    # Normalize energy level and account balance if necessary
    observation = np.array(observation, dtype=np.float32)
    observation[-3] /= 1.0  # Energy level is already normalized between 0 and 1
    observation[-2] /= 10000.0  # Adjust based on expected scale
    return observation

def action_from_index(action_idx):
    return ACTIONS[action_idx]

def index_from_action(action):
    # Use the ACTIONS list from the environment
    return ACTIONS.index(action)

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
    action_space_size = len(ACTIONS)

    agent = DQNAgent(state_size=observation_space_size, action_size=action_space_size)

    episode_rewards = []
    account_balances_per_episode = []
    losses = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_state(observation)
        total_reward = 0
        done = False
        step_number = 0
        running = True

        account_balances = []

        episode_loss = 0

        while not done and running:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        sys.exit()

            # Get valid actions from the environment
            valid_actions = env.get_valid_actions()

            # Select and perform an action
            action_idx = agent.select_action(state, valid_actions)
            action = action_from_index(action_idx)
            current_action = action

            # Take the action in the environment
            observation_next, reward, done, truncated, info = env.step(action_idx)
            state_next = preprocess_state(observation_next)

            # Remember the transition
            agent.remember(state, action_idx, state_next, reward, done)

            # Move to the next state
            state = state_next

            # Perform one step of the optimization
            loss = agent.optimize_model()
            if loss:
                episode_loss += loss.item()

            # Update the target network periodically
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            total_reward += reward
            step_number += 1

            # Record the current account balance
            account_balance = env.ship_state['account_balance']
            account_balances.append(account_balance)

            # Draw the environment only if rendering is enabled
            if render:
                action_description = get_action_description(action)
                draw_environment(screen, env, step_number, action_description, episode)
                clock.tick(60)  # Control the frame rate

        if episode_loss > 0:
            losses.append(episode_loss)

        episode_rewards.append(total_reward)
        account_balances_per_episode.append(account_balances)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "dqn_agent.pth")

    if render:
        pygame.quit()

    # Return the collected data for plotting
    return episode_rewards, account_balances_per_episode, losses

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

            # Get valid actions from the environment
            valid_actions = env.get_valid_actions()

            # Select action (without exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor).cpu().numpy().squeeze()
                # Mask invalid actions
                masked_q_values = np.full_like(q_values, -np.inf)
                masked_q_values[valid_actions] = q_values[valid_actions]
                action_idx = np.argmax(masked_q_values)

            action = action_from_index(action_idx)
            current_action = action

            # Take the action in the environment
            observation_next, reward, done, truncated, info = env.step(action_idx)
            state_next = preprocess_state(observation_next)

            # Move to the next state
            state = state_next

            total_reward += reward
            step_number += 1

            # Record the current account balance
            account_balance = env.ship_state['account_balance']
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
        5: 'Sell Minerals',
        6: 'Buy Fuel',
        7: 'Wait'
    }
    action_type = action['action_type']
    action_name = action_names.get(action_type, 'Unknown')

    target = action.get('target', 'N/A')
    miner_id = action.get('miner_id', 'N/A')

    return f"{action_name}, Target: {target}, Miner ID: {miner_id}"

if __name__ == "__main__":
    env = SingleSystemSoloAgentEnv()

    # Train the agent
    episode_rewards, account_balances_per_episode, losses = run_dqn_agent(env, num_episodes=50)

    # Load the trained model for evaluation
    observation_space_size = preprocess_state(env.reset()[0]).shape[0]
    action_space_size = len(ACTIONS)

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

    # Plot the loss over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('DQN Agent Training Loss')
    plt.show()