import pygame
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from ac_env import SingleSystemSoloAgentEnv, MAX_ENERGY, ACTIONS, draw_environment

# Screen and Training Hyperparameters
WIDTH = 1500
HEIGHT = 1000

# Actor-Critic Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
ENTROPY_WEIGHT = 0.01  # Encourage exploration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature layers
        self.feature_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor head: Policy network
        self.actor = nn.Linear(128, action_size)
        
        # Critic head: Value network
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        features = self.feature_network(state)
        policy_logits = self.actor(features)
        state_value = self.critic(features)
        return policy_logits, state_value

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.network = ActorCriticNetwork(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.action_size = action_size

    def select_action(self, state, valid_actions):
        if not valid_actions:
            print("Warning: No valid actions available. Choosing default action.")
            return random.choice(range(self.action_size))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            policy_logits, _ = self.network(state_tensor)
            
            # Mask invalid actions
            masked_policy_logits = policy_logits.clone()
            masked_policy_logits[0, :] = float('-inf')
            masked_policy_logits[0, valid_actions] = policy_logits[0, valid_actions]
            
            # Sample action from policy distribution
            action_probs = torch.softmax(masked_policy_logits, dim=1)
            action = torch.multinomial(action_probs, 1).item()
        
        return action

    def compute_returns(self, rewards, dones, values):
        """Compute discounted returns using bootstrapping."""
        returns = []
        R = 0
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = reward + GAMMA * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns).float().to(DEVICE)

    def update(self, states, actions, rewards, dones):
        # Convert lists to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        
        # Forward pass
        policy_logits, values = self.network(states)
        
        # Compute returns and advantages
        values = values.squeeze()
        returns = self.compute_returns(rewards, dones, values.detach().cpu().numpy())
        advantages = returns - values.detach()
        
        # Policy loss (actor)
        log_probs = torch.log_softmax(policy_logits, dim=1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Value loss (critic)
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy loss to encourage exploration
        entropy_loss = -(torch.softmax(policy_logits, dim=1) * log_probs).sum(dim=1).mean()
        
        # Combine losses
        total_loss = policy_loss + 0.5 * value_loss - ENTROPY_WEIGHT * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

def preprocess_state(observation):
    """Normalize state features."""
    observation = np.array(observation, dtype=np.float32)
    observation[-3] /= 1.0  # Energy level
    observation[-2] /= 10000.0  # Account balance
    return observation

def run_actor_critic_agent(env, num_episodes=500, render=True):
    """Train the Actor-Critic agent in the environment."""
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("EVE Mining Environment - Actor-Critic Agent")
        clock = pygame.time.Clock()
    else:
        screen = None

    # Initialize agent
    observation_space_size = preprocess_state(env.reset()[0]).shape[0]
    action_space_size = len(ACTIONS)
    agent = ActorCriticAgent(observation_space_size, action_space_size)

    episode_rewards = []
    account_balances_per_episode = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_state(observation)
        total_reward = 0
        done = False
        running = True
        step_number = 0

        # Episode memory
        episode_states, episode_actions, episode_rewards_list, episode_dones = [], [], [], []
        account_balances = []

        while not done and running:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        sys.exit()

            # Get valid actions
            valid_actions = env.get_valid_actions()

            # Select and perform an action
            action_idx = agent.select_action(state, valid_actions)
            action = ACTIONS[action_idx]

            # Take action
            observation_next, reward, done, truncated, info = env.step(action_idx)
            state_next = preprocess_state(observation_next)

            # Store experience
            episode_states.append(state)
            episode_actions.append(action_idx)
            episode_rewards_list.append(reward)
            episode_dones.append(done)

            # Update state
            state = state_next
            total_reward += reward
            step_number += 1

            # Record account balance
            account_balance = env.ship_state['account_balance']
            account_balances.append(account_balance)

            # Draw environment if rendering
            if render:
                action_description = get_action_description(action)
                draw_environment(screen, env, step_number, action_description, episode)
                clock.tick(60)

        # Update network at end of episode
        agent.update(episode_states, episode_actions, episode_rewards_list, episode_dones)

        episode_rewards.append(total_reward)
        account_balances_per_episode.append(account_balances)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Save the trained model
    torch.save(agent.network.state_dict(), "actor_critic_agent.pth")

    if render:
        pygame.quit()

    return episode_rewards, account_balances_per_episode

def evaluate_actor_critic_agent(env, agent, num_episodes=10):
    """Evaluate the trained Actor-Critic agent."""
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EVE Mining Environment - Actor-Critic Agent Evaluation")

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
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                policy_logits, _ = agent.network(state_tensor)
                
                # Mask invalid actions
                masked_policy_logits = policy_logits.clone()
                masked_policy_logits[0, :] = float('-inf')
                masked_policy_logits[0, valid_actions] = policy_logits[0, valid_actions]
                
                # Sample action from policy distribution
                action_probs = torch.softmax(masked_policy_logits, dim=1)
                action_idx = torch.argmax(action_probs, dim=1).item()

            action = ACTIONS[action_idx]
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
        0: 'Warp', 1: 'Mine', 2: 'Move', 3: 'Dock', 
        4: 'Empty Cargo', 5: 'Sell Minerals', 6: 'Buy Fuel'
    }
    action_type = action['action_type']
    action_name = action_names.get(action_type, 'Unknown')

    target = action.get('target', 'N/A')
    miner_id = action.get('miner_id', 'N/A')

    return f"{action_name}, Target: {target}, Miner ID: {miner_id}"

if __name__ == "__main__":
    env = SingleSystemSoloAgentEnv()

    # Train the Actor-Critic agent
    episode_rewards, account_balances_per_episode = run_actor_critic_agent(env, num_episodes=50)

    # Load the trained model for evaluation
    observation_space_size = preprocess_state(env.reset()[0]).shape[0]
    action_space_size = len(ACTIONS)

    trained_agent = ActorCriticAgent(state_size=observation_space_size, action_size=action_space_size)
    trained_agent.network.load_state_dict(torch.load("actor_critic_agent.pth"))
    trained_agent.network.eval()

    # Evaluate the Actor-Critic agent
    evaluation_episode_rewards, evaluation_account_balances_per_episode = evaluate_actor_critic_agent(env, trained_agent, num_episodes=10)

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
    plt.title('Actor-Critic Agent Training Performance')
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
    plt.title('Actor-Critic Agent Evaluation Performance')
    plt.show()
