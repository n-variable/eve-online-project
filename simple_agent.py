import pygame
import sys
from environments import SingleSystemSoloAgentEnv, ASTEROID_BELTS, STATIONS, MINERALS, draw_environment

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

def simple_agent(observation):
    """Decide on an action based on the current observation."""
    # Extract relevant information from the observation
    ship_state = observation['ship_state']
    position = ship_state['position']
    cargo = ship_state['cargo']
    energy = ship_state.get('energy', 100)
    account_balance = ship_state.get('account_balance', 0)
    miners = ship_state['miners']
    local_objects = observation['local_state']['objects']

    # Define maximums
    MAX_CARGO = 1000  # Adjust if your environment uses a different value
    MAX_ENERGY = 100
    MIN_ENERGY_THRESHOLD = 20  # Energy level at which the agent decides to buy fuel

    # Calculate total cargo
    cargo_total = cargo.sum()

    # Agent's decision-making logic
    if position in STATIONS:
        # At a station
        if cargo_total > 0:
            # Sell minerals
            action = {'action_type': 5, 'target': None, 'miner_id': None}
        elif energy < (MAX_ENERGY - MIN_ENERGY_THRESHOLD) and account_balance > 0:
            # Buy fuel if energy is low and there's enough money
            action = {'action_type': 6, 'target': None, 'miner_id': None}
        else:
            # Warp to the first asteroid belt
            action = {'action_type': 0, 'target': 0, 'miner_id': None}  # Target index 0 is the first belt
    elif position in ASTEROID_BELTS:
        # At an asteroid belt
        if cargo_total >= MAX_CARGO:
            # Warp to the first station to sell minerals
            action = {'action_type': 0, 'target': len(ASTEROID_BELTS), 'miner_id': None}
        elif local_objects:
            # Mine the first available asteroid
            action = {'action_type': 1, 'target': 0, 'miner_id': 0}  # Use miner ID 0
        else:
            # Warp to the next asteroid belt (rotating through belts)
            current_belt_index = ASTEROID_BELTS.index(position)
            next_belt_index = (current_belt_index + 1) % len(ASTEROID_BELTS)
            action = {'action_type': 0, 'target': next_belt_index, 'miner_id': None}
    else:
        # If in an unknown position, warp to the first station
        action = {'action_type': 0, 'target': len(ASTEROID_BELTS), 'miner_id': None}

    return action

def run_agent_environment(env):
    """Run the environment with the simple agent and visualization."""
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EVE Mining Environment")

    clock = pygame.time.Clock()
    running = True

    step_number = 0
    current_action = None

    observation, info = env.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent decides on an action
        action = simple_agent(observation)
        current_action = action

        # Take the action in the environment
        observation, reward, done, truncated, info = env.step(action)
        step_number += 1

        # Draw the environment
        draw_environment(screen, env, step_number, current_action)

        # Control the frame rate
        clock.tick(1)  # Adjust the number to control the speed (frames per second)

    pygame.quit()
    sys.exit()

# The rest of your code, including draw_environment, should remain as it is or be adjusted as needed.
