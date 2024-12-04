import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

# Define constants
MINERALS = {
    'Veldspar': 10,
    'Scordite': 20,
    'Pyroxeres': 30,
}

ASTEROID_BELTS = ['Belt1', 'Belt2', 'Belt3']
STATIONS = ['Station1', 'Station2']

class SingleSystemSoloAgentEnv(gym.Env):
    def __init__(self):
        super(SingleSystemSoloAgentEnv, self).__init__()

        # Action space
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(6),  # 0: Warp, 1: Mine, 2: Move, 3: Dock, 4: Empty Cargo, 5: Sell Minerals
            'target': spaces.Discrete(10),      # Target ID (asteroid, station, etc.)
            'miner_id': spaces.Discrete(2),     # Miner ID (if applicable)
        })

        # Observation space
        self.observation_space = spaces.Dict({
            'system_state': spaces.Dict({
                'locations': spaces.MultiDiscrete([len(ASTEROID_BELTS), len(STATIONS)]),
            }),
            'local_state': spaces.Dict({
                'objects': spaces.MultiDiscrete([100]),  # IDs of nearby objects
            }),
            'ship_state': spaces.Dict({
                'cargo': spaces.Box(low=0, high=1000, shape=(len(MINERALS),), dtype=np.int32),
                'position': spaces.Discrete(10),  # Simplified position
                'miners': spaces.MultiBinary(2),  # Status of miners (0: idle, 1: active)
            }),
        })

        # System-wide state: Map each belt to its resources
        self.belt_resources = {belt: self.generate_asteroids() for belt in ASTEROID_BELTS}

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize system state
        self.system_state = {
            'locations': ASTEROID_BELTS + STATIONS
        }

        # Initialize local state
        self.local_state = {
            'objects': [],
        }

        # Initialize ship state
        self.ship_state = {
            'cargo': np.zeros(len(MINERALS), dtype=np.int32),
            'position': 'Station1',
            'miners': np.zeros(2, dtype=np.int8),
        }

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        done = False
        reward = 0
        info = {}

        action_type = action['action_type']
        target = action['target']
        miner_id = action.get('miner_id', None)

        if action_type == 0:  # Warp
            self._warp(target)
        elif action_type == 1:  # Mine
            self._mine(miner_id, target)
        elif action_type == 2:  # Move
            self._move(target)
        elif action_type == 3:  # Dock
            self._dock(target)
        elif action_type == 4:  # Empty Cargo
            self._empty_cargo()
        elif action_type == 5:  # Sell Minerals
            reward += self._sell_minerals()

        # Regenerate resources occasionally
        if random.random() < 0.1:  # 10% chance to regenerate in this step
            self._regenerate_belt_resources()

        observation = self._get_obs()
        return observation, reward, done, False, info

    def _get_obs(self):
        # Map mineral names to indices
        mineral_indices = {mineral: idx for idx, mineral in enumerate(MINERALS.keys())}
        cargo_array = np.array([self.ship_state['cargo'][idx] for idx in mineral_indices.values()], dtype=np.int32)

        observation = {
            'system_state': {
                'locations': self.system_state['locations'],
            },
            'local_state': {
                'objects': self.local_state['objects'],
            },
            'ship_state': {
                'cargo': cargo_array,
                'position': self.ship_state['position'],
                'miners': self.ship_state['miners'],
            },
        }
        return observation

    def render(self):
        print("Current Position:", self.ship_state['position'])
        print("Cargo Hold:", self.ship_state['cargo'])
        print("Miners Status:", self.ship_state['miners'])
        print("Local Objects:", self.local_state['objects'])

    def _warp(self, target_id):
        if target_id < len(self.system_state['locations']):
            location = self.system_state['locations'][target_id]
            self.ship_state['position'] = location
            print(f"Warped to {self.ship_state['position']}")

            if location in ASTEROID_BELTS:
                self.local_state['objects'] = self.belt_resources[location]
            else:
                self.local_state['objects'] = []
        else:
            print("Invalid warp target")

    def _mine(self, miner_id, target_id):
        if self.ship_state['miners'][miner_id] == 0:
            self.ship_state['miners'][miner_id] = 1
            if target_id < len(self.local_state['objects']):
                asteroid = self.local_state['objects'][target_id]
                mineral_idx = list(MINERALS.keys()).index(asteroid['mineral'])
                self.ship_state['cargo'][mineral_idx] += asteroid['quantity']
                print(f"Mined {asteroid['quantity']} units of {asteroid['mineral']}")
                self.local_state['objects'].pop(target_id)
                self.belt_resources[self.ship_state['position']] = self.local_state['objects']
        else:
            print("Miner is already active")

    def _move(self, target):
        print(f"Moving towards {target}")

    def _dock(self, target_id):
        if target_id < len(STATIONS):
            self.ship_state['position'] = STATIONS[target_id]
            print(f"Docked at {self.ship_state['position']}")
        else:
            print("Invalid docking target")

    def _empty_cargo(self):
        self.ship_state['cargo'] = np.zeros(len(MINERALS), dtype=np.int32)
        print("Emptied cargo hold")

    def _sell_minerals(self):
        total_earnings = 0
        for idx, quantity in enumerate(self.ship_state['cargo']):
            if quantity > 0:
                mineral = list(MINERALS.keys())[idx]
                price = MINERALS[mineral]
                total_earnings += quantity * price
                print(f"Sold {quantity} units of {mineral} for {quantity * price} ISK")
                self.ship_state['cargo'][idx] = 0
        print(f"Total Earnings: {total_earnings} ISK")
        return total_earnings

    def _regenerate_belt_resources(self):
        for belt in self.belt_resources:
            if not self.belt_resources[belt]:  # If the belt is empty
                self.belt_resources[belt] = self.generate_asteroids()
                print(f"Resources regenerated in {belt}")

    def generate_asteroids(self):
        asteroids = []
        for _ in range(random.randint(5, 10)):
            mineral = random.choice(list(MINERALS.keys()))
            quantity = random.randint(50, 200)
            asteroids.append({'mineral': mineral, 'quantity': quantity})
        return asteroids

    def close(self):
        pass

import pygame
import sys

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Screen dimensions
WIDTH = 1500
HEIGHT = 1000

def draw_environment(screen, env, step_number):
    """Draw the current state of the environment."""
    # Clear the screen
    screen.fill(BLACK)

    # Title
    font = pygame.font.Font(None, 36)
    title_text = font.render(f"Step {step_number} - EVE Mining Environment", True, WHITE)
    screen.blit(title_text, (20, 20))

    # Draw ship's position
    ship_pos_text = font.render(f"Ship Position: {env.ship_state['position']}", True, GREEN)
    screen.blit(ship_pos_text, (20, 60))

    # Draw cargo hold
    cargo_text = "\n".join([f"{mineral}: {qty}" for mineral, qty in zip(MINERALS.keys(), env.ship_state['cargo'])])
    cargo_lines = cargo_text.split("\n")
    cargo_y = 100
    for line in cargo_lines:
        cargo_line_text = font.render(line, True, WHITE)
        screen.blit(cargo_line_text, (20, cargo_y))
        cargo_y += 20

    # Draw local objects
    local_text = font.render("Local Objects:", True, BLUE)
    screen.blit(local_text, (20, cargo_y + 20))
    local_objects = env.local_state['objects']
    if local_objects:
        for i, obj in enumerate(local_objects):
            obj_text = font.render(f"{i}: {obj['mineral']} ({obj['quantity']})", True, WHITE)
            screen.blit(obj_text, (40, cargo_y + 50 + i * 20))
    else:
        no_objects_text = font.render("No nearby objects", True, RED)
        screen.blit(no_objects_text, (40, cargo_y + 50))

    system_x = 300

    # Draw asteroid belts
    belt_x = system_x
    belt_y = 200

    system_text = font.render("System", True, WHITE)
    screen.blit(system_text, (system_x + 10, belt_y - 20))
    
    for belt in ASTEROID_BELTS:
        belt_rect = pygame.Rect(belt_x, belt_y, 100, 50)
        pygame.draw.rect(screen, GRAY, belt_rect)
        belt_text = font.render(belt, True, WHITE)
        screen.blit(belt_text, (belt_x + 10, belt_y + 10))

        belt_resources = env.belt_resources[belt]
        if belt_resources:
            for i, asteroid in enumerate(belt_resources[:3]):  # Display up to 3 resources
                resource_text = font.render(f"{asteroid['mineral']} ({asteroid['quantity']})", True, GREEN)
                screen.blit(resource_text, (belt_x + 10, belt_y + 30 + i * 20))
        else:
            empty_text = font.render("Empty", True, RED)
            screen.blit(empty_text, (belt_x + 10, belt_y + 30))

        belt_x += 150

    # Draw stations
    station_x = system_x
    station_y = 500
    for station in STATIONS:
        station_rect = pygame.Rect(station_x, station_y, 100, 50)
        pygame.draw.rect(screen, BLUE, station_rect)
        station_text = font.render(station, True, WHITE)
        screen.blit(station_text, (station_x + 10, station_y + 10))
        station_x += 150

    # Update the screen
    pygame.display.flip()

def run_environment(env):
    """Run the environment with visualization."""
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EVE Mining Environment")

    clock = pygame.time.Clock()
    running = True

    step_number = 0
    obs, info = env.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Example action sequence for testing
        if step_number == 0:
            action = {'action_type': 0, 'target': 0, 'miner_id': None}  # Warp to Belt1
        elif step_number == 1:
            action = {'action_type': 1, 'target': 0, 'miner_id': 0}  # Mine first asteroid
        elif step_number == 2:
            action = {'action_type': 0, 'target': len(ASTEROID_BELTS), 'miner_id': None}  # Warp to Station1
        elif step_number == 3:
            action = {'action_type': 5, 'target': None, 'miner_id': None}  # Sell minerals
        else:
            action = {'action_type': 0, 'target': 1, 'miner_id': None}  # Warp to Belt2

        obs, reward, done, truncated, info = env.step(action)

        # Draw the environment
        draw_environment(screen, env, step_number)

        # Increment step counter
        step_number += 1

        # Control the frame rate
        clock.tick(2)  # 2 frames per second

    pygame.quit()
    sys.exit()
