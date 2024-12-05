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

# Add this dictionary
STATION_MINERAL_PRICES = {
    'Station1': {
        'Veldspar': 12,
        'Scordite': 18,
        'Pyroxeres': 28,
    },
    'Station2': {
        'Veldspar': 11,
        'Scordite': 19,
        'Pyroxeres': 29,
    },
}

# Add this dictionary for fuel prices
STATION_FUEL_PRICES = {
    'Station1': 5,  # ISK per unit of energy
    'Station2': 6,
}

# Energy constants
MAX_ENERGY = 100
ENERGY_REGEN_RATE = 5  # Energy regenerated per step
ENERGY_COSTS = {
    'warp': 15,
    'mine': 10,
    'move': 5,
}

class SingleSystemSoloAgentEnv(gym.Env):
    def __init__(self):
        super(SingleSystemSoloAgentEnv, self).__init__()

        # Update the action space to include the new action type
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(7),  # 0: Warp, ..., 6: Buy Fuel
            'target': spaces.Discrete(10),      # Target ID or amount
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
            'account_balance': 0,
            'energy': MAX_ENERGY,  # Add energy attribute
        }

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        done = False
        reward = 0
        info = {}

        action_type = action['action_type']
        target = action.get('target', None)
        miner_id = action.get('miner_id', None)

        # Regenerate energy at the beginning of the step
        self.ship_state['energy'] = min(
            self.ship_state['energy'] + ENERGY_REGEN_RATE,
            MAX_ENERGY
        )

        # Energy cost mapping
        action_energy_costs = {
            0: ENERGY_COSTS['warp'],   # Warp
            1: ENERGY_COSTS['mine'],   # Mine
            2: ENERGY_COSTS['move'],   # Move
            3: 0,                      # Dock (no energy cost)
            4: 0,                      # Empty Cargo (no energy cost)
            5: 0,                      # Sell Minerals (no energy cost)
            6: 0,                      # Buy Fuel (no energy cost)
        }

        energy_cost = action_energy_costs.get(action_type, 0)

        # Check if there's enough energy
        if self.ship_state['energy'] >= energy_cost:
            # Deduct energy cost
            self.ship_state['energy'] -= energy_cost

            # Perform the action
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
            elif action_type == 6:  # Buy Fuel
                self._buy_fuel()
        else:
            print("Not enough energy to perform this action.")

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
                'energy': self.ship_state['energy'],  # Include energy in observations
            },
        }
        return observation

    def render(self):
        print("Current Position:", self.ship_state['position'])
        print("Cargo Hold:", self.ship_state['cargo'])
        print("Miners Status:", self.ship_state['miners'])
        print("Account Balance:", self.ship_state['account_balance'], "ISK")
        print("Energy Level:", self.ship_state['energy'], "/", MAX_ENERGY)
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
        
        # Get the current station
        current_station = self.ship_state['position']
        
        # Check if the ship is at a station with specific prices
        if current_station in STATION_MINERAL_PRICES:
            prices = STATION_MINERAL_PRICES[current_station]
            print(f"Selling minerals at {current_station} with station-specific prices.")
        else:
            prices = MINERALS  # Default to base prices if not at a known station
            print(f"Selling minerals at {current_station} with default prices.")
        
        # Sell minerals using the appropriate prices
        for idx, quantity in enumerate(self.ship_state['cargo']):
            if quantity > 0:
                mineral = list(MINERALS.keys())[idx]
                price = prices.get(mineral, MINERALS[mineral])
                total_sale = quantity * price
                total_earnings += total_sale
                print(f"Sold {quantity} units of {mineral} for {total_sale} ISK at {current_station}")
                self.ship_state['cargo'][idx] = 0  # Empty the cargo for this mineral
        
        # Update account balance
        self.ship_state['account_balance'] += total_earnings
        print(f"Total Earnings: {total_earnings} ISK")
        print(f"Updated Account Balance: {self.ship_state['account_balance']} ISK")
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

    def _buy_fuel(self):
        """Allows the ship to purchase fuel (energy) at the current station."""
        current_station = self.ship_state['position']
        if current_station in STATIONS:
            # Get fuel price at the current station
            fuel_price = STATION_FUEL_PRICES.get(current_station, None)
            if fuel_price is not None:
                # Calculate the amount of energy needed to reach MAX_ENERGY
                energy_needed = MAX_ENERGY - self.ship_state['energy']
                if energy_needed > 0:
                    total_cost = energy_needed * fuel_price
                    if self.ship_state['account_balance'] >= total_cost:
                        # Deduct the cost and refill energy
                        self.ship_state['account_balance'] -= total_cost
                        self.ship_state['energy'] = MAX_ENERGY
                        print(f"Bought {energy_needed} units of fuel for {total_cost} ISK at {current_station}")
                        print(f"Updated Account Balance: {self.ship_state['account_balance']} ISK")
                    else:
                        print("Not enough ISK to buy fuel.")
                else:
                    print("Energy is already full.")
            else:
                print("No fuel available at this station.")
        else:
            print("Cannot buy fuel outside of a station.")

import pygame
import sys

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

def draw_environment(screen, env, step_number, current_action):
    """Draw the current state of the environment."""
    # Clear the screen
    screen.fill(BLACK)

    # Title
    font = pygame.font.Font(None, 36)
    title_text = font.render(f"Step {step_number} - EVE Mining Environment", True, WHITE)
    screen.blit(title_text, (20, 20))

    # Display current action
    action_text = font.render(f"Current Action: {current_action}", True, YELLOW)
    screen.blit(action_text, (20, 60))

    # Draw account balance
    account_balance_text = font.render(f"Account Balance: {env.ship_state['account_balance']} ISK", True, GREEN)
    screen.blit(account_balance_text, (20, 100))

    # Draw energy level
    energy_text = font.render(f"Energy Level: {env.ship_state['energy']} / {MAX_ENERGY}", True, RED)
    screen.blit(energy_text, (20, 140))

    # Draw cargo hold
    cargo_text = "\n".join([f"{mineral}: {qty}" for mineral, qty in zip(MINERALS.keys(), env.ship_state['cargo'])])
    cargo_lines = cargo_text.split("\n")
    cargo_y = 180
    for line in cargo_lines:
        cargo_line_text = font.render(line, True, WHITE)
        screen.blit(cargo_line_text, (20, cargo_y))
        cargo_y += 20

    # Draw local objects
    local_text = font.render("Local Objects", True, BLUE)
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

    # Dictionary to hold positions of belts and stations
    locations_positions = {}

    # Draw asteroid belts
    belt_x = system_x
    belt_y = 200

    system_text = font.render("Solar System", True, WHITE)
    screen.blit(system_text, (system_x + 10, belt_y - 40))

    belt_spacing = 250  # Spacing between belts
    for belt in ASTEROID_BELTS:
        belt_rect = pygame.Rect(belt_x, belt_y + 10, 100, 50)
        pygame.draw.rect(screen, GRAY, belt_rect)
        belt_text = font.render(belt, True, WHITE)
        screen.blit(belt_text, (belt_x + 10, belt_y + 20))

        # Store the position of the belt
        locations_positions[belt] = (belt_x + 50, belt_y + 35)  # Center of the belt rectangle

        # Display belt resources
        belt_resources = env.belt_resources[belt]
        if belt_resources:
            for i, asteroid in enumerate(belt_resources[:3]):  # Display up to 3 resources
                resource_text = font.render(f"{asteroid['mineral']} ({asteroid['quantity']})", True, GREEN)
                screen.blit(resource_text, (belt_x + 10, belt_y + 70 + i * 20))
        else:
            empty_text = font.render("Empty", True, RED)
            screen.blit(empty_text, (belt_x + 10, belt_y + 70))

        belt_x += belt_spacing  # Increase horizontal spacing between belts

    # Draw stations
    station_x = system_x
    station_y = 500

    station_spacing = 250  # Spacing between stations
    for station in STATIONS:
        station_rect = pygame.Rect(station_x, station_y, 120, 50)
        pygame.draw.rect(screen, BLUE, station_rect)
        station_text = font.render(station, True, WHITE)
        screen.blit(station_text, (station_x + 10, station_y + 10))

        # Store the position of the station
        locations_positions[station] = (station_x + 60, station_y + 25)  # Center of the station rectangle

        # Display mineral prices under each station
        price_y = station_y + 60
        if station in STATION_MINERAL_PRICES:
            prices = STATION_MINERAL_PRICES[station]
            for mineral, price in prices.items():
                price_text = font.render(f"{mineral}: {price} ISK", True, WHITE)
                screen.blit(price_text, (station_x, price_y))
                price_y += 20

        # Display fuel price
        fuel_price = STATION_FUEL_PRICES.get(station, None)
        if fuel_price is not None:
            fuel_price_text = font.render(f"Fuel: {fuel_price} ISK/unit", True, WHITE)
            screen.blit(fuel_price_text, (station_x, price_y))
            price_y += 20
        else:
            no_fuel_text = font.render("No fuel available", True, RED)
            screen.blit(no_fuel_text, (station_x, price_y))
            price_y += 20

        station_x += station_spacing  # Increase horizontal spacing between stations

    # Draw the ship to the right of its current location
    ship_location = env.ship_state['position']
    if ship_location in locations_positions:
        ship_x, ship_y = locations_positions[ship_location]
        ship_offset_x = 70  # Horizontal offset to move the ship to the right
        ship_offset_y = 0   # Vertical offset (if needed)
        # Draw a simple triangle to represent the ship
        pygame.draw.polygon(
            screen, RED, [
                (ship_x + ship_offset_x, ship_y - 15 + ship_offset_y),     # Top point
                (ship_x - 10 + ship_offset_x, ship_y + 10 + ship_offset_y),  # Bottom left
                (ship_x + 10 + ship_offset_x, ship_y + 10 + ship_offset_y)   # Bottom right
            ]
        )

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

        # Updated action sequence for testing
        if step_number == 0:
            action = {'action_type': 0, 'target': 0, 'miner_id': None}  # Warp to Belt1
        elif step_number == 1:
            action = {'action_type': 1, 'target': 0, 'miner_id': 0}  # Mine first asteroid
        elif step_number == 2:
            action = {'action_type': 0, 'target': len(ASTEROID_BELTS), 'miner_id': None}  # Warp to Station1
        elif step_number == 3:
            action = {'action_type': 5, 'target': None, 'miner_id': None}  # Sell minerals at Station1
        elif step_number == 4:
            action = {'action_type': 6, 'target': None, 'miner_id': None}  # Buy Fuel at Station1
        elif step_number == 5:
            action = {'action_type': 0, 'target': 1, 'miner_id': None}  # Warp to Belt2
        elif step_number == 6:
            action = {'action_type': 1, 'target': 0, 'miner_id': 0}  # Mine first asteroid
        elif step_number == 7:
            action = {'action_type': 0, 'target': len(ASTEROID_BELTS) + 1, 'miner_id': None}  # Warp to Station2
        elif step_number == 8:
            action = {'action_type': 5, 'target': None, 'miner_id': None}  # Sell minerals at Station2
        elif step_number == 9:
            action = {'action_type': 6, 'target': None, 'miner_id': None}  # Buy Fuel at Station2
        else:
            action = {'action_type': 4, 'target': None, 'miner_id': None}  # Empty Cargo (for demonstration)

        # Get a description of the current action
        action_description = get_action_description(action)

        obs, reward, done, truncated, info = env.step(action)

        # Pass the action description to draw_environment
        draw_environment(screen, env, step_number, action_description)

        # Increment step counter
        step_number += 1

        # Control the frame rate
        clock.tick(2)  # 2 frames per second

    pygame.quit()
    sys.exit()

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
