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
MAX_ENERGY = 300
ENERGY_REGEN_RATE = 2  # Energy regenerated per step
ENERGY_COSTS = {
    'warp': 15,
    'mine': 10,
    'move': 5,
}

MAX_STEPS_PER_EPISODE = 150

# Define the list of possible actions
ACTIONS = [
    {'action_type': 0, 'target': 0, 'miner_id': None},  # Warp to Belt1
    {'action_type': 0, 'target': 1, 'miner_id': None},  # Warp to Belt2
    {'action_type': 0, 'target': 2, 'miner_id': None},  # Warp to Belt3
    {'action_type': 0, 'target': 3, 'miner_id': None},  # Warp to Station1
    {'action_type': 0, 'target': 4, 'miner_id': None},  # Warp to Station2
    {'action_type': 1, 'target': 0, 'miner_id': 0},     # Mine asteroid 0
    {'action_type': 1, 'target': 1, 'miner_id': 0},     # Mine asteroid 1
    {'action_type': 1, 'target': 2, 'miner_id': 0},     # Mine asteroid 2
    {'action_type': 5, 'target': None, 'miner_id': None},  # Sell minerals
    {'action_type': 6, 'target': None, 'miner_id': None},  # Buy Fuel
    {'action_type': 4, 'target': None, 'miner_id': None},  # Empty Cargo
    {'action_type': 7, 'target': None, 'miner_id': None},  # Wait
    # Add more actions as needed
]

class SingleSystemSoloAgentEnv(gym.Env):
    def __init__(self):
        super(SingleSystemSoloAgentEnv, self).__init__()

        # Update the action space to be discrete
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation space dimensions
        num_minerals = len(MINERALS)
        num_positions = len(ASTEROID_BELTS + STATIONS)
        num_miners = len(np.zeros(2, dtype=np.int8))
        observation_size = num_minerals + num_positions + num_miners + 1 + 1 + 1  # cargo + position + miners + energy + account_balance + local_objects_count

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32
        )

        # System-wide state: Map each belt to its resources
        self.belt_resources = {belt: self.generate_asteroids() for belt in ASTEROID_BELTS}

        # Initialize state
        self.reset()
        self.step_count = 0  # Initialize step counter

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0  # Reset step counter

        # Regenerate asteroid belts
        self.belt_resources = {belt: self.generate_asteroids() for belt in ASTEROID_BELTS}

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
            'energy': MAX_ENERGY,
        }

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action_idx):
        self.step_count += 1  # Increment step counter
        done = False
        reward = 0
        info = {}

        # Get the action dictionary from the action index
        action = ACTIONS[action_idx]

        action_type = action['action_type']
        target = action.get('target', None)
        miner_id = action.get('miner_id', None)

        # Energy cost mapping
        action_energy_costs = {
            0: ENERGY_COSTS['warp'],   # Warp
            1: ENERGY_COSTS['mine'],   # Mine
            2: ENERGY_COSTS['move'],   # Move
            3: 0,                      # Dock (no energy cost)
            4: 0,                      # Empty Cargo (no energy cost)
            5: 0,                      # Sell Minerals (no energy cost)
            6: 0,                      # Buy Fuel (no energy cost),
        }

        energy_cost = action_energy_costs.get(action_type, 0)

        # Check if there's enough energy
        if self.ship_state['energy'] >= energy_cost:
            # Deduct energy cost
            self.ship_state['energy'] -= energy_cost

            # Perform the action and collect rewards or penalties
            if action_type == 0:  # Warp
                warp_reward = self._warp(target)
                reward += warp_reward
            elif action_type == 1:  # Mine
                mine_reward = self._mine(miner_id, target)
                reward += mine_reward
            elif action_type == 2:  # Move
                move_reward = self._move(target)
                reward += move_reward
            elif action_type == 3:  # Dock
                dock_reward = self._dock(target)
                reward += dock_reward
            elif action_type == 4:  # Empty Cargo
                empty_cargo_reward = self._empty_cargo()
                reward += empty_cargo_reward
            elif action_type == 5:  # Sell Minerals
                sell_reward = self._sell_minerals()
                reward += sell_reward
            elif action_type == 6:  # Buy Fuel
                buy_fuel_reward = self._buy_fuel()
                reward += buy_fuel_reward
            elif action_type == 7:  # Wait
                # print("Waiting to regenerate energy.")
                # Regenerate more energy when waiting
                self.ship_state['energy'] = min(self.ship_state['energy'] + ENERGY_REGEN_RATE * 2, MAX_ENERGY)
                reward -= 0.5  # Small penalty for waiting
            else:
                print("Unknown action type.")
                reward -= 5  # Penalty for invalid action type
        else:
            print("Not enough energy to perform this action.")
            reward -= 20  # Penalty for insufficient energy

        # Regenerate energy
        self.ship_state['energy'] = min(self.ship_state['energy'] + ENERGY_REGEN_RATE, MAX_ENERGY)

        # Regenerate resources occasionally
        if random.random() < 0.03:  # 10% chance to regenerate in this step
            self._regenerate_belt_resources()

        observation = self._get_obs()

        # Check for episode termination
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            done = True

        return observation, reward, done, False, info

    def _get_obs(self):
        # Encode cargo
        cargo = self.ship_state['cargo']

        # Encode position as one-hot vector
        positions = ASTEROID_BELTS + STATIONS
        position_enc = np.zeros(len(positions))
        position_idx = positions.index(self.ship_state['position'])
        position_enc[position_idx] = 1

        # Encode miners' status
        miners_status = self.ship_state['miners']

        # Encode energy level (normalize between 0 and 1)
        energy = np.array([self.ship_state['energy'] / MAX_ENERGY])

        # Encode account balance (you can normalize or cap if necessary)
        account_balance = np.array([self.ship_state['account_balance']])

        # Encode local objects (e.g., count of asteroids)
        local_objects_count = np.array([len(self.local_state['objects'])])

        # Concatenate all features into a single flat array
        observation = np.concatenate([
            cargo,
            position_enc,
            miners_status,
            energy,
            account_balance,
            local_objects_count
        ])

        return observation.astype(np.float32)

    def _buy_fuel(self):
        """Allows the ship to purchase fuel (energy) at the current station."""
        current_station = self.ship_state['position']
        reward = 0
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
                        # No explicit reward for successful fuel purchase
                    else:
                        print("Not enough ISK to buy fuel.")
                        reward -= 10  # Penalty for insufficient funds
                else:
                    print("Energy is already full.")
                    reward -= 5  # Penalty for unnecessary action
            else:
                print("No fuel available at this station.")
                reward -= 5  # Penalty for invalid action at this station
        else:
            print("Cannot buy fuel outside of a station.")
            reward -= 10  # Penalty for invalid location
        return reward

    def render(self):
        print("Current Position:", self.ship_state['position'])
        print("Cargo Hold:", self.ship_state['cargo'])
        print("Miners Status:", self.ship_state['miners'])
        print("Account Balance:", self.ship_state['account_balance'], "ISK")
        print("Energy Level:", self.ship_state['energy'], "/", MAX_ENERGY)
        print("Local Objects:", self.local_state['objects'])

    def _warp(self, target_id):
        reward = 0
        if target_id is not None and target_id < len(self.system_state['locations']):
            location = self.system_state['locations'][target_id]
            self.ship_state['position'] = location
            # print(f"Warped to {self.ship_state['position']}")
            if location in ASTEROID_BELTS:
                self.local_state['objects'] = self.belt_resources[location]
                reward += 5
            else:
                self.local_state['objects'] = []
        else:
            print("Invalid warp target.")
            reward -= 10  # Penalty for invalid warp target
        return reward

    def _mine(self, miner_id, target_id):
        """Allows the ship to mine asteroids."""
        reward = 0
        if self.ship_state['miners'][miner_id] == 0:
            if target_id < len(self.local_state['objects']):
                asteroid = self.local_state['objects'][target_id]
                self.ship_state['miners'][miner_id] = 1
                mineral_idx = list(MINERALS.keys()).index(asteroid['mineral'])
                self.ship_state['cargo'][mineral_idx] += asteroid['quantity']
                # print(f"Mined {asteroid['quantity']} units of {asteroid['mineral']}")
                self.local_state['objects'].pop(target_id)
                self.belt_resources[self.ship_state['position']] = self.local_state['objects']
                # Reset miner status back to inactive
                self.ship_state['miners'][miner_id] = 0  # Reset miner status here
                # Provide a reward proportional to the mineral value
                mineral_value = MINERALS[asteroid['mineral']] * asteroid['quantity']
                reward += mineral_value * 0.1  # Adjust the scaling factor as needed
            else:
                print("Invalid mining target.")
                reward -= 5  # Penalty for invalid target
        else:
            print("Miner is already active.")
            reward -= 10  # Penalty for attempting to mine with an active miner
        return reward

    def _move(self, target):
        reward = 0
        print(f"Moving towards {target}")
        # Implement movement logic if necessary
        return reward

    def _dock(self, target_id):
        reward = 0
        if target_id is not None and target_id < len(STATIONS):
            self.ship_state['position'] = STATIONS[target_id]
            print(f"Docked at {self.ship_state['position']}")
        else:
            print("Invalid docking target.")
            reward -= 10  # Penalty for invalid docking target
        return reward

    def _empty_cargo(self):
        reward = 0
        if np.any(self.ship_state['cargo'] > 0):
            
            total = 0
            prices = STATION_MINERAL_PRICES.get('Station1', MINERALS)
            for idx, quantity in enumerate(self.ship_state['cargo']):
                if quantity > 0:
                    mineral = list(MINERALS.keys())[idx]
                    price = prices.get(mineral, MINERALS[mineral])
                    total += quantity * price
                    
            reward -= 20  # Small penalty for discarding cargo
            
            self.ship_state['cargo'] = np.zeros(len(MINERALS), dtype=np.int32)
            print("Emptied cargo hold.", 'Penalty: ', reward)
        else:
            print("Cargo hold is already empty.")
            reward -= 2  # Penalty for unnecessary action
        return reward

    def _sell_minerals(self):
        """Allows the ship to sell minerals at the current station."""
        total_earnings = 0
        reward = 0
        current_station = self.ship_state['position']
        if current_station in STATIONS:
            prices = STATION_MINERAL_PRICES.get(current_station, MINERALS)
            print(f"Selling minerals at {current_station} with station-specific prices.")

            has_sold = False
            for idx, quantity in enumerate(self.ship_state['cargo']):
                if quantity > 0:
                    mineral = list(MINERALS.keys())[idx]
                    price = prices.get(mineral, MINERALS[mineral])
                    total_sale = quantity * price
                    total_earnings += total_sale
                    print(f"Sold {quantity} units of {mineral} for {total_sale} ISK at {current_station}")
                    self.ship_state['cargo'][idx] = 0  # Empty the cargo for this mineral
                    has_sold = True

            if has_sold:
                # Update account balance
                self.ship_state['account_balance'] += total_earnings
                print(f"Total Earnings: {total_earnings} ISK")
                print(f"Updated Account Balance: {self.ship_state['account_balance']} ISK")
                reward += total_earnings * 0.1  # Positive reward proportional to earnings
            else:
                print("No minerals to sell.")
                reward -= 10  # Penalty for attempting to sell with empty cargo
        else:
            print("Cannot sell minerals outside of a station.")
            reward -= 10  # Penalty for invalid location

        return reward

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

    def get_valid_actions(self):
        valid_action_indices = []
        position = self.ship_state['position']
        cargo_full = np.any(self.ship_state['cargo'] > 0)
        energy = self.ship_state['energy']
        local_objects = self.local_state['objects']

        for idx, action in enumerate(ACTIONS):
            action_type = action['action_type']
            action_energy_costs = {
                0: ENERGY_COSTS['warp'],   # Warp
                1: ENERGY_COSTS['mine'],   # Mine
                2: ENERGY_COSTS['move'],   # Move
                3: 0,                      # Dock
                4: 0,                      # Empty Cargo
                5: 0,                      # Sell Minerals
                6: 0,                      # Buy Fuel
            }
            energy_cost = action_energy_costs.get(action_type, 0)
            if energy < energy_cost:
                continue  # Skip actions that cannot be performed due to insufficient energy

            is_valid = True
            if action_type == 0:  # Warp
                target = action.get('target')
                if target >= len(self.system_state['locations']):
                    is_valid = False
            elif action_type == 1:  # Mine
                if position not in ASTEROID_BELTS:
                    is_valid = False
                else:
                    target = action.get('target')
                    if target >= len(local_objects):
                        is_valid = False
            elif action_type == 5:  # Sell Minerals
                if position not in STATIONS or not cargo_full:
                    is_valid = False
            elif action_type == 6:  # Buy Fuel
                if position not in STATIONS:
                    is_valid = False
            elif action_type == 4:  # Empty Cargo
                if not cargo_full:
                    is_valid = False

            if is_valid:
                valid_action_indices.append(idx)

        # Add 'Wait' action as always valid
        wait_action_idx = next((idx for idx, action in enumerate(ACTIONS) if action['action_type'] == 7), None)
        if wait_action_idx is not None:
            valid_action_indices.append(wait_action_idx)

        return valid_action_indices

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

def draw_environment(screen, env, step_number, current_action, episode_number):
    """Draw the current state of the environment."""
    # Clear the screen
    screen.fill(BLACK)

    # Title with Episode and Step numbers
    font = pygame.font.Font(None, 36)
    title_text = font.render(
        f"Episode {episode_number} - Step {step_number} - EVE Mining Environment",
        True,
        WHITE
    )
    screen.blit(title_text, (20, 20))

    # Display current action
    action_text = font.render(f"Current Action: {current_action}", True, YELLOW)
    screen.blit(action_text, (20, 60))

    # Draw account balance
    account_balance_text = font.render(
        f"Account Balance: {env.ship_state['account_balance']} ISK",
        True,
        GREEN
    )
    screen.blit(account_balance_text, (20, 100))

    # Draw energy level
    energy_text = font.render(
        f"Energy Level: {env.ship_state['energy']} / {MAX_ENERGY}",
        True,
        RED
    )
    screen.blit(energy_text, (20, 140))

    # Draw cargo hold
    cargo_text = "\n".join([
        f"{mineral}: {qty}" 
        for mineral, qty in zip(MINERALS.keys(), env.ship_state['cargo'])
    ])
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
            obj_text = font.render(
                f"{i}: {obj['mineral']} ({obj['quantity']})",
                True,
                WHITE
            )
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
                resource_text = font.render(
                    f"{asteroid['mineral']} ({asteroid['quantity']})",
                    True,
                    GREEN
                )
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
                (ship_x + ship_offset_x, ship_y - 15 + ship_offset_y),       # Top point
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

    episode_number = 1  # Initialize episode number

    while running:
        step_number = 0
        obs, info = env.reset()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            # Define action based on step_number or any other logic
            if step_number == 0:
                action = {'action_type': 0, 'target': 0, 'miner_id': None}  # Warp to Belt1
            elif step_number == 1:
                action = {'action_type': 1, 'target': 0, 'miner_id': 0}    # Mine first asteroid
            elif step_number == 2:
                action = {'action_type': 0, 'target': len(ASTEROID_BELTS), 'miner_id': None}  # Warp to Station1
            elif step_number == 3:
                action = {'action_type': 5, 'target': None, 'miner_id': None}  # Sell minerals at Station1
            elif step_number == 4:
                action = {'action_type': 6, 'target': None, 'miner_id': None}  # Buy Fuel at Station1
            elif step_number == 5:
                action = {'action_type': 0, 'target': 1, 'miner_id': None}    # Warp to Belt2
            elif step_number == 6:
                action = {'action_type': 1, 'target': 0, 'miner_id': 0}    # Mine first asteroid
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

            # Perform the action
            obs, reward, done, truncated, info = env.step(action)

            # Pass the episode_number to draw_environment
            draw_environment(screen, env, step_number, action_description, episode_number)

            # Increment step counter
            step_number += 1

            if done:
                episode_number += 1  # Increment episode number
                break

            # Control the frame rate
            clock.tick(1)  # 1 frame per second

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
