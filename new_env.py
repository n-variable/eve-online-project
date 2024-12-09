import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Constants
MINERALS = {
    'Veldspar': 10,
    'Scordite': 20,
    'Pyroxeres': 30,
}

ASTEROID_BELTS = ['Belt1', 'Belt2', 'Belt3']
STATIONS = ['Station1', 'Station2']

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

STATION_FUEL_PRICES = {
    'Station1': 5,
    'Station2': 6,
}

MAX_ENERGY = 300
ENERGY_REGEN_RATE = 1
ENERGY_COSTS = {
    'warp': 20,
    'mine': 15,
    'wait': 0,   # Wait now costs no energy, only opportunity cost
}

MAX_STEPS_PER_EPISODE = 100
MAX_CARGO_CAPACITY = 500

# Define actions in a simpler manner:
# 0: Warp Belt1
# 1: Warp Belt2
# 2: Warp Belt3
# 3: Warp Station1
# 4: Warp Station2
# 5: Mine
# 6: Sell minerals (if at station)
# 7: Buy fuel (if at station)
# 8: Wait
ACTIONS = list(range(9))

class PPOFriendlyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # Observation:
        # cargo: 3 mineral counts
        # position: one-hot of length 5 (3 belts + 2 stations)
        # energy: normalized [0,1]
        # account_balance: normalized (divide by some constant, e.g. 10,000)
        # local_object_count: normalized count (e.g., divide by max possible?)
        self.num_minerals = len(MINERALS)
        self.positions = ASTEROID_BELTS + STATIONS
        self.obs_size = self.num_minerals + len(self.positions) + 2  # cargo(3) + pos(5) + energy(1) + balance(1)
        
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=np.float32)

        self.max_balance_scale = 10000.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.belt_resources = {belt: self.generate_asteroids() for belt in ASTEROID_BELTS}
        self.ship_state = {
            'cargo': np.zeros(self.num_minerals, dtype=np.int32),
            'position': 'Station1',
            'account_balance': 0,
            'energy': MAX_ENERGY
        }
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False
        truncated = False

        # Process action
        if action in [0,1,2,3,4]:  # Warp
            target_id = action
            reward += self._warp(target_id)
        elif action == 5:  # Mine
            reward += self._mine()
        elif action == 6:  # Sell
            reward += self._sell()
        elif action == 7:  # Buy fuel
            reward += self._buy_fuel()
        elif action == 8:  # Wait
            # Regenerate a bit more energy if waiting to encourage a strategy
            self.ship_state['energy'] = min(self.ship_state['energy'] + 2*ENERGY_REGEN_RATE, MAX_ENERGY)
            reward -= 0.01  # Slight penalty for waiting
        
        # Natural energy regeneration
        self.ship_state['energy'] = min(self.ship_state['energy'] + ENERGY_REGEN_RATE, MAX_ENERGY)

        # End of episode condition
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            truncated = True

        # Compute observation
        obs = self._get_obs()
        info = {}
        
        return obs, reward, done, truncated, info

    def _get_obs(self):
        cargo = self.ship_state['cargo']  # large, normalize by MAX_CARGO_CAPACITY
        cargo_norm = cargo / MAX_CARGO_CAPACITY

        # Position one-hot
        pos_enc = np.zeros(len(self.positions), dtype=np.float32)
        pos_idx = self.positions.index(self.ship_state['position'])
        pos_enc[pos_idx] = 1.0

        # Normalize energy
        energy_norm = self.ship_state['energy'] / MAX_ENERGY

        # Normalize account balance
        balance_norm = min(self.ship_state['account_balance'] / self.max_balance_scale, 1.0)

        obs = np.concatenate([
            cargo_norm,
            pos_enc,
            [energy_norm],
            [balance_norm]
        ]).astype(np.float32)

        return obs

    def _warp(self, target_id):
        cost = ENERGY_COSTS['warp']
        if self.ship_state['energy'] < cost:
            return -0.1  # Not enough energy penalty
        self.ship_state['energy'] -= cost
        new_pos = self.positions[target_id]
        self.ship_state['position'] = new_pos
        # Warping to a belt might have potential for future mining (slight reward)
        # Warping to a station is neutral.
        return 0.01 if new_pos in ASTEROID_BELTS else 0

    def _mine(self):
        cost = ENERGY_COSTS['mine']
        if self.ship_state['energy'] < cost:
            return -0.1

        position = self.ship_state['position']
        if position not in ASTEROID_BELTS:
            return -0.1  # Can't mine outside belts

        belt_asteroids = self.belt_resources[position]
        if not belt_asteroids:
            return -0.05  # Nothing to mine

        self.ship_state['energy'] -= cost
        asteroid = random.choice(belt_asteroids)
        available_capacity = MAX_CARGO_CAPACITY - np.sum(self.ship_state['cargo'])
        if available_capacity <= 0:
            return -0.1  # Full cargo

        mined = min(asteroid['quantity'], available_capacity)
        mineral_idx = list(MINERALS.keys()).index(asteroid['mineral'])
        self.ship_state['cargo'][mineral_idx] += mined
        asteroid['quantity'] -= mined
        if asteroid['quantity'] <= 0:
            belt_asteroids.remove(asteroid)

        # Reward scaled down
        mineral_value = MINERALS[asteroid['mineral']] * mined
        return 0.001 * mineral_value  # Scaled reward

    def _sell(self):
        pos = self.ship_state['position']
        if pos not in STATIONS:
            return -0.1
        if np.sum(self.ship_state['cargo']) == 0:
            return -0.05  # Nothing to sell

        prices = STATION_MINERAL_PRICES[pos]
        total_sale = 0
        for i, qty in enumerate(self.ship_state['cargo']):
            if qty > 0:
                mineral = list(MINERALS.keys())[i]
                price = prices[mineral]
                total_sale += qty * price

        self.ship_state['account_balance'] += total_sale
        self.ship_state['cargo'] = np.zeros(self.num_minerals, dtype=np.int32)
        # Scaled reward
        return 0.001 * total_sale

    def _buy_fuel(self):
        pos = self.ship_state['position']
        if pos not in STATIONS:
            return -0.1

        fuel_price = STATION_FUEL_PRICES[pos]
        energy_needed = MAX_ENERGY - self.ship_state['energy']
        if energy_needed <= 0:
            # Already full energy
            return -0.05

        total_cost = energy_needed * fuel_price
        if self.ship_state['account_balance'] < total_cost:
            return -0.1  # Not enough funds
        # Buy fuel
        self.ship_state['account_balance'] -= total_cost
        self.ship_state['energy'] = MAX_ENERGY
        return 0.0  # Neutral reward, but benefits future actions

    def generate_asteroids(self):
        asteroids = []
        for _ in range(random.randint(3, 5)):
            mineral = random.choice(list(MINERALS.keys()))
            quantity = random.randint(50, 200)
            asteroids.append({'mineral': mineral, 'quantity': quantity})
        return asteroids

    def render(self):
        if self.render_mode == 'human':
            print(f"Pos: {self.ship_state['position']}, Bal: {self.ship_state['account_balance']}, Energy: {self.ship_state['energy']}, Cargo: {self.ship_state['cargo']}")

    def close(self):
        pass
