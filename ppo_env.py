import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class ModularMiningEnv(gym.Env):
    """
    A modular mining environment where complexity is controlled by a declarative configuration.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config=None):
        """
        env_config is a dictionary that defines the environment complexity and features.
        
        Example env_config:
        {
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
                    "fuel_price": None  # No fuel purchase at this stage
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
                "allow_wait_regen": True,  # If False, must buy fuel for energy
            },
            "cargo_capacity": 500,
            "reward_scale": 0.001,
            "max_steps": 100
        }
        
        You can increase complexity by adding more belts, stations, adjusting prices, enabling fuel purchase, etc.
        """
        super().__init__()
        
        # Set defaults if no config is given
        if env_config is None:
            env_config = {
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
                "max_steps": 100
            }

        self.config = env_config

        # Parse config
        self.num_belts = self.config["num_belts"]
        self.num_stations = self.config["num_stations"]
        self.belt_config = self.config["belt_config"]
        self.station_config = self.config["station_config"]
        
        # Energy config
        self.max_energy = self.config["energy_config"]["max_energy"]
        self.energy_regen_rate = self.config["energy_config"]["energy_regen_rate"]
        self.energy_costs = self.config["energy_config"]["energy_costs"]
        self.allow_wait_regen = self.config["energy_config"]["allow_wait_regen"]
        
        self.cargo_capacity = self.config["cargo_capacity"]
        self.reward_scale = self.config["reward_scale"]
        self.max_steps = self.config["max_steps"]

        # Construct environment layout
        self.belts = [bc["belt_name"] for bc in self.belt_config]
        self.stations = [sc["station_name"] for sc in self.station_config]
        self.locations = self.belts + self.stations
        
        # Minerals set derived from belt_config and station_config
        # We'll gather all mentioned minerals
        self.minerals = set()
        for bc in self.belt_config:
            self.minerals.update(bc["mineral_distribution"].keys())
        self.minerals = list(self.minerals)
        self.num_minerals = len(self.minerals)

        # Define action space:
        # Actions: Warp to each location + Mine + Sell + Buy Fuel + Wait
        # Warp actions = len(self.locations)
        # Then: Mine (1), Sell (1), Buy Fuel (1), Wait (1)
        # Total = len(self.locations) + 4
        # If you want more modularity, you can dynamically build this action space.
        self.action_map = []
        # Warp actions:
        for i, loc in enumerate(self.locations):
            self.action_map.append(("warp", loc))
        self.action_map.append(("mine", None))
        self.action_map.append(("sell", None))
        self.action_map.append(("buy_fuel", None))
        self.action_map.append(("wait", None))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation: cargo(num_minerals), position(one-hot), energy(1), balance(1)
        obs_size = self.num_minerals + len(self.locations) + 2
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        # Internal state
        self.step_count = 0
        self._reset_state()

    def _reset_state(self):
        self.step_count = 0
        # Generate asteroids
        self.belt_resources = {}
        for bc in self.belt_config:
            belt_name = bc["belt_name"]
            mineral_dist = bc["mineral_distribution"]
            self.belt_resources[belt_name] = self._generate_asteroids(mineral_dist)

        self.ship_state = {
            "position": self.stations[0],  # start at a station
            "cargo": np.zeros(self.num_minerals, dtype=np.int32),
            "account_balance": 0,
            "energy": self.max_energy
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action_idx):
        self.step_count += 1
        action_type, target = self.action_map[action_idx]

        reward = 0
        done = False
        truncated = False

        # Handle action
        if action_type == "warp":
            reward += self._warp(target)
        elif action_type == "mine":
            reward += self._mine()
        elif action_type == "sell":
            reward += self._sell()
        elif action_type == "buy_fuel":
            reward += self._buy_fuel()
        elif action_type == "wait":
            reward += self._wait()

        # Natural energy regeneration if allowed
        if self.allow_wait_regen:
            self.ship_state['energy'] = min(self.ship_state['energy'] + self.energy_regen_rate, self.max_energy)

        # Check termination
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}

        return obs, reward, done, truncated, info

    def _warp(self, location):
        cost = self.energy_costs.get('warp', 0)
        if self.ship_state['energy'] < cost:
            return -0.01  # penalty for not enough energy
        self.ship_state['energy'] -= cost
        self.ship_state['position'] = location
        return 0.0

    def _mine(self):
        cost = self.energy_costs.get('mine', 0)
        if self.ship_state['energy'] < cost:
            return -0.01
        
        if self.ship_state['position'] not in self.belts:
            return -0.01  # Can't mine outside belts

        belt_asteroids = self.belt_resources[self.ship_state['position']]
        if not belt_asteroids:
            return -0.005  # belt empty

        self.ship_state['energy'] -= cost
        asteroid = random.choice(belt_asteroids)
        available_capacity = self.cargo_capacity - np.sum(self.ship_state['cargo'])
        if available_capacity <= 0:
            return -0.01  # cargo full, no mining

        mined = min(asteroid['quantity'], available_capacity)
        mineral_idx = self.minerals.index(asteroid['mineral'])
        self.ship_state['cargo'][mineral_idx] += mined
        asteroid['quantity'] -= mined
        if asteroid['quantity'] <= 0:
            belt_asteroids.remove(asteroid)

        mineral_value = self._get_mineral_base_value(asteroid['mineral']) * mined
        return mineral_value * self.reward_scale

    def _sell(self):
        pos = self.ship_state['position']
        if pos not in self.stations:
            return -0.01
        
        total_qty = np.sum(self.ship_state['cargo'])
        if total_qty == 0:
            return -0.005

        station_info = self._get_station_info(pos)
        prices = station_info["mineral_prices"]

        total_sale = 0
        for i, qty in enumerate(self.ship_state['cargo']):
            if qty > 0:
                mineral = self.minerals[i]
                price = prices.get(mineral, self._get_mineral_base_value(mineral))
                total_sale += qty * price

        self.ship_state['account_balance'] += total_sale
        self.ship_state['cargo'] = np.zeros(self.num_minerals, dtype=np.int32)

        return total_sale * self.reward_scale

    def _buy_fuel(self):
        pos = self.ship_state['position']
        if pos not in self.stations:
            return -0.01

        station_info = self._get_station_info(pos)
        fuel_price = station_info["fuel_price"]
        if fuel_price is None:
            return -0.005  # no fuel here

        energy_needed = self.max_energy - self.ship_state['energy']
        if energy_needed <= 0:
            return -0.005  # full energy already

        total_cost = energy_needed * fuel_price
        if self.ship_state['account_balance'] < total_cost:
            return -0.01  # not enough money
        
        self.ship_state['account_balance'] -= total_cost
        self.ship_state['energy'] = self.max_energy
        return 0.0

    def _wait(self):
        cost = self.energy_costs.get('wait', 0)
        # If waiting doesn't cost energy, maybe apply a small negative reward to discourage too much waiting
        # Without wait regen, waiting might be pointless or minimal benefit
        # Adjust as needed
        if self.allow_wait_regen:
            # Extra energy is gained at the end of step anyway
            return -0.001
        else:
            # If no wait regen allowed, waiting might have no benefit
            return -0.001

    def _get_obs(self):
        # cargo normalized
        cargo_norm = self.ship_state['cargo'] / self.cargo_capacity
        pos_enc = np.zeros(len(self.locations), dtype=np.float32)
        pos_idx = self.locations.index(self.ship_state['position'])
        pos_enc[pos_idx] = 1.0
        energy_norm = self.ship_state['energy'] / self.max_energy
        # Normalize account balance by some factor, say 10000 or dynamically chosen
        balance_norm = np.tanh(self.ship_state['account_balance'] / 10000.0)

        obs = np.concatenate([cargo_norm, pos_enc, [energy_norm], [balance_norm]]).astype(np.float32)
        return obs

    def _get_mineral_base_value(self, mineral):
        # fallback if station doesn't specify price
        # This could be a constant baseline. Adjust as needed.
        return 10.0

    def _get_station_info(self, station_name):
        for sc in self.station_config:
            if sc["station_name"] == station_name:
                return sc
        return None

    def _generate_asteroids(self, mineral_distribution):
        asteroids = []
        # mineral_distribution is {mineral_name: (min_qty, max_qty)}
        for mineral, (low, high) in mineral_distribution.items():
            # number of asteroids could also vary
            # For simplicity, generate a few asteroids
            for _ in range(random.randint(3,5)):
                qty = random.randint(low, high)
                asteroids.append({"mineral": mineral, "quantity": qty})
        return asteroids
    
    def render(self):
        # Only print if render_mode is 'human'
        if self.render_mode == 'human':
            pos = self.ship_state['position']
            bal = self.ship_state['account_balance']
            eng = self.ship_state['energy']
            cargo_str = ", ".join([f"{mineral}:{qty}" for mineral, qty in zip(self.minerals, self.ship_state['cargo'])])
            
            print("------- Environment State -------")
            print(f"Position: {pos}")
            print(f"Balance: {bal}")
            print(f"Energy: {eng}/{self.max_energy}")
            print(f"Cargo: {cargo_str}")

            if pos in self.belts:
                # Show local asteroids
                belt_asteroids = self.belt_resources[pos]
                if belt_asteroids:
                    print("Asteroids here:")
                    for ast in belt_asteroids:
                        print(f"  {ast['mineral']} (Qty: {ast['quantity']})")
                else:
                    print("No asteroids at this belt.")

            elif pos in self.stations:
                # Show station prices
                station_info = self._get_station_info(pos)
                print("Station prices:")
                for mineral, price in station_info["mineral_prices"].items():
                    print(f"  {mineral}: {price} ISK/unit")
                
                if station_info["fuel_price"] is not None:
                    print(f"Fuel Price: {station_info['fuel_price']} ISK/unit")
                else:
                    print("No fuel available here.")

            print("--------------------------------")


    def close(self):
        pass
