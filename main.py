import time

from environments import SingleSystemSoloAgentEnv, ASTEROID_BELTS, STATIONS, MINERALS, run_environment
env = SingleSystemSoloAgentEnv()

obs, info = env.reset()

run_environment(env)