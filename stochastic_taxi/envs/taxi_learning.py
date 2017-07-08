import time, random, math
import numpy as np
import gym

GAME = 'Taxi-v2'
env = gym.make(GAME)
env.seed(1995)

RECORD = None
MAX_EPISODES = 100001
MAX_STEPS = env.spec.timestep_limit     # 100 for FrozenLake v0
EPSILON = 1
DISCOUNT = 0.99
LEARNING_RATE = 0.01