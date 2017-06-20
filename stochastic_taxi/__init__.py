from gym.envs.registration import register

register(
    id='stochastic_taxi-v0',
    entry_point='gym_foo.envs:StochasticTaxiEnv',
)
register(
    id='stochastic_taxi-extrahard-v0',
    entry_point='gym_foo.envs:StochasticTaxiExtraHardEnv',
)
