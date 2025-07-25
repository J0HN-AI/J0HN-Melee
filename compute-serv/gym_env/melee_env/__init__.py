from gymnasium.envs.registration import register

register(
    id="melee_env/MeleeEnv-v0",
    entry_point="melee_env.envs:MeleeEnv",
)