import numpy as np
import gymnasium as gym
from gymnasium import spaces as sp

class MeleeEnv(gym.Env):
    def __init__(self, n_projectiles:int):
        projectiles = {"n_active_projectiles": sp.Box(low=0, high=n_projectiles, shape=(1,), dtype=np.int8)}

        for i in range(n_projectiles):
            projectiles_template = {f"{i}pos_x": sp.Box(low=-255.0, high=255, shape=(1,0), dtype=np.float64), 
                                    f"{i}pos_y": sp.Box(low=-255.0, high=255, shape=(1,0), dtype=np.float64), 
                                    f"{i}speed_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64), 
                                    f"{i}speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                                    f"{i}owner": sp.Box(low=-1, high=127, shape=(1,), dtype=np.int8), 
                                    f"{i}type": sp.Box(low=0, high=512, shape=(1,), dtype=np.int16), 
                                    f"{i}frame": sp.Box(low=-16384, high=32767, shape=(1,), dtype=np.int32), 
                                    f"{i}subtype": sp.Box(low=-16384, high=65536, shape=(1,), dtype=np.int32)}
            projectiles.update(projectiles_template)

        self.observation_space = sp.Dict({
            "frame": sp.Box(low=-16384, high=32767, shape=(1,), dtype=np.int32),
            "total_projectiles": sp.Discrete(n_projectiles+1),
            "stage": sp.Dict({
                "blastzones": sp.Box(low=np.array([-255.0, -255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0, 255.0]), dtype=np.float32),
                "edge": sp.Box(low=0.0, high=100.0, dtype=np.float64),
                "edge_ground": sp.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float64),
                "right_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float64),
                "left_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float64),
                "top_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float64)
            }),
            "agent": sp.Dict({
                "position": sp.Box(low=np.array([-255.0, -255.0]), high=np.array([255.0, 255.0]), dtype=np.float64),
                "percent": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "shield_strenght": sp.Box(low=0.0, high=60.0, shape=(1,), dtype=np.float64),
                "is_powershield": sp.Discrete(2),
                "stock": sp.Box(low=0, high=4, shape=(1,), dtype=np.int8),
                "facing": sp.Discrete(2),
                "action_frame": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "action_id": sp.Box(low=0, high=65536, shape=(1,), dtype=np.int32),
                "invulnerable": sp.Discrete(2),
                "invulnerability_left": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "hitlag_left": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "hitstun_left": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "jumps_left": sp.Box(low=0, high=127, shape=(1,), dtype=np.int8),
                "on_ground": sp.Discrete(2),
                "speed_air_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "speed_attack_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "speed_attack_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "is_moonwalking": sp.Discrete(2),
                "ecb_top_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_top_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_bottom_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64)
            }),
            "agent": sp.Dict({
                "position": sp.Box(low=np.array([-255.0, -255.0]), high=np.array([255.0, 255.0]), dtype=np.float64),
                "percent": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "shield_strenght": sp.Box(low=0.0, high=60.0, shape=(1,), dtype=np.float64),
                "is_powershield": sp.Discrete(2),
                "stock": sp.Box(low=0, high=4, shape=(1,), dtype=np.int8),
                "facing": sp.Discrete(2),
                "action_frame": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "action_id": sp.Box(low=0, high=65536, shape=(1,), dtype=np.int32),
                "invulnerable": sp.Discrete(2),
                "invulnerability_left": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "hitlag_left": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "hitstun_left": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "jumps_left": sp.Box(low=0, high=127, shape=(1,), dtype=np.int8),
                "on_ground": sp.Discrete(2),
                "speed_air_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "speed_attack_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "speed_attack_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "is_moonwalking": sp.Discrete(2),
                "ecb_top_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_top_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_bottom_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64)
            }),
            "projectiles": sp.Dict(projectiles)
        })