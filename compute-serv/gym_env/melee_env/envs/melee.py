import numpy as np
import gymnasium as gym
from gymnasium import spaces as sp
import socket
import time
import struct
from halo import Halo
from random import randint
import math

class tcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class match_maker:
    def __init__(self, config:dict, rank:int):
        self.config = config
        self.rank = rank
        self.match_history = {
            "n_games": 0,
            "victory": 0,
            "defeat": 0,
            "stage": self._get_stage(config["instances"][f"{self.rank}"]["stage"]),
            "agent_character": self._get_player(config["instances"][f"{self.rank}"]["agent_character"]),
            "cpu_character": self._get_player(config["instances"][f"{self.rank}"]["cpu_character"]),
            "cpu_level": config["instances"][f"{self.rank}"]["cpu_level"],
        }

    def register_match(self, winner:str):
        if winner == "agent":
            self.match_history.update({
                "n_games": self.match_history["n_games"] + 1,
                "victory": self.match_history["victory"] + 1,
            })
        else:
            self.match_history.update({
                "n_games": self.match_history["n_games"] + 1,
                "defeat": self.match_history["defeat"] + 1,
            })
    
    def new_match(self):
        stage_change_rate = self.config["instances"][f"{self.rank}"]["stage_change_rate"]
        agent_character_change_rate = self.config["instances"][f"{self.rank}"]["agent_character_change_rate"]
        cpu_character_change_rate = self.config["instances"][f"{self.rank}"]["cpu_character_change_rate"]
        minimum_games_berfore_changing_cpu_level = self.config["instances"][f"{self.rank}"]["minimum_games_berfore_changing_cpu_level"]
        cpu_level_progression_rate = self.config["instances"][f"{self.rank}"]["cpu_level_progression_rate"]
        max_cpu_level =  self.config["instances"][f"{self.rank}"]["max_cpu_level"]
        
        if stage_change_rate != 0 and self.match_history["n_games"] % stage_change_rate == 0 and self.match_history["n_games"] != 0:
            random_stage = randint(0, 5)
            if random_stage == self.match_history["stage"]:
                self.match_history.update({ "stage": (random_stage + 1) % 6 })
            else:
                self.match_history.update({ "stage": random_stage })

        if agent_character_change_rate != 0 and self.match_history["n_games"] % agent_character_change_rate == 0 and self.match_history["n_games"] != 0:
            random_character = randint(0, 24)
            if random_character == self.match_history["agent_character"]:
                self.match_history.update({ "agent_character": (random_character + 1) % 25 })
            else:
                self.match_history.update({ "agent_character": random_character })
        
        if cpu_character_change_rate != 0 and self.match_history["n_games"] % cpu_character_change_rate == 0 and self.match_history["n_games"] != 0:
            random_character = randint(0, 24)
            if random_character == self.match_history["cpu_character"]:
                self.match_history.update({ "cpu_character": (random_character + 1) % 25 })
            else:
                self.match_history.update({ "cpu_character": random_character })

        if self.match_history["n_games"] % minimum_games_berfore_changing_cpu_level == 0 and self.match_history["cpu_level"] < max_cpu_level and self.match_history["n_games"] != 0:
            if self.match_history["defeat"] == 0:
                if self.match_history["victory"] > cpu_level_progression_rate:
                    self.match_history.update({ "cpu_level": self.match_history["cpu_level"] + 1 })
                else:
                    if (self.match_history["victory"] / self.match_history["defeat"]) > cpu_level_progression_rate:
                        self.match_history.update({ "cpu_level": self.match_history["cpu_level"] + 1 })

        return (self.match_history["stage"], self.match_history["agent_character"], self.match_history["cpu_character"], self.match_history["cpu_level"])
        
    def _get_stage(self, stage):
        match stage:
            case "BATTLEFIELD":
                return 0x0
            case "FINAL_DESTINATION":
                return 0x1
            case "DREAMLAND":
                return 0x2
            case "FOUNTAIN_OF_DREAMS":
                return 0x3
            case "POKEMON_STADIUM":
                return 0x4
            case "YOSHIS_STORY":
                return 0x5
    
    def _get_player(self, str_player):
        match str_player:
            case "DOC":
                return 0x00
            case "MARIO":
                return 0x01
            case "LUIGI":
                return 0x02
            case "BOWSER":
                return 0x03
            case "PEACH":
                return 0x04
            case "YOSHI":
                return 0x05
            case "DK":
                return 0x06
            case "CPTFALCON":
                return 0x07
            case "GANONDORF":
                return 0x08
            case "FALCO":
                return 0x09
            case "FOX":
                return 0x0a
            case "NESS":
                return 0x0b
            case "POPO":
                return 0x0c
            case "KIRBY":
                return 0x0d
            case "SAMUS":
                return 0x0e
            case "ZELDA":
                return 0x0f
            case "LINK":
                return 0x10
            case "YLINK":
                return 0x11
            case "PICHU":
                return 0x12
            case "PIKACHU":
                return 0x13
            case "JIGGLYPUFF":
                return 0x14
            case "MEWTWO":
                return 0x15
            case "GAMEANDWATCH":
                return 0x16
            case "MARTH":
                return 0x17
            case "ROY":
                return 0x18

class MeleeEnv(gym.Env):
    def __init__(self, config:dict, rank:int, debug:bool = False):
        self.config = config
        self.rank = rank
        self.debug = debug
        self.n_projectiles = config["training-config"]["n_projectiles"]
        self.match_maker = match_maker(config, rank)

        self.observation_space = sp.Dict({
            "frame": sp.Box(low=-16384, high=32767, shape=(1,), dtype=np.int32),
            "total_projectiles": sp.Discrete(self.n_projectiles+1),
            "stage": sp.Dict({
                "stage_id": sp.Discrete(6),
                "blastzones": sp.Box(low=np.array([-255.0, -255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0, 255.0]), dtype=np.float32),
                "edge": sp.Box(low=np.array([-100.0, -100.0]), high=np.array([100.0, 100.0]), dtype=np.float32),
                "edge_ground": sp.Box(low=np.array([-100.0, -100.0]), high=np.array([100.0, 100.0]), dtype=np.float32),
                "right_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float32),
                "left_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float32),
                "top_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float32)
            }),
            "agent": sp.Dict({
                "character": sp.Discrete(25),
                "position": sp.Box(low=np.array([-255.0, -255.0]), high=np.array([255.0, 255.0]), dtype=np.float32),
                "percent": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "shield_strenght": sp.Box(low=0.0, high=60.0, shape=(1,), dtype=np.float32),
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
                "speed_air_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "speed_attack_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "speed_attack_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "is_moonwalking": sp.Discrete(2),
                "ecb_top_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_top_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_bottom_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_bottom_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_left_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_left_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_right_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_right_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32)
            }),
            "cpu": sp.Dict({
                "character": sp.Discrete(25),
                "position": sp.Box(low=np.array([-255.0, -255.0]), high=np.array([255.0, 255.0]), dtype=np.float32),
                "percent": sp.Box(low=0, high=32767, shape=(1,), dtype=np.int16),
                "shield_strenght": sp.Box(low=0.0, high=60.0, shape=(1,), dtype=np.float32),
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
                "speed_air_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "speed_attack_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "speed_attack_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "is_moonwalking": sp.Discrete(2),
                "ecb_top_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_top_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_bottom_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_bottom_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_left_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_left_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_right_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                "ecb_right_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32)
            }),
            "projectiles": sp.Dict(self._make_projectiles_dict(self.n_projectiles))
        })

        #Buttons: A B X Y Z DPAD_UP DPAD_DOWN DPAD_LEFT DPAD_RIGHT MAIN_X MAIN_Y C_X C_Y LEFT_TRIGGER RIGHT_TRIGGER
        #Indexes: 0 1 2 3 4    5       6         7         8          9     10   11  12      13            14
        self.action_space = sp.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype=np.float16)
        
        self._connect_to_router()
        self._connect_to_logger()

    def _action_to_controller(self, actions):
        digital_buttons = np.array(np.round(actions[:9], decimals=0), dtype=np.int8).tolist()
        analog_buttons = actions[9:].tolist()

        return digital_buttons + analog_buttons
    
    def _clamp(self, nb, nb_min, nb_max):
        return max(min(nb_max, nb), nb_min)

    def _make_projectiles_dict(self, n_projectiles:int):
        projectiles = {"n_active_projectiles": sp.Box(low=0, high=n_projectiles, shape=(1,), dtype=np.int8)}

        for i in range(n_projectiles):
            projectiles_template = {f"{i}pos_x": sp.Box(low=-255.0, high=255, shape=(1,), dtype=np.float32), 
                                    f"{i}pos_y": sp.Box(low=-255.0, high=255, shape=(1,), dtype=np.float32), 
                                    f"{i}speed_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32), 
                                    f"{i}speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float32),
                                    f"{i}owner": sp.Box(low=-1, high=127, shape=(1,), dtype=np.int8), 
                                    f"{i}type": sp.Box(low=0, high=512, shape=(1,), dtype=np.int16), 
                                    f"{i}frame": sp.Box(low=-16384, high=32767, shape=(1,), dtype=np.int32), 
                                    f"{i}subtype": sp.Box(low=-16384, high=65536, shape=(1,), dtype=np.int32)}
            projectiles.update(projectiles_template)
        
        return projectiles
    
    def _connect_to_router(self):
        router_port = self.config["network-config"]["router_port"]
        env_base_port = self.config["network-config"]["envs_base_port"]
        timeout = self.config["network-config"]["envs_connection_timeout"]
        
        action_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        action_sock.bind(("127.0.0.1", env_base_port + self.rank))
        
        now = time.time()
        while True:
            try:
                action_sock.connect(("127.0.0.1", router_port))
                break
            except socket.error:
                if time.time() > (now + timeout):
                    print(f"{tcolors.BOLD}{tcolors.FAIL}Unable to connect to router{tcolors.ENDC}")
                    exit(-1)
                else:
                    pass
        
        self.action_sock = action_sock
    
    def _connect_to_logger(self):
        logger_port = self.config["network-config"]["logger_port"]
        env_base_port = self.config["network-config"]["envs_logger_base_port"]
        timeout = self.config["network-config"]["envs_connection_timeout"]
        
        logger_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger_sock.bind(("127.0.0.1", env_base_port + self.rank))
        
        now = time.time()
        while True:
            try:
                logger_sock.connect(("127.0.0.1", logger_port))
                break
            except socket.error:
                if time.time() > (now + timeout):
                    print(f"{tcolors.BOLD}{tcolors.FAIL}Unable to connect to logger{tcolors.ENDC}")
                    exit(-1)
                else:
                    pass

        self.logger_sock = logger_sock

    def _extract_projectiles_data(self, observsation):
        base_data = {"n_active_projectiles": observsation[58]}

        for i in range(self.n_projectiles):
            projectiles_data = {f"{i}pos_x": np.array([observsation[59 + 8*i]], dtype=np.float32), 
                                f"{i}pos_y":  np.array([observsation[60 + 8*i]], dtype=np.float32), 
                                f"{i}speed_x":  np.array([observsation[61 + 8*i]], dtype=np.float32), 
                                f"{i}speed_y":  np.array([observsation[62 + 8*i]], dtype=np.float32),
                                f"{i}owner":  np.array([observsation[63 + 8*i]], dtype=np.int8), 
                                f"{i}type":  np.array([observsation[64 + 8*i]], dtype=np.int16), 
                                f"{i}frame":  np.array([observsation[65 + 8*i]], dtype=np.int32), 
                                f"{i}subtype":  np.array([observsation[66 + 8*i]], dtype=np.int32)}
            
            base_data.update(projectiles_data)
        
        return base_data

    def _get_obs(self, infos, game_settings):
        observation_payload_char = "l?ffffiiff??ii??iiii??iiiiiiii??ffffffff??ffffffffffffffffi" + "ffffiiii"*self.n_projectiles
        observation_payload_size = struct.calcsize(observation_payload_char)

        while True:
            observation_payload = self.action_sock.recv(observation_payload_size)
            if observation_payload:
                observation = struct.unpack(observation_payload_char, observation_payload)
                return {
                    "frame": np.array([observation[0]], dtype=np.int32),
                    "total_projectiles": self.n_projectiles,
                    "stage": {
                        "stage_id": game_settings[0],
                        "blastzones": np.array([infos[4], infos[5], infos[6], infos[7]], dtype=np.float32),
                        "edge": np.array([infos[8], infos[9]], dtype=np.float32),
                        "edge_ground": np.array([infos[10], infos[11]], dtype=np.float32),
                        "right_platform": np.array([infos[12], infos[13], infos[14]], dtype=np.float32),
                        "left_platform": np.array([infos[15], infos[16], infos[17]], dtype=np.float32),
                        "top_platform": np.array([infos[18], infos[19], infos[20]], dtype=np.float32)
                    },
                    "agent": {
                        "character": game_settings[1],
                        "position": np.array([observation[2], observation[3]], dtype=np.float32),
                        "percent": np.array([observation[6]], dtype=np.int16),
                        "shield_strenght": np.array([observation[8]], dtype=np.float32),
                        "is_powershield": int(observation[10]),
                        "stock": np.array([observation[12]], dtype=np.int8),
                        "facing": int(observation[14]),
                        "action_frame": np.array([observation[16]], dtype=np.int16),
                        "action_id": np.array([observation[18]], dtype=np.int32),
                        "invulnerable": int(observation[20]),
                        "invulnerability_left": np.array([observation[22]], dtype=np.int16),
                        "hitlag_left": np.array([observation[24]], dtype=np.int16),
                        "hitstun_left": np.array([observation[26]], dtype=np.int16),
                        "jumps_left": np.array([observation[28]], dtype=np.int8),
                        "on_ground": int(observation[30]),
                        "speed_air_x": np.array([observation[32]], dtype=np.float32),
                        "speed_y": np.array([observation[34]], dtype=np.float32),
                        "speed_attack_x": np.array([observation[36]], dtype=np.float32),
                        "speed_attack_y": np.array([observation[38]], dtype=np.float32),
                        "is_moonwalking": int(observation[40]),
                        "ecb_top_x": np.array([observation[42]], dtype=np.float32),
                        "ecb_top_y": np.array([observation[44]], dtype=np.float32),
                        "ecb_bottom_x": np.array([observation[46]], dtype=np.float32),
                        "ecb_bottom_y": np.array([observation[48]], dtype=np.float32),
                        "ecb_left_x": np.array([observation[50]], dtype=np.float32),
                        "ecb_left_y": np.array([observation[52]], dtype=np.float32),
                        "ecb_right_x": np.array([observation[54]], dtype=np.float32),
                        "ecb_right_y": np.array([observation[56]], dtype=np.float32)
                    },
                    "cpu": {
                        "character": game_settings[2],
                        "position": np.array([observation[4], observation[5]], dtype=np.float32),
                        "percent": np.array([observation[7]], dtype=np.int16),
                        "shield_strenght": np.array([observation[9]], dtype=np.float32),
                        "is_powershield": int(observation[11]),
                        "stock": np.array([observation[13]], dtype=np.int8),
                        "facing": int(observation[15]),
                        "action_frame": np.array([observation[17]], dtype=np.int16),
                        "action_id": np.array([observation[19]], dtype=np.int32),
                        "invulnerable": int(observation[21]),
                        "invulnerability_left": np.array([observation[23]], dtype=np.int16),
                        "hitlag_left": np.array([observation[25]], dtype=np.int16),
                        "hitstun_left": np.array([observation[27]], dtype=np.int16),
                        "jumps_left": np.array([observation[29]], dtype=np.int8),
                        "on_ground": int(observation[31]),
                        "speed_air_x": np.array([observation[33]], dtype=np.float32),
                        "speed_y": np.array([observation[35]], dtype=np.float32),
                        "speed_attack_x": np.array([observation[37]], dtype=np.float32),
                        "speed_attack_y": np.array([observation[39]], dtype=np.float32),
                        "is_moonwalking": int(observation[41]),
                        "ecb_top_x": np.array([observation[43]], dtype=np.float32),
                        "ecb_top_y": np.array([observation[45]], dtype=np.float32),
                        "ecb_bottom_x": np.array([observation[47]], dtype=np.float32),
                        "ecb_bottom_y": np.array([observation[49]], dtype=np.float32),
                        "ecb_left_x": np.array([observation[51]], dtype=np.float32),
                        "ecb_left_y": np.array([observation[53]], dtype=np.float32),
                        "ecb_right_x": np.array([observation[55]], dtype=np.float32),
                        "ecb_right_y": np.array([observation[57]], dtype=np.float32)
                    },
                    "projectiles": self._extract_projectiles_data(observation)
                }, bool(observation[1])
    
    def _get_infos(self):
        infos_payload_char = "fffffffffffffffffffff"
        infos_payload_size = struct.calcsize(infos_payload_char)

        while True:
            infos_payload = self.action_sock.recv(infos_payload_size)
            if infos_payload:
                infos = struct.unpack(infos_payload_char, infos_payload)
                
                return infos

    def _wait_for_instance(self):
        if self.debug:
            ready_spinner = Halo(f"Waiting for instance {self.rank}. IP: {self.config["instances"][f"{self.rank}"]["ip"]}", spinner="dots")
            ready_spinner.start()
            
            try:
                while True:
                    if self.action_sock.recv(25):
                        ready_spinner.succeed(f"Instance {self.rank} is READY !!")
                        break
            except KeyboardInterrupt:
                ready_spinner.fail(f"Instance {self.rank} is UNREACHABLE !!")
        else:
            while True:
                if self.action_sock.recv(25):
                    ready_spinner.succeed(f"Instance {self.rank} is READY !!")
                    break
    
    def _send_match_settings(self, settings:tuple):
        settings_payload = struct.pack("hhhhh", *settings, self.n_projectiles)
        self.action_sock.send(settings_payload)

    def _get_game_logs(self, observation):
        game_time = round(observation["frame"]/60, 3)
        stage = observation["stage"]["stage_id"]
        agent_character = observation["agent"]["character"]
        agent_percent = observation["agent"]["percent"].item()
        agent_stock = observation["agent"]["stock"].item()
        cpu_character = observation["cpu"]["character"]
        cpu_percent = observation["cpu"]["percent"].item()
        cpu_stock = observation["cpu"]["stock"].item()

        return (game_time, stage, agent_character, agent_percent, agent_stock, cpu_character, cpu_percent, cpu_stock)

    def _calculate_reward(self, current_percent_agent, current_percent_cpu, current_frame, stock_agent, 
                          stock_cpu, agent_punch_power_modifier, cpu_punch_power_modifier, agent_combo_modifier, 
                          cpu_combo_modifier, sub_frame_damage_modifier, percent_modifier, agent_win_reward, cpu_win_reward):
        percent_agent_change = max(0, current_percent_agent - self.reward_memory["last_percent_agent"])
        percent_cpu_change = max(0, current_percent_cpu - self.reward_memory["last_percent_cpu"])
        percent_cpu_agent_difference = abs(current_percent_cpu - current_percent_agent)

        if percent_agent_change > 0:
            self.reward_memory["last_change_frame_agent"] = current_frame
        if percent_cpu_change > 0:
            self.reward_memory["last_change_frame_cpu"] = current_frame

        delta_frame_agent = (current_frame - self.reward_memory["last_change_frame_agent"]) / cpu_combo_modifier
        delta_frame_cpu = (current_frame - self.reward_memory["last_change_frame_cpu"]) / agent_combo_modifier

        percent_agent_no_0 = current_percent_agent if current_percent_agent != 0 else 1
        percent_cpu_no_0 = current_percent_cpu if current_percent_cpu != 0 else 1

        reward_agent = math.pow(percent_cpu_agent_difference / percent_agent_no_0, percent_cpu_change * agent_punch_power_modifier) / max(sub_frame_damage_modifier, delta_frame_cpu)
        reward_cpu = math.pow(percent_cpu_agent_difference / percent_cpu_no_0, percent_agent_change * cpu_punch_power_modifier) / max(sub_frame_damage_modifier, delta_frame_agent)

        cpu_stock_bonus = 1 + (4 - stock_cpu)
        agent_stock_penalty = 1 / (1 + (4 - stock_agent))
        stock_modifier = cpu_stock_bonus * agent_stock_penalty

        total_reward = (reward_agent - reward_cpu) * stock_modifier

        if stock_agent < self.reward_memory["last_stock_agent"]:
            total_reward = total_reward - cpu_win_reward / (current_percent_agent / percent_modifier)
        
        if stock_cpu < self.reward_memory["last_stock_cpu"]:
            total_reward = total_reward + agent_win_reward / (current_percent_cpu / percent_modifier)
        
        self.reward_memory["last_percent_agent"] = current_percent_agent
        self.reward_memory["last_percent_cpu"] = current_percent_cpu
        self.reward_memory["last_stock_agent"] = stock_agent
        self.reward_memory["last_stock_cpu"] = stock_cpu

        return self._clamp(total_reward, -10000.0, 10000.0)

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        self.reward_memory = {
            "last_percent_agent": 0,
            "last_percent_cpu": 0,
            "last_stock_agent": 4,
            "last_stock_cpu": 4,
            "last_change_frame_agent": 0,
            "last_change_frame_cpu": 0
            }

        self._wait_for_instance()

        self.match_settings = self.match_maker.new_match()
        self._send_match_settings(self.match_settings)
    
        self.match_infos = self._get_infos()

        observation, done = self._get_obs(self.match_infos, self.match_settings)

        self.game_logs = self._get_game_logs(observation)

        return observation, {"match_settings": self.match_settings}
    
    def step(self, action):
        agent_punch_power_modifier = self.config["reward-settings"]["agent_punch_power_modifier"]
        cpu_punch_power_modifier = self.config["reward-settings"]["cpu_punch_power_modifier"]
        agent_combo_modifier = self.config["reward-settings"]["agent_combo_modifier"]
        cpu_combo_modifier = self.config["reward-settings"]["cpu_combo_modifier"]
        sub_frame_damage_modifier = self.config["reward-settings"]["sub_frame_damage_modifier"]
        percent_modifier = self.config["reward-settings"]["percent_modifier"]
        agent_win_reward = self.config["reward-settings"]["agent_win_reward"]
        cpu_win_reward = self.config["reward-settings"]["cpu_win_reward"]

        action_payload_char = "iiiiiiiiiiffffff"
        controller_action = self._action_to_controller(action)
        self.last_action = controller_action

        action_payload = struct.pack(action_payload_char, 0, *controller_action)
        self.action_sock.send(action_payload)

        observation, done = self._get_obs(self.match_infos, self.match_settings)

        self.game_logs = self._get_game_logs(observation)
        
        current_percent_agent = observation["agent"]["percent"]
        current_percent_cpu = observation["cpu"]["percent"]
        current_frame = observation["frame"]
        stock_agent = observation["agent"]["stock"]
        stock_cpu = observation["cpu"]["stock"]

        reward = self._calculate_reward(current_percent_agent, current_percent_cpu, current_frame, stock_agent, 
                               stock_cpu, agent_punch_power_modifier, cpu_punch_power_modifier, agent_combo_modifier, 
                               cpu_combo_modifier, sub_frame_damage_modifier, percent_modifier, agent_win_reward, cpu_win_reward)
        
        return observation, reward, done, False, {}
    
    def pause_game(self):
        action_payload_char = "iiiiiiiiiiffffff"

        action_payload = struct.pack(action_payload_char, 1, *self.last_action)
        self.action_sock.send(action_payload)

    def resume_game(self):
        action_payload_char = "iiiiiiiiiiffffff"

        action_payload = struct.pack(action_payload_char, 2, *self.last_action)
        self.action_sock.send(action_payload)

    def send_logs(self, game, score, avg_score, learn_iters):
        payload_char = "fiiiiiiiiffi"

        logs_payload = struct.pack(payload_char, *self.game_logs, game, score, avg_score, learn_iters)
        self.logger_sock.send(logs_payload)

    def close(self):
        self.action_sock.close()
        self.logger_sock.close()