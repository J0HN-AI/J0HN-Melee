import numpy as np
import gymnasium as gym
from gymnasium import spaces as sp
import socket
import time
import struct
from halo import Halo
from random import randint

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
            "stage": self._get_stage(config["instances"][self.rank]["stage"]),
            "agent_character": self._get_player(config["instances"][self.rank]["agent_character"]),
            "cpu_character": self._get_player(config["instances"][self.rank]["cpu_character"]),
            "cpu_level": config["instances"][self.rank]["cpu_level"],
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
        stage_change_rate = self.config["instances"][self.rank]["stage_change_rate"]
        agent_character_change_rate = self.config["instances"][self.rank]["agent_character_change_rate"]
        cpu_character_change_rate = self.config["instances"][self.rank]["cpu_character_change_rate"]
        minimum_games_berfore_changing_cpu_level = self.config["instances"][self.rank]["minimum_games_berfore_changing_cpu_level"]
        cpu_level_progression_rate = self.config["instances"][self.rank]["cpu_level_progression_rate"]
        max_cpu_level =  self.config["instances"][self.rank]["max_cpu_level"]
        
        if stage_change_rate != 0 and stage_change_rate <= self.match_history["n_games"]:
            random_stage = randint(0, 5)
            if random_stage == self.match_history["stage"]:
                self.match_history.update({ "stage": (random_stage + 1) % 6 })
            else:
                self.match_history.update({ "stage": random_stage })

        if agent_character_change_rate != 0 and agent_character_change_rate <= self.match_history["n_games"]:
            random_character = randint(0, 24)
            if random_character == self.match_history["agent_character"]:
                self.match_history.update({ "agent_character": (random_character + 1) % 25 })
            else:
                self.match_history.update({ "agent_character": random_character })
        
        if cpu_character_change_rate != 0 and cpu_character_change_rate <= self.match_history["n_games"]:
            random_character = randint(0, 24)
            if random_character == self.match_history["cpu_character"]:
                self.match_history.update({ "cpu_character": (random_character + 1) % 25 })
            else:
                self.match_history.update({ "cpu_character": random_character })

        if minimum_games_berfore_changing_cpu_level <= self.match_history["n_games"] % (minimum_games_berfore_changing_cpu_level + 1) and self.match_history["cpu_level"] < max_cpu_level:
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
    def __init__(self, config:dict, rank:int, debug:float = False):
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
                "edge": sp.Box(low=np.array([-100.0, -100.0]), high=np.array([100.0, 100.0]), dtype=np.float64),
                "edge_ground": sp.Box(low=np.array([-100.0, -100.0]), high=np.array([100.0, 100.0]), dtype=np.float64),
                "right_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float64),
                "left_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float64),
                "top_platform": sp.Box(low=np.array([-255.0, -255.0, -255.0]), high=np.array([255.0, 255.0, 255.0]), dtype=np.float64)
            }),
            "agent": sp.Dict({
                "character": sp.Discrete(25),
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
                "ecb_bottom_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64)
            }),
            "cpu": sp.Dict({
                "character": sp.Discrete(25),
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
                "ecb_bottom_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_left_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
                "ecb_right_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64)
            }),
            "projectiles": sp.Dict(self._make_projectiles_dict(self.n_projectiles))
        })

        #Buttons: A B X Y Z DPAD_UP DPAD_DOWN DPAD_LEFT DPAD_RIGHT MAIN_X MAIN_Y C_X C_Y LEFT_TRIGGER RIGHT_TRIGGER
        #Indexes: 0 1 2 3 4    5       6         7         8          9     10   11  12      13            14
        self.action_space = sp.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype=np.float16)
        
        self._connect_to_router()

    def _action_to_controller(self, actions):
        digital_buttons = np.array(np.round(actions[:5], decimals=0), dtype=np.int8).tolist()
        analog_buttons = actions[5:].tolist()

        return digital_buttons + analog_buttons
    
    def _make_projectiles_dict(self, n_projectiles:int):
        projectiles = {"n_active_projectiles": sp.Box(low=0, high=n_projectiles, shape=(1,), dtype=np.int8)}

        for i in range(n_projectiles):
            projectiles_template = {f"{i}pos_x": sp.Box(low=-255.0, high=255, shape=(1,), dtype=np.float64), 
                                    f"{i}pos_y": sp.Box(low=-255.0, high=255, shape=(1,), dtype=np.float64), 
                                    f"{i}speed_x": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64), 
                                    f"{i}speed_y": sp.Box(low=-32768, high=32767, shape=(1,), dtype=np.float64),
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
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", env_base_port + self.rank))
        
        now = time.time()
        while True:
            try:
                sock.connect(("127.0.0.1", router_port))
                break
            except socket.error:
                if time.time() > (now + timeout):
                    print(f"{tcolors.BOLD}{tcolors.FAIL}Unable to connect to router{tcolors.ENDC}")
                    exit(-1)
                else:
                    pass

        self.sock = sock

    def _extract_projectiles_data(self, observsation):
        base_data = {}

        for i in range(self.n_projectiles):
            projectiles_data = {f"{i}pos_x": np.array([observsation[59 + 8*i]], dtype=np.float64), 
                                f"{i}pos_y":  np.array([observsation[60 + 8*i]], dtype=np.float64), 
                                f"{i}speed_x":  np.array([observsation[61 + 8*i]], dtype=np.float64), 
                                f"{i}speed_y":  np.array([observsation[62 + 8*i]], dtype=np.float64),
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
            observation_payload = self.sock.recv(observation_payload_size)
            if observation_payload:
                observation = struct.unpack(observation_payload_char, observation_payload)
                return {
                    "frame": np.array([observation[0]], dtype=np.int32),
                    "total_projectiles": sp.Discrete(self.n_projectiles+1),
                    "stage": {
                        "stage_id": game_settings[0],
                        "blastzones": np.array([infos[4], infos[5], infos[6], infos[7]], dtype=np.float32),
                        "edge": np.array([infos[8], infos[9]], dtype=np.float64),
                        "edge_ground": np.array([infos[10], infos[11]], dtype=np.float64),
                        "right_platform": np.array([infos[12], infos[13], infos[14]], dtype=np.float64),
                        "left_platform": np.array([infos[15], infos[16], infos[17]], dtype=np.float64),
                        "top_platform": np.array([infos[18], infos[19], infos[20]], dtype=np.float64)
                    },
                    "agent": {
                        "character": game_settings[1],
                        "position": np.array([observation[2], observation[3]], dtype=np.float64),
                        "percent": np.array([observation[6]], dtype=np.int16),
                        "shield_strenght": np.array([observation[8]], dtype=np.float64),
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
                        "speed_air_x": np.array([observation[32]], dtype=np.float64),
                        "speed_y": np.array([observation[34]], dtype=np.float64),
                        "speed_attack_x": np.array([observation[36]], dtype=np.float64),
                        "speed_attack_y": np.array([observation[38]], dtype=np.float64),
                        "is_moonwalking": int(observation[40]),
                        "ecb_top_x": np.array([observation[42]], dtype=np.float64),
                        "ecb_top_y": np.array([observation[44]], dtype=np.float64),
                        "ecb_bottom_x": np.array([observation[46]], dtype=np.float64),
                        "ecb_bottom_y": np.array([observation[48]], dtype=np.float64),
                        "ecb_left_x": np.array([observation[50]], dtype=np.float64),
                        "ecb_left_y": np.array([observation[52]], dtype=np.float64),
                        "ecb_right_x": np.array([observation[54]], dtype=np.float64),
                        "ecb_right_y": np.array([observation[56]], dtype=np.float64)
                    },
                    "cpu": {
                        "character": game_settings[2],
                        "position": np.array([observation[4], observation[5]], dtype=np.float64),
                        "percent": np.array([observation[7]], dtype=np.int16),
                        "shield_strenght": np.array([observation[9]], dtype=np.float64),
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
                        "speed_air_x": np.array([observation[33]], dtype=np.float64),
                        "speed_y": np.array([observation[35]], dtype=np.float64),
                        "speed_attack_x": np.array([observation[37]], dtype=np.float64),
                        "speed_attack_y": np.array([observation[39]], dtype=np.float64),
                        "is_moonwalking": int(observation[41]),
                        "ecb_top_x": np.array([observation[43]], dtype=np.float64),
                        "ecb_top_y": np.array([observation[45]], dtype=np.float64),
                        "ecb_bottom_x": np.array([observation[47]], dtype=np.float64),
                        "ecb_bottom_y": np.array([observation[49]], dtype=np.float64),
                        "ecb_left_x": np.array([observation[51]], dtype=np.float64),
                        "ecb_left_y": np.array([observation[53]], dtype=np.float64),
                        "ecb_right_x": np.array([observation[55]], dtype=np.float64),
                        "ecb_right_y": np.array([observation[57]], dtype=np.float64)
                    },
                    "projectiles": self._extract_projectiles_data(observation)
                }
    
    def _get_infos(self):
        infos_payload_char = "fffffffffffffffffffff"
        infos_payload_size = struct.calcsize(infos_payload_char)

        while True:
            infos_payload = self.sock.recv(infos_payload_size)
            if infos_payload:
                infos = struct.unpack(infos_payload_char, infos_payload)
                
                return infos

    def _wait_for_instance(self):
        if self.debug:
            ready_spinner = Halo(f"Waiting for instance {self.rank}. IP: {self.config["instances"][0]["ip"]}", spinner="dots")
            ready_spinner.start()
            
            try:
                while True:
                    if self.sock.recv(25):
                        ready_spinner.succeed(f"Instance {self.rank} is READY !!")
                        break
            except KeyboardInterrupt:
                ready_spinner.fail(f"Instance {self.rank} is UNREACHABLE !!")
        else:
            while True:
                if self.sock.recv(25):
                    ready_spinner.succeed(f"Instance {self.rank} is READY !!")
                    break
    
    def _send_match_settings(self, settings:tuple):
        settings_payload = struct.pack("hhhhh", *settings, self.n_projectiles)
        self.sock.send(settings_payload)

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        self._wait_for_instance()

        match_settings = self.match_maker.new_match()
        self._send_match_settings(match_settings)

        match_infos = self._get_infos()

        observation = self._get_obs(match_infos, match_settings)

        return observation, {"match_settings": match_settings}