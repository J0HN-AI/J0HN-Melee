import melee
import time
import configparser
import os

def platform_None_fix(platform):
    if platform == (None, None, None):
        return (-255, -255, -255)
    else:
        return platform

class Melee:
    def __init__(self, agent_port, cpu_port, iso_path,
                  dolphin_path="/home/jul/.config/Slippi Launcher/netplay/squashfs-root/usr/bin", fullscreen=False, backend="Vulkan"):
        self._iso_path = iso_path
        self._dolphin_path = dolphin_path
        self._fullscreen = fullscreen
        self.agent_port = agent_port
        self.cpu_port = cpu_port
        self.paused = False

        self.console = melee.Console(path=self._dolphin_path, fullscreen=self._fullscreen, gfx_backend=backend)

        self.agent_controller = melee.Controller(console=self.console, port=self.agent_port)
        #self.other_dolphin_controller(self.agent_port, "Xbox")
        self.cpu_controller = melee.Controller(console=self.console, port=self.cpu_port)

        self.console.connect()
        print("Console connected")

        self.console.run(iso_path=self._iso_path)

        self.agent_controller.connect()
        self.cpu_controller.connect()

        while True:
            gamestate = self.console.step()
            if gamestate.menu_state != melee.Menu.CHARACTER_SELECT:
                 melee.MenuHelper.choose_versus_mode(gamestate, self.cpu_controller)
            else:
                break
    
    def other_dolphin_controller(self, port, config_name):
        pipes_path = self.console.get_dolphin_pipes_path(port)
        if not os.path.exists(pipes_path):
                os.mkfifo(pipes_path)

        controller_config_dolphin_path = self.console._get_dolphin_config_path() + "GCPadNew.ini"
        controller_config_slippi_path = self._dolphin_path + f"/Sys/Config/Profiles/GCPad/{config_name}.ini" 
        config_dolphin = configparser.ConfigParser()
        config_slippi = configparser.ConfigParser()
        config_dolphin.read(controller_config_dolphin_path)
        config_slippi.read(controller_config_slippi_path)

        section = "GCPad" + str(port)
        if not config_dolphin.has_section(section):
            config_dolphin.add_section(section)
        
        for key, val in config_slippi.items("Profile"):
            config_dolphin.set(section, key, val)
        
        with open(controller_config_dolphin_path, "w") as configfile:
            config_dolphin.write(configfile)
        
        dolphin_config_path = self.console._get_dolphin_config_path() + "Dolphin.ini"
        config = configparser.ConfigParser()
        config.read(dolphin_config_path)
        # Indexed at 0. "6" means standard controller, "12" means GCN Adapter
        #  The enum is scoped to the proper value, here
        config.set("Core", "SIDevice" + str(port - 1), "6")
        with open(dolphin_config_path, "w") as dolphinfile:
            config.write(dolphinfile)

    def game_init(self, stage, ppo, cpu, cpu_level):
        self.paused = False
        self.stage = self.get_stage(stage)
        self.agent_player = self.get_player(ppo)
        self.cpu_player = self.get_player(cpu)
        self.cpu_level = cpu_level
        controller_jitter = ""
        isChosen = False
        while True:
            gamestate = self.console.step()
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self.get_stage_data(gamestate)
            
            elif gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
                if isChosen == False:
                    melee.MenuHelper.choose_character(self.cpu_player,
                                              gamestate,
                                              self.cpu_controller,
                                              cpu_level=self.cpu_level,
                                              costume=0,
                                              start=False)
                       
                    melee.MenuHelper.choose_character(self.agent_player,
                                              gamestate,
                                              self.agent_controller,
                                              cpu_level=0,
                                              costume=0,
                                              start=False)

                controller_jitter += str(gamestate.players[self.cpu_port].controller_status.value)

                if controller_jitter.find("330011") != -1:
                    isChosen = True
                    time.sleep(6/60)
                    self.cpu_controller.simple_press(button=melee.Button.BUTTON_START, x=0.5, y=0.5)
                    controller_jitter = ""
                        
            else:                                  
                if (gamestate.menu_state == melee.Menu.STAGE_SELECT):
                    melee.MenuHelper.choose_stage(self.stage, gamestate, self.cpu_controller)

    def reset(self, stage, ppo, cpu,  cpu_level):
        agent_port = self.agent_port
        cpu_port = self.cpu_port

        self.agent_controller.disconnect()
        self.cpu_controller.disconnect()
        self.console.stop()
        time.sleep(1.5)

        self.__init__(agent_port, cpu_port, self._iso_path, self._dolphin_path, self._fullscreen)
        return self.game_init(stage, ppo, cpu, cpu_level)

    def pause(self):
        if not self.paused:
            self.agent_controller.simple_press(0.5, 0.5, melee.Button.BUTTON_START)
            self.paused = True

    def resume(self):
        if self.paused:
            self.agent_controller.simple_press(0.5, 0.5, melee.Button.BUTTON_START)
            self.paused = False
    
    def get_stage(self, hex_stage):
        #Convert hex bytes to comprehensible things for melee api, hex bytes bcs it is simple so send over ETH protocol
        match hex_stage:
            case 0x0:
                return melee.Stage.BATTLEFIELD
            case 0x1:
                return melee.Stage.FINAL_DESTINATION
            case 0x2:
                return melee.Stage.DREAMLAND
            case 0x3:
                return melee.Stage.FOUNTAIN_OF_DREAMS
            case 0x4:
                return melee.Stage.POKEMON_STADIUM
            case 0x5:
                return melee.Stage.YOSHIS_STORY
    
    def get_player(self, hex_player):
        #Same as get_stage
        match hex_player:
            case 0x00:
                return melee.Character.DOC
            case 0x01:
                return melee.Character.MARIO
            case 0x02:
                return melee.Character.LUIGI
            case 0x03:
                return melee.Character.BOWSER
            case 0x04:
                return melee.Character.PEACH
            case 0x05:
                return melee.Character.YOSHI
            case 0x06:
                return melee.Character.DK
            case 0x07:
                return melee.Character.CPTFALCON
            case 0x08:
                return melee.Character.GANONDORF
            case 0x09:
                return melee.Character.FALCO
            case 0x0a:
                return melee.Character.FOX
            case 0x0b:
                return melee.Character.NESS
            case 0x0c:
                return melee.Character.POPO
            case 0x0d:
                return melee.Character.KIRBY
            case 0x0e:
                return melee.Character.SAMUS
            case 0x0f:
                return melee.Character.ZELDA
            case 0x10:
                return melee.Character.LINK
            case 0x11:
                return melee.Character.YLINK
            case 0x12:
                return melee.Character.PICHU
            case 0x13:
                return melee.Character.PIKACHU
            case 0x14:
                return melee.Character.JIGGLYPUFF
            case 0x15:
                return melee.Character.MEWTWO
            case 0x16:
                return melee.Character.GAMEANDWATCH
            case 0x17:
                return melee.Character.MARTH
            case 0x18:
                return melee.Character.ROY

    def get_stage_data(self, gamestate: melee.GameState):
        return ([gamestate.players[self.agent_port].position.x, gamestate.players[self.agent_port].position.y],
                [gamestate.players[self.cpu_port].position.x, gamestate.players[self.cpu_port].position.y],
                melee.BLASTZONES[self.stage],  
                (melee.EDGE_POSITION[self.stage], -melee.EDGE_POSITION[self.stage]),
                (melee.EDGE_GROUND_POSITION[self.stage], -melee.EDGE_GROUND_POSITION[self.stage]),
                platform_None_fix(melee.right_platform_position(gamestate)),
                platform_None_fix(melee.left_platform_position(gamestate)),
                platform_None_fix(melee.top_platform_position(gamestate)))