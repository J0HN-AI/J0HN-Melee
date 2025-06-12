import melee
import time
import configparser
import os

def platform_None_fix(platform):
    if platform == (None, None, None):
        return (-1, -1, -1)
    else:
        return platform

class Melee:
    def __init__(self, ppoPort, cpuPort, isoPath,
                  dolphinPath="/home/jul/.config/Slippi Launcher/netplay/squashfs-root/usr/bin", fullscreen=False):
        self._isoPath = isoPath
        self._dolphinPath = dolphinPath
        self._fullscreen = fullscreen
        self.ppoPort = ppoPort
        self.cpuPort = cpuPort
        self.paused = False

        self.console = melee.Console(path=self._dolphinPath, fullscreen=self._fullscreen, gfx_backend="Vulkan")

        self.ppoController = melee.Controller(console=self.console, port=self.ppoPort)
        #self.other_dolphin_controller(self.ppoPort, "Xbox")
        self.cpuController = melee.Controller(console=self.console, port=self.cpuPort)

        self.console.connect()
        print("Console connected")

        self.console.run(iso_path=self._isoPath)

        self.ppoController.connect()
        self.cpuController.connect()

        while True:
            gamestate = self.console.step()
            if gamestate.menu_state != melee.Menu.CHARACTER_SELECT:
                 melee.MenuHelper.choose_versus_mode(gamestate, self.cpuController)
            else:
                break
    
    def other_dolphin_controller(self, port, config_name):
        pipes_path = self.console.get_dolphin_pipes_path(port)
        if not os.path.exists(pipes_path):
                os.mkfifo(pipes_path)

        controller_config_dolphin_path = self.console._get_dolphin_config_path() + "GCPadNew.ini"
        controller_config_slippi_path = self._dolphinPath + f"/Sys/Config/Profiles/GCPad/{config_name}.ini" 
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

    def game_init(self, stage, ppo, cpu, cpuLevel):
        self.paused = False
        self.stage = self.get_stage(stage)
        self.ppoPlayer = self.get_player(ppo)
        self.cpuPlayer = self.get_player(cpu)
        self.cpuLevel = cpuLevel
        ControllerJitter = ""
        isChosen = False
        while True:
            gamestate = self.console.step()
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self.get_stage_data(gamestate)
            
            elif gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
                if isChosen == False:
                    melee.MenuHelper.choose_character(self.cpuPlayer,
                                              gamestate,
                                              self.cpuController,
                                              cpu_level=self.cpuLevel,
                                              costume=0,
                                              start=False)
                       
                    melee.MenuHelper.choose_character(self.ppoPlayer,
                                              gamestate,
                                              self.ppoController,
                                              cpu_level=0,
                                              costume=0,
                                              start=False)

                ControllerJitter += str(gamestate.players[self.cpuPort].controller_status.value)

                if ControllerJitter.find("330011") != -1:
                    isChosen = True
                    time.sleep(6/60)
                    self.cpuController.simple_press(button=melee.Button.BUTTON_START, x=0.5, y=0.5)
                    ControllerJitter = ""
                        
            else:                                  
                if (gamestate.menu_state == melee.Menu.STAGE_SELECT):
                    melee.MenuHelper.choose_stage(self.stage, gamestate, self.cpuController)

    def reset(self, stage, ppo, cpu,  cpuLevel):
        ppoPort = self.ppoPort
        cpuPort = self.cpuPort

        self.ppoController.disconnect()
        self.cpuController.disconnect()
        self.console.stop()
        time.sleep(1.5)

        self.__init__(ppoPort, cpuPort, self._isoPath, self._dolphinPath, self._fullscreen)
        return self.game_init(stage, ppo, cpu, cpuLevel)

    def pause(self):
        if not self.paused:
            self.ppoController.simple_press(0.5, 0.5, melee.Button.BUTTON_START)
            self.paused = True

    def resume(self):
        if self.paused:
            self.ppoController.simple_press(0.5, 0.5, melee.Button.BUTTON_START)
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
        return ([gamestate.players[self.ppoPort].position.x, gamestate.players[self.ppoPort].position.y],
                [gamestate.players[self.cpuPort].position.x, gamestate.players[self.cpuPort].position.y],
                melee.BLASTZONES[self.stage],  
                (melee.EDGE_POSITION[self.stage], -melee.EDGE_POSITION[self.stage]),
                (melee.EDGE_GROUND_POSITION[self.stage], -melee.EDGE_GROUND_POSITION[self.stage]),
                platform_None_fix(melee.right_platform_position(gamestate)),
                platform_None_fix(melee.left_platform_position(gamestate)),
                platform_None_fix(melee.top_platform_position(gamestate)))