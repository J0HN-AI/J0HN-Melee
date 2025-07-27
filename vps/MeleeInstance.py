import melee
import time
import configparser
import math
import os
import subprocess

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

        self.console = melee.Console(path=self._dolphin_path, fullscreen=self._fullscreen, gfx_backend=backend, disable_audio=True)

        self.agent_controller = melee.Controller(console=self.console, port=self.agent_port)
        #self._other_dolphin_controller(self.agent_port, "Xbox")
        self.cpu_controller = melee.Controller(console=self.console, port=self.cpu_port)

        self.console.connect()
        print("Console connected")

        self.console.run(iso_path=self._iso_path)

        self.agent_controller.connect()
        self.cpu_controller.connect()
    
    def _other_dolphin_controller(self, port, config_name):
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

    def _release_all(self, controller:melee.Controller):
        command = f"RELEASE A" + "\n"
        command += f"RELEASE B" + "\n"
        command += f"RELEASE X" + "\n"
        command += f"RELEASE Y" + "\n"
        command += f"RELEASE Z" + "\n"
        command += f"RELEASE L" + "\n"
        command += f"RELEASE R" + "\n"
        command += f"RELEASE D_UP" + "\n"
        command += f"RELEASE D_DOWN" + "\n"
        command += f"RELEASE D_LEFT" + "\n"
        command += f"RELEASE D_RIGHT" + "\n"
        command += f"SET MAIN 0.5 0.5" + "\n"
        command += f"SET C 0.5 0.5" + "\n"
        command += f"SET L 0" + "\n"
        command += f"SET R 0" + "\n"

        controller._write(command)
        controller.flush()

    def _press_once(self, controller:melee.Controller, button:str, delay=0):
        self.console.step()
        command = f"PRESS {button}" + "\n"

        controller._write(command)
        controller.flush()
        time.sleep(delay)

        self.console.step()
        command = f"RELEASE {button}" + "\n"

        controller._write(command)
        controller.flush()

    def _analog_tilt(self, tilt_x, tilt_y, stick_name:str, controller:melee.Controller):
        command = f"SET {stick_name} {tilt_x} {tilt_y}" + "\n"

        controller._write(command)
        controller.flush()

    def _caluclate_joystick_tilt(self, cursor_pos:tuple, target_x:float, target_y:float):
        cursor_x, cursor_y = cursor_pos

        distance_x = target_x - cursor_x
        distance_y = target_y - cursor_y

        distance = math.hypot(distance_x, distance_y)

        if distance == 0:
            return (0.5, 0.5)
        
        dir_vector_x = distance_x / distance
        dir_vector_y = distance_y / distance

        tilt_x = 0.5 + (dir_vector_x * 0.5)
        tilt_y = 0.5 + (dir_vector_y * 0.5) 

        return (tilt_x, tilt_y)

    def _choose_character(self, character, controller:melee.Controller, cpu_level:int=0, autostart:bool = False):
        self._release_all(controller)
        controlling_port = controller.port
        cpu_level_postition = {
            1: {
                1: (-30.8, -15.12),
                2: (-29.65, -15.12),
                3: (-28.45, -15.12),
                4: (-27.25, -15.12),
                5: (-26.06, -15.12),
                6: (-24.88, -15.12),
                7: (-23.7, -15.12),
                8: (-22.51, -15.12),
                9: (-21.17, -15.12)
            },
            2: {
                1: (-15.43, -15.12),
                2: (-14.1, -15.12),
                3: (-12.9, -15.12),
                4: (-11.7, -15.12),
                5: (-10.48, -15.12),
                6: (-9.28, -15.12),
                7: (-8.1, -15.12),
                8: (-6.89, -15.12),
                9: (-5.57, -15.12)
            },
            3: {
                1: (0.17, -15.12),
                2: (1.24, -15.12),
                3: (2.46, -15.12),
                4: (3.67, -15.12),
                5: (4.89, -15.12),
                6: (6.1, -15.12),
                7: (7.32, -15.12),
                8: (8.53, -15.12),
                9: (9.67, -15.12)
            },
            4: {
                1: (15.5, -15.12),
                2: (16.52, -15.12),
                3: (17.72, -15.12),
                4: (18.93, -15.12),
                5: (20.15, -15.12),
                6: (21.36, -15.12),
                7: (22.57, -15.12),
                8: (23.78, -15.12),
                9: (25.0, -15.12)
            }
        }

        while True:
            gamestate = self.console.step()
            controller_state = gamestate.players[controlling_port]

            cursor_pos = (controller_state.cursor_x, controller_state.cursor_y)

            row = melee.from_internal(character) // 9
            column = melee.from_internal(character) % 9

            row = melee.from_internal(character) // 9
            column = melee.from_internal(character) % 9
            #The random slot pushes the bottom row over a slot, so compensate for that
            if row == 2:
                column = column+1
            #re-order rows so the math is simpler
            row = 2-row

            #Height starts at 1, plus half a box height, plus the number of rows
            target_y = 1 + 3.5 + (row * 7.0)
            #Starts at -32.5, plus half a box width, plus the number of columns
            #NOTE: Technically, each column isn't exactly the same width, but it's close enough
            target_x = -32.5 + 3.5 + (column * 7.0)
            #Wiggle room in positioning character
            wiggleroom = 1

            tilt_x, tilt_y = self._caluclate_joystick_tilt(cursor_pos, target_x, target_y)
            self._analog_tilt(tilt_x, tilt_y, "MAIN", controller)

            if math.isclose(cursor_pos[0], target_x, abs_tol=wiggleroom) and math.isclose(cursor_pos[1], target_y, abs_tol=wiggleroom):
                self._release_all(controller)
                self._press_once(controller, "B")
                self._press_once(controller, "A")
                
                if cpu_level == 0:
                    return True

                if autostart and cpu_level == 0:
                    time.sleep(0.5)
                    self._press_once(controller, "START", delay=0.5)
                    self._release_all(controller)
                break
        
        global current_cpu_level
        current_cpu_level = 1
        if cpu_level != 0:
            while True:
                gamestate = self.console.step()
                controller_state = gamestate.players[controlling_port]

                if gamestate.players[controlling_port].controller_status == melee.ControllerStatus.CONTROLLER_CPU:
                    self._release_all(controller)
                    current_cpu_level = controller_state.cpu_level
                    break
                else:
                    melee.MenuHelper.change_controller_status(controller, gamestate, controlling_port, melee.ControllerStatus.CONTROLLER_CPU)

            while True:
                gamestate = self.console.step()
                controller_state = gamestate.players[controlling_port]
                wiggleroom = 1

                cursor_pos = (controller_state.cursor_x, controller_state.cursor_y)
                target_x, target_y = cpu_level_postition[controlling_port][current_cpu_level]
                
                tilt_x, tilt_y = self._caluclate_joystick_tilt(cursor_pos, target_x, target_y)
                self._analog_tilt(tilt_x, tilt_y, "MAIN", controller)
                
                if math.isclose(cursor_pos[0], target_x, abs_tol=wiggleroom) and math.isclose(cursor_pos[1], target_y, abs_tol=wiggleroom):
                    self._release_all(controller)
                    self._press_once(controller, "A")
                    break
            
            while True:
                gamestate = self.console.step()
                controller_state = gamestate.players[controlling_port]
                wiggleroom = 0.5

                cursor_pos = (controller_state.cursor_x, controller_state.cursor_y)
                target_x, target_y = cpu_level_postition[controlling_port][cpu_level]
                
                tilt_x, tilt_y = self._caluclate_joystick_tilt(cursor_pos, target_x, target_y)
                self._analog_tilt(tilt_x, tilt_y, "MAIN", controller)
                
                if math.isclose(cursor_pos[0], target_x, abs_tol=wiggleroom) and math.isclose(cursor_pos[1], target_y, abs_tol=wiggleroom):
                    self._release_all(controller)
                    self._press_once(controller, "A")

                    if autostart and cpu_level != 0:
                        time.sleep(0.5)
                        self._press_once(controller, "START", delay=0.5)
                        self._release_all(controller)
                    return True

    def _choose_stage(self, stage, controller):
        global stage_target_x, stage_target_y
        stage_target_x, stage_target_y = 0, 0
        if stage == melee.Stage.BATTLEFIELD:
            stage_target_x, stage_target_y = 1.25, -9.1
        if stage == melee.Stage.FINAL_DESTINATION:
            stage_target_x, stage_target_y = 6.48, -9.1
        if stage == melee.Stage.DREAMLAND:
            stage_target_x, stage_target_y = 12.16, -9.1
        if stage == melee.Stage.POKEMON_STADIUM:
            stage_target_x, stage_target_y = 15.06, 3.6
        if stage == melee.Stage.YOSHIS_STORY:
            stage_target_x, stage_target_y = 3.27, 15.62
        if stage == melee.Stage.FOUNTAIN_OF_DREAMS:
            stage_target_x, stage_target_y = 9.75, 15.62
        if stage == melee.Stage.RANDOM_STAGE:
            stage_target_x, stage_target_y = -14.3, 3.6

        self._release_all(controller)
        controlling_port = controller.port
        
        while True:
            gamestate = self.console.step()
            controller_state = gamestate.players[controlling_port]

            cursor_pos = (controller_state.cursor.x, controller_state.cursor.y)
            wiggleroom = 1

            tilt_x, tilt_y = self._caluclate_joystick_tilt(cursor_pos, stage_target_x, stage_target_y)
            self._analog_tilt(tilt_x, tilt_y, "MAIN", controller)

            if math.isclose(cursor_pos[0], stage_target_x, abs_tol=wiggleroom) and math.isclose(cursor_pos[1], stage_target_y, abs_tol=wiggleroom):
                self._release_all(controller)
                self._press_once(controller, "A")
                self._release_all(controller)
                return True

    def game_init(self, stage, agent, cpu, cpu_level):
        self.paused = False
        self.stage = self._get_stage(stage)
        self.agent_player = self._get_player(agent)
        self.cpu_player = self._get_player(cpu)
        self.cpu_level = cpu_level
        agent_character_chosen = False
        cpu_character_chosen = False
        stage_selected = False

        while True:
            gamestate = self.console.step()
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self._get_stage_data(gamestate)
            if gamestate.menu_state in [melee.Menu.PRESS_START, melee.Menu.MAIN_MENU]:   
                melee.MenuHelper.choose_versus_mode(gamestate, self.agent_controller)
            elif gamestate.menu_state == melee.Menu.POSTGAME_SCORES:
                melee.MenuHelper.skip_postgame(self.agent_controller, gamestate)
            elif gamestate.menu_state == melee.Menu.CHARACTER_SELECT and not agent_character_chosen and not cpu_character_chosen:
                agent_character_chosen = self._choose_character(self.agent_player, self.agent_controller, cpu_level=0, autostart=False)
                cpu_character_chosen = self._choose_character(self.cpu_player, self.cpu_controller, cpu_level=self.cpu_level, autostart=True)
            elif gamestate.menu_state == melee.Menu.STAGE_SELECT and not stage_selected:
                stage_selected = self._choose_stage(self.stage, self.agent_controller)

    def reset(self, stage, agent, cpu,  cpu_level):
        agent_port = self.agent_port
        cpu_port = self.cpu_port

        self.agent_controller.disconnect()
        self.cpu_controller.disconnect()
        self.console.stop()
        time.sleep(5)

        self.__init__(agent_port, cpu_port, self._iso_path, self._dolphin_path, self._fullscreen)
        return self.game_init(stage, agent, cpu, cpu_level)

    def stop(self):
        self.agent_controller.disconnect()
        self.cpu_controller.disconnect()
        self.console.stop()

    def pause(self):
        if not self.paused:
            subprocess.run(["xdotool", "keydown", "F10"])
            time.sleep(0.1)
            subprocess.run(["xdotool", "keyup", "F10"])
            self.paused = True

    def resume(self):
        if self.paused:
            subprocess.run(["xdotool", "keydown", "F10"])
            time.sleep(0.1)
            subprocess.run(["xdotool", "keyup", "F10"])
            self.paused = False
    
    def _get_stage(self, hex_stage):
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
    
    def _get_player(self, hex_player):
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

    def _get_stage_data(self, gamestate: melee.GameState):
        return ([gamestate.players[self.agent_port].position.x, gamestate.players[self.agent_port].position.y],
                [gamestate.players[self.cpu_port].position.x, gamestate.players[self.cpu_port].position.y],
                melee.BLASTZONES[self.stage],  
                (melee.EDGE_POSITION[self.stage], -melee.EDGE_POSITION[self.stage]),
                (melee.EDGE_GROUND_POSITION[self.stage], -melee.EDGE_GROUND_POSITION[self.stage]),
                platform_None_fix(melee.right_platform_position(gamestate)),
                platform_None_fix(melee.left_platform_position(gamestate)),
                platform_None_fix(melee.top_platform_position(gamestate)))