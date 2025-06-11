import MeleeInstance
import socket
import struct
import melee
import math
from halo import Halo

melee.ProjectileType.ARROW.value

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

def clamp(nb, nb_min, nb_max):
    return max(min(nb_max, nb), nb_min)

def connect(sock: socket.socket, host:str, port:int):
    try:
        sock.connect((host, port))
    except socket.error:
        print(f"{tcolors.BOLD}{tcolors.FAIL}Brain not found{tcolors.ENDC}")
        exit(-1)

def get_projectiles(gamestate: melee.GameState, n_projectiles: int, blastzones):
    i = 0
    reach_projectiles = []
    distance_projectiles = []
    for projectile in gamestate.projectiles:
        xdist = gamestate.players[1].position.x - projectile.position.x
        ydist = gamestate.players[1].position.y - projectile.position.y
        distance = math.sqrt((xdist**2) + (ydist**2))
        if i < n_projectiles:
            distance_projectiles.append(distance)
            reach_projectiles += [float(clamp(projectile.position.x, blastzones[0], blastzones[1])),
                                  float(clamp(projectile.position.y, blastzones[3], blastzones[2])),
                                  projectile.speed.x,
                                  projectile.speed.y,
                                  projectile.owner,
                                  projectile.type.value,
                                  projectile.frame,
                                  projectile.subtype]
            
        elif (max(distance_projectiles) > distance and min(distance_projectiles) < distance) or min(distance_projectiles) > distance:
            i_max = distance_projectiles.index(max(distance_projectiles))
            distance_projectiles[i_max] = distance
            reach_projectiles[8*i_max:8*(i_max+1)] = [float(clamp(projectile.position.x, blastzones[0], blastzones[1])),
                                                      float(clamp(projectile.position.y, blastzones[3], blastzones[2])),
                                                      projectile.speed.x,
                                                      projectile.speed.y,
                                                      projectile.owner,
                                                      projectile.type.value,
                                                      projectile.frame,
                                                      projectile.subtype]
        i += 1
    
    if len(gamestate.projectiles) < n_projectiles:
        for j in range(n_projectiles - len(gamestate.projectiles)):
            reach_projectiles += [0.0, 0.0, 0.0, 0.0, 0, -1, 0, 0]
    
    return [len(gamestate.projectiles)] + reach_projectiles

def ping(sock: socket.socket, payload_size: int):
    while True:
        if sock.recv(payload_size):
            break
    sock.send(b'\0' * payload_size)

def get_match_settings(sock: socket.socket, melee: MeleeInstance.Melee):
    settings_spinner = Halo(text='Waiting for match settings', spinner='dots')
    settings_spinner.start()
    while True:
        payload = sock.recv(10)
        if payload:
            settings_spinner.succeed()
            settings_spinner.stop()
            stage, ppo_character, cpu_character, cpu_level, n_projectiles = struct.unpack("hhhhh", payload)
            print(f"{tcolors.OKGREEN}{melee.get_player(ppo_character).name}{tcolors.ENDC} VS {tcolors.FAIL}{melee.get_player(cpu_character)}{tcolors.ENDC} "
                   +f"level {tcolors.BOLD}{cpu_level}{tcolors.ENDC} on stage {tcolors.OKBLUE}{melee.get_stage(stage).name}{tcolors.ENDC}")
            return stage, ppo_character, cpu_character, cpu_level, n_projectiles

def send_infos(sock: socket.socket, ppo_position, cpu_position, blastzones, edge, edge_ground, right_platform, left_platform, top_platform):
    info_payload = struct.pack("fffffffffffffffffffff", *ppo_position, *cpu_position, *blastzones, *edge, *edge_ground, *right_platform, *left_platform, *top_platform)
    sock.send(info_payload[:84])

def send_observation(payload_char:str, payload_size:int, projectiles:list, blastzones, sock:socket.socket, gamestate:melee.GameState, done:bool):
    payload = struct.pack(payload_char, 
                        int(gamestate.frame),
                        done,

                        float(clamp(gamestate.players[1].position.x, blastzones[0], blastzones[1])),
                        float(clamp(gamestate.players[1].position.y, blastzones[3], blastzones[2])),
                        float(clamp(gamestate.players[melee_match.cpuController.port].position.x, blastzones[0], blastzones[1])),
                        float(clamp(gamestate.players[melee_match.cpuController.port].position.y, blastzones[3], blastzones[2])),
                                      
                        int(gamestate.players[1].percent),
                        int(gamestate.players[melee_match.cpuController.port].percent),
                          
                        float(gamestate.players[1].shield_strength),
                        float(gamestate.players[melee_match.cpuController.port].shield_strength),
                                      
                        bool(gamestate.players[1].is_powershield),
                        bool(gamestate.players[melee_match.cpuController.port].is_powershield),
                                      
                        int(gamestate.players[1].stock),
                        int(gamestate.players[melee_match.cpuController.port].stock),
                                      
                        bool(gamestate.players[1].facing),
                        bool(gamestate.players[melee_match.cpuController.port].facing),
                                      
                        int(gamestate.players[1].action_frame),
                        int(gamestate.players[melee_match.cpuController.port].action_frame),

                        int(gamestate.players[1].action.value),
                        int(gamestate.players[melee_match.cpuController.port].action.value),
                            
                        bool(gamestate.players[1].invulnerable),
                        bool(gamestate.players[melee_match.cpuController.port].invulnerable),
                                    
                        int(gamestate.players[1].invulnerability_left),
                        int(gamestate.players[melee_match.cpuController.port].invulnerability_left),
                                    
                        int(gamestate.players[1].hitlag_left),
                        int(gamestate.players[melee_match.cpuController.port].hitlag_left),
                                    
                        int(gamestate.players[1].hitstun_frames_left),
                        int(gamestate.players[melee_match.cpuController.port].hitstun_frames_left),
                                    
                        int(gamestate.players[1].jumps_left),
                        int(gamestate.players[melee_match.cpuController.port].jumps_left),
                                    
                        bool(gamestate.players[1].on_ground),
                        bool(gamestate.players[melee_match.cpuController.port].on_ground),
                                    
                        float(gamestate.players[1].speed_air_x_self),
                        float(gamestate.players[melee_match.cpuController.port].speed_air_x_self),
                                    
                        float(gamestate.players[1].speed_y_self),
                        float(gamestate.players[melee_match.cpuController.port].speed_y_self),
                                    
                        float(gamestate.players[1].speed_x_attack),
                        float(gamestate.players[melee_match.cpuController.port].speed_x_attack),
                                    
                        float(gamestate.players[1].speed_y_attack),
                        float(gamestate.players[melee_match.cpuController.port].speed_y_attack),
                                    
                        bool(gamestate.players[1].moonwalkwarning),
                        bool(gamestate.players[melee_match.cpuController.port].moonwalkwarning),
                                    
                        float(gamestate.players[1].ecb.top.x),
                        float(gamestate.players[melee_match.cpuController.port].ecb.top.x),
                                    
                        float(gamestate.players[1].ecb.top.y),
                        float(gamestate.players[melee_match.cpuController.port].ecb.top.y),
                                    
                        float(gamestate.players[1].ecb.bottom.x),
                        float(gamestate.players[melee_match.cpuController.port].ecb.bottom.x),
                                    
                        float(gamestate.players[1].ecb.bottom.y),
                        float(gamestate.players[melee_match.cpuController.port].ecb.bottom.y),
                                    
                        float(gamestate.players[1].ecb.left.x),
                        float(gamestate.players[melee_match.cpuController.port].ecb.left.x),
                                    
                        float(gamestate.players[1].ecb.left.y),
                        float(gamestate.players[melee_match.cpuController.port].ecb.left.y),
                                    
                        float(gamestate.players[1].ecb.right.x),
                        float(gamestate.players[melee_match.cpuController.port].ecb.right.x),
                                    
                        float(gamestate.players[1].ecb.right.y),
                        float(gamestate.players[melee_match.cpuController.port].ecb.right.y),
                                    
                        *projectiles)
    
    sock.send(payload[:payload_size])

def controller_bool_2_dolphin(btn_state: int):
    if btn_state == 1:
        return "PRESS"
    else:
        return "RELEASE"

def actions_2_console(actions: tuple, controller:melee.Controller):
    command = f"{controller_bool_2_dolphin(actions[1])} A" + "\n"
    command += f"{controller_bool_2_dolphin(actions[2])} B" + "\n"
    command += f"{controller_bool_2_dolphin(actions[3])} X" + "\n"
    command += f"{controller_bool_2_dolphin(actions[4])} Y" + "\n"
    command += f"{controller_bool_2_dolphin(actions[5])} Z" + "\n"
    command += f"{controller_bool_2_dolphin(int(actions[14]))} L" + "\n"
    command += f"{controller_bool_2_dolphin(int(actions[15]))} R" + "\n"
    command += f"{controller_bool_2_dolphin(actions[6])} D_UP" + "\n"
    command += f"{controller_bool_2_dolphin(actions[7])} D_DOWN" + "\n"
    command += f"{controller_bool_2_dolphin(actions[8])} D_LEFT" + "\n"
    command += f"{controller_bool_2_dolphin(actions[9])} D_RIGHT" + "\n"
    command += f"SET MAIN {actions[10]} {actions[11]}" + "\n"
    command += f"SET C {actions[12]} {actions[13]}" + "\n"
    command += f"SET L {actions[14]}" + "\n"
    command += f"SET R {actions[15]}" + "\n"

    controller._write(command)
    controller.flush()

def get_actions(sock: socket.socket):
    action_payload_char = "iiiiiiiiiiffffff"
    action_payload_size = struct.calcsize(action_payload_char)

    while True:
        payload = sock.recv(action_payload_size)
        if payload:
            actions = struct.unpack(action_payload_char, payload)
            break

    return actions

def process_actions(actions: tuple, melee_match:MeleeInstance.Melee):
    options = actions[0]
    if options == 0:
        actions_2_console(actions, melee_match.ppoController)
    elif options == 1:
        melee_match.pause()
    elif options == 2:
        melee_match.resume()
    elif options == 3:
        melee_match.reset()
    
def game_loop(melee_match: MeleeInstance.Melee, sock: socket.socket):
    sock.send(b'\0'*25)

    stage, ppo_character, cpu_character, cpu_level, n_projectiles = get_match_settings(sock, melee_match)

    ppo_position, cpu_position, blastzones, edge, edge_ground, right_platform, left_platform, top_platform = melee_match.game_init(stage, ppo_character, cpu_character, cpu_level)

    send_infos(sock, ppo_position, cpu_position, blastzones, edge, edge_ground, right_platform, left_platform, top_platform)

    observation_payload_char = "l?ffffiiff??ii??iiii??iiiiiiii??ffffffff??ffffffffffffffffi" + "ffffiiii"*n_projectiles
    observation_payload_size = struct.calcsize(observation_payload_char)

    while True:
        gamestate = melee_match.console.step()
        if gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
            projectiles = get_projectiles(gamestate, n_projectiles, blastzones)
            send_observation(observation_payload_char, observation_payload_size, projectiles, blastzones, sock, gamestate, True)
            break

        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            projectiles = get_projectiles(gamestate, n_projectiles, blastzones)
            send_observation(observation_payload_char, observation_payload_size, projectiles, blastzones, sock, gamestate, False)

            actions = get_actions(sock)
            process_actions(actions, melee_match)
            if actions[0] == 3:
                break

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect(sock, "172.16.1.32", 8888)

    melee_match = MeleeInstance.Melee(1, 2, "/home/jul/Downloads/SSBM.iso", "/home/jul/.config/Slippi Launcher/netplay/squashfs-root/usr/bin", False)

    while True:
        try:
            game_loop(melee_match, sock)
        except KeyboardInterrupt:
            melee_match.console.stop()
            sock.close()
            break