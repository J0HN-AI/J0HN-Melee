from pyfiglet import Figlet
import threading as th
from halo import Halo
import multiprocessing as mp

import pathlib
import socket
import struct
import tomli
import os
import time

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

def get_stage(hex_stage):
    match hex_stage:
        case 0x0:
            return "BATTLEFIELD"
        case 0x1:
            return "FINAL_DESTINATION"
        case 0x2:
            return "DREAMLAND"
        case 0x3:
            return "FOUNTAIN_OF_DREAMS"
        case 0x4:
            return "POKEMON_STADIUM"
        case 0x5:
            return "YOSHIS_STORY"
            
def get_player(hex_player):
    match hex_player:
        case 0x00:
            return "DOC"
        case 0x01:
            return "MARIO"
        case 0x02:
            return "LUIGI"
        case 0x03:
            return "BOWSER"
        case 0x04:
            return "PEACH"
        case 0x05:
            return "YOSHI"
        case 0x06:
            return "DK"
        case 0x07:
            return "CPTFALCON"
        case 0x08:
            return "GANONDORF"
        case 0x09:
            return "FALCO"
        case 0x0a:
            return "FOX"
        case 0x0b:
            return "NESS"
        case 0x0c:
            return "POPO"
        case 0x0d:
            return "KIRBY"
        case 0x0e:
            return "SAMUS"
        case 0x0f:
            return "ZELDA"
        case 0x10:
            return "LINK"
        case 0x11:
            return "YLINK"
        case 0x12:
            return "PICHU"
        case 0x13:
            return "PIKACHU"
        case 0x14:
            return "JIGGLYPUFF"
        case 0x15:
            return "MEWTWO"
        case 0x16:
            return "GAMEANDWATCH"
        case 0x17:
            return "MARTH"
        case 0x18:
            return "ROY"

def client_2_loopback_router(loopback_socket:socket.socket, client_socket:socket.socket, stop:th.Event, n_projectiles:int):
    payload_char = "l?ffffiiff??ii??iiii??iiiiiiii??ffffffff??ffffffffffffffffi" + "ffffiiii"*n_projectiles
    payload_size = struct.calcsize(payload_char)

    try:
        while not stop.is_set():
            payload = client_socket.recv(payload_size)
            loopback_socket.send(payload)
    except OSError as e:
        client_socket.close()

def loopback_2_client_router(loopback_socket:socket.socket, client_socket:socket.socket, stop:th.Event):
    payload_char = "iiiiiiiiiiffffff"
    payload_size = struct.calcsize(payload_char)
    
    try:
        while not stop.is_set():
            payload = loopback_socket.recv(payload_size)
            client_socket.send(payload)
    except OSError as e:
        loopback_socket.close()
    
def loopback_2_logs(loopback_socket:socket.socket, logs_dict, int_addr, stop, lock):
    payload_char = "fiiiiiiiiffi"
    payload_size = struct.calcsize(payload_char)

    while not stop.is_set():
        log_payload = loopback_socket.recv(payload_size)
        log = struct.unpack(payload_char, log_payload)
        instance_log = {
            str(int_addr[1]): {
                "game": log[8],
                "game_time": round(log[0], 3),
                "stage": get_stage(log[1]),
                "agent": get_player(log[2]),
                "agent_percent": log[3],
                "agent_stock": log[4],
                "CPU": get_player(log[5]),
                "CPU_percent": log[6],
                "CPU_stock": log[7],
                "score": round(log[9], 5),
                "avg_score": round(log[10], 5),
                "learn_iters": log[11]
                }
        }

        with lock:
            logs_dict.update(instance_log)

def logs_2_terminal(logs_dict, lock, nb_instances:int):
    logs = {}
    old_logs = {}

    with lock:
        logs = dict(logs_dict)
    
    if old_logs != logs:
        print(logs)
        old_logs = logs

def setup_rooting_table(config:dict):
    routing_table = {}
    for i, instance in instances_configuration.items():
        routing_table[config["network-config"]["envs_base_port"] + int(i)] = instance["ip"]
    
    return routing_table

def setup_loopback_int(network_config:dict, nb_instances:int):
    try:
        loopback_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        loopback_int_socket.bind(("127.0.0.1", network_config["router_port"]))
        loopback_int_socket.listen(nb_instances)

        return loopback_int_socket
    except OSError as e:
        print(e)

def setup_loopback_connections(loopback_int_socket: socket.socket, routing_table: dict, nb_instances:int):
    socket_table = {}
    try:
        loopback_spinner = Halo(f"{tcolors.WARNING}Waiting for loopback instances 0/{nb_instances}{tcolors.ENDC}", spinner="dots")
        loopback_spinner.start()
        loopback_instances_connected = 0

        while loopback_instances_connected != nb_instances:
            loopback, addr = loopback_int_socket.accept()
            socket_table[routing_table.get(addr[1])] = loopback
            loopback_instances_connected += 1
            loopback_spinner.text = f"{tcolors.WARNING}Waiting for loopback instances {loopback_instances_connected}/{nb_instances}{tcolors.ENDC}"

        loopback_spinner.succeed(f"{tcolors.OKGREEN}All loopback instances are connected ~(^-^)~{tcolors.ENDC}")
    except KeyboardInterrupt:
            loopback_spinner.fail(f"{tcolors.FAIL}Only {loopback_instances_connected} loopback instance/s are connected (╯°□°）╯︵ ┻━┻{tcolors.ENDC}")
    
    return socket_table

def setup_client_int(network_config:dict, nb_instances:int):
    try:
        client_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_int_socket.bind((network_config["HPC_ip"], network_config["HPC_port"]))
        client_int_socket.listen(nb_instances)
        
        return client_int_socket
    except OSError as e:
        print(e)

def setup_client_connections(client_int_socket:socket.socket, socket_table:dict, stop_routers:th.Event, nb_projectiles:int, nb_instances:int):
    try:
        clients_spinner = Halo(f"{tcolors.WARNING}Waiting for client instances 0/{nb_instances}{tcolors.ENDC}", spinner="dots")
        clients_spinner.start()
        clients_instances_connected = 0

        while clients_instances_connected != nb_instances:
            client, addr = client_int_socket.accept()
            th.Thread(target=client_2_loopback_router, args=(socket_table.get(addr[0]), client, stop_routers, nb_projectiles)).start()
            th.Thread(target=loopback_2_client_router, args=(socket_table.get(addr[0]), client, stop_routers)).start()

            clients_instances_connected += 1
            clients_spinner.text = f"{tcolors.WARNING}Waiting for client instances {clients_instances_connected}/{nb_instances}{tcolors.ENDC}"

        clients_spinner.succeed(f"{tcolors.OKGREEN}All client instances are connected ~(^-^)~{tcolors.ENDC}")
    except KeyboardInterrupt:
            clients_spinner.fail(f"{tcolors.FAIL}Only {clients_instances_connected} client instance/s are connected (╯°□°）╯︵ ┻━┻{tcolors.ENDC}")

def setup_logger_int(network_config:dict, nb_instances:int):
    try:
        logger_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger_int_socket.bind(("127.0.0.1", network_config["logger_port"]))
        logger_int_socket.listen(nb_instances)

        return logger_int_socket
    except OSError as e:
        print(e)

def setup_logger_connections(logger_int_socket: socket.socket, logs_dict, stop_logger, lock, nb_instances:int):
    try:
        logger_spinner = Halo(f"{tcolors.WARNING}Waiting for loopback instances 0/{nb_instances}{tcolors.ENDC}", spinner="dots")
        logger_spinner.start()
        logger_instances_connected = 0

        while logger_instances_connected != nb_instances:
            loopback, addr = logger_int_socket.accept()
            th.Thread(target=loopback_2_logs, args=(loopback, logs_dict, addr, stop_logger, lock)).start()
            logger_instances_connected += 1
            logger_spinner.text = f"{tcolors.WARNING}Waiting for loopback instances {logger_instances_connected}/{nb_instances}{tcolors.ENDC}"

        logger_spinner.succeed(f"{tcolors.OKGREEN}All loopback instances are connected ~(^-^)~{tcolors.ENDC}")
    except KeyboardInterrupt:
            logger_spinner.fail(f"{tcolors.FAIL}Only {logger_instances_connected} loopback instance/s are connected (╯°□°）╯︵ ┻━┻{tcolors.ENDC}")

if __name__ == "__main__":
    print(Figlet(font="larry3d").renderText("J0HN Melee"))
    print()

    config = tomli.load(open(f"{pathlib.Path(__file__).parent.resolve()}/config.toml", "rb"))

    network_config = config["network-config"]
    instances_configuration = config["instances"]
    nb_projectiles = config["training-config"]["n_projectiles"]
    nb_instances = len(instances_configuration)

    stop_routers = th.Event()

    stop_loggers = mp.Event()
    loggers_manager = mp.Manager()
    logs_dict = loggers_manager.dict()
    logs_lock = loggers_manager.Lock()

    routing_table = setup_rooting_table(config)

    loopback_int_socket = setup_loopback_int(network_config, nb_instances)
    socket_table = setup_loopback_connections(loopback_int_socket, routing_table, nb_instances)

    logger_int_socket = setup_logger_int(network_config, nb_instances)
    setup_logger_connections(logger_int_socket, logs_dict, stop_loggers, logs_lock, nb_instances)
    
    client_int_socket = setup_client_int(network_config, nb_instances)
    setup_client_connections(client_int_socket, socket_table, stop_routers, nb_projectiles, nb_instances)
    
    while True:
        try:
            logs_2_terminal(logs_dict, logs_lock, nb_instances)
        except KeyboardInterrupt:
            stop_routers.set()
            stop_loggers.set()
            client_int_socket.close()
            loopback_int_socket.close()
            logger_int_socket.close()
            del logs_lock
            break