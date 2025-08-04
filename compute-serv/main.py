from pyfiglet import Figlet
import threading as th
from halo import Halo
import multiprocessing as mp
from rich.console import Console
from rich.table import Table
from rich.live import Live
from datetime import  timedelta
from select import select
import pandas as pd

import pathlib
import socket
import struct
import tomli
import time
import sys
import tty
import sys
import termios
import os

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
    
def loopback_2_logs(loopback_socket:socket.socket, logs_dict, int_addr, config, stop, lock):
    payload_char = "fiiiiiiiiiffiiiiiiii"
    payload_size = struct.calcsize(payload_char)
    log_id_mode = config["backup-logs"]["show_instances_id_as"]
    logger_base_port = config["network-config"]["envs_logger_base_port"]
    clients_base_port = config["network-config"]["envs_base_port"]
    instances_config = config["instances"]
    instance_id = ""

    while not stop.is_set():
        log_payload = loopback_socket.recv(payload_size)
        log = struct.unpack(payload_char, log_payload)

        instance_base_id = (int(int_addr[1]) - logger_base_port)
        if log_id_mode == "IP":
            instance_id = instances_config[str(instance_base_id)]["ip"]
        elif log_id_mode == "PORT":
            instance_id = str(instance_base_id + clients_base_port)
        else: #ID option
            instance_id = str(instance_base_id)

        instance_log = {
            instance_id: {
                "game": log[9],
                "game_time": round(log[0], 3),
                "stage": get_stage(log[1]),
                "agent": get_player(log[2]),
                "agent_percent": log[3],
                "agent_stock": log[4],
                "CPU": get_player(log[5]),
                "CPU_percent": log[6],
                "CPU_stock": log[7],
                "frame": log[8],
                "score": round(log[10], 5),
                "avg_score": round(log[11], 5),
                "learn_iters": log[12],
                "epoch": log[13],
                "max_epoch": log[14],
                "victory": log[15],
                "defeat": log[16],
                "cpu_level": log[17],
                "learn_mode": bool(log[18]),
                "done": bool(log[19])
                }
        }

        with lock:
            logs_dict.update(instance_log)

def logs_2_terminal(logs_dict, lock, current_page:int = 1, terminal_height:int = 42, n_epochs:int=20):
    instance_log = {}
    current_logs = {}

    with lock:
        current_logs = dict(logs_dict)

    if current_logs != {}:
        instance_log = current_logs
    else:
        instance_log = {
            "N/A": {
                "game": -1,
                "game_time": -1,
                "stage": "N/A",
                "agent": "N/A",
                "agent_percent": -1,
                "agent_stock": -1,
                "CPU": "N/A",
                "CPU_percent": -1,
                "CPU_stock": -1,
                "score": -1,
                "avg_score": -1,
                "learn_iters": -1,
                "epoch": -1,
                "max_epoch": -1,
                "victory": -1,
                "defeat": -1,
                "cpu_level": -1
            },
        }

    pages = generate_pages(instance_log, terminal_height)

    table = Table(title="J0HN MELEE Stats", title_style="bold #9ece6a", caption_style="#7aa2f7", caption=f"<a {current_page}/{len(pages)} d>")
        
    table.add_column("Instance\n", header_style="bold #bb9af7", style="#f7768e")
    table.add_column("Game\n", header_style="bold #bb9af7", style="#c0caf5", justify="center")
    table.add_column("Time elaped\n(MM:SS:MS)", header_style="bold #bb9af7", style="#9aa5ce")
    table.add_column("Stage\n", header_style="bold #bb9af7", style="#7aa2f7", min_width=18, justify="center")
    table.add_column("Agent\n", header_style="bold #bb9af7", style="#a3be8c", justify="center")
    table.add_column("Agent\nPercent", header_style="bold #bb9af7", style="#a3be8c", min_width=4, justify="right")
    table.add_column("Agent\nStock", header_style="bold #bb9af7", style="#a3be8c", justify="center")
    table.add_column("CPU\n", header_style="bold #bb9af7", style="#ebcb8b", justify="center")
    table.add_column("CPU\nPercent", header_style="bold #bb9af7", style="#ebcb8b", min_width=4, justify="right")
    table.add_column("CPU\nStock", header_style="bold #bb9af7", style="#ebcb8b", justify="center")
    table.add_column("CPU\nLevel", header_style="bold #bb9af7", style="#d08770", justify="center")
    table.add_column("Victory\n", header_style="bold #bb9af7", style="#a3be8c", justify="center")
    table.add_column("Defeat\n", header_style="bold #bb9af7", style="#bf616a", justify="center")
    table.add_column("Score\n", header_style="bold #bb9af7", style="#9ece6a", min_width=14)
    table.add_column("Average\nScore", header_style="bold #bb9af7", style="#9ece6a", justify="center")
    table.add_column("Learning\nIterations", header_style="bold #bb9af7", style="#73daca", justify="center")
    table.add_column("Epochs\n", header_style="bold #bb9af7", style="#ff9e64", min_width=len(str(n_epochs))*2+1, justify="right")
    

    for key in pages[str(current_page)]:
        log_data = instance_log[key]
        td = timedelta(seconds=log_data["game_time"])
        hh, mm, ss = str(td).split(":") 
        ss, ms = str(float(ss)).split(".")
        table.add_row(key, str(log_data["game"]), f"{mm}:{ss}:{ms}", log_data["stage"], 
                    log_data["agent"], f"{log_data["agent_percent"]}%", str(log_data["agent_stock"]),
                    log_data["CPU"], f"{log_data["CPU_percent"]}%", str(log_data["CPU_stock"]),
                    str(log_data["cpu_level"]), str(log_data["victory"]), str(log_data["defeat"]),
                    str(log_data["score"]), str(log_data["avg_score"]), str(log_data["learn_iters"]),
                    f"{log_data["epoch"]}/{log_data["max_epoch"]}")
        table.add_section()
    
    return table

def setup_rooting_table(config:dict):
    routing_table = {}
    for i, instance in instances_configuration.items():
        routing_table[config["network-config"]["envs_base_port"] + int(i)] = instance["ip"]
    
    return routing_table

def setup_loopback_int(network_config:dict, nb_instances:int):
    bind_retry_timeout = network_config["bind_retry_timeout"]
    while True:
        try:
            loopback_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            loopback_int_socket.bind(("127.0.0.1", network_config["router_port"]))
            loopback_int_socket.listen(nb_instances)

            return loopback_int_socket
        except OSError:
            print(f"{tcolors.BOLD}{tcolors.FAIL}Unable to bind loopback interface retrying in {bind_retry_timeout}s{tcolors.ENDC}")
            time.sleep(bind_retry_timeout)

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
    bind_retry_timeout = network_config["bind_retry_timeout"]
    while True:
        try:
            client_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_int_socket.bind((network_config["HPC_ip"], network_config["HPC_port"]))
            client_int_socket.listen(nb_instances)
            
            return client_int_socket
        except OSError:
            print(f"{tcolors.BOLD}{tcolors.FAIL}Unable to bind client interface retrying in {bind_retry_timeout}s{tcolors.ENDC}")
            time.sleep(bind_retry_timeout)

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
    bind_retry_timeout = network_config["bind_retry_timeout"]
    while True:
        try:
            logger_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger_int_socket.bind(("127.0.0.1", network_config["logger_port"]))
            logger_int_socket.listen(nb_instances)

            return logger_int_socket
        except OSError:
            print(f"{tcolors.BOLD}{tcolors.FAIL}Unable to bind logger interface retrying in {bind_retry_timeout}s{tcolors.ENDC}")
            time.sleep(bind_retry_timeout)

def setup_logger_connections(logger_int_socket: socket.socket, logs_dict, config, stop_logger, lock, nb_instances:int):
    try:
        logger_spinner = Halo(f"{tcolors.WARNING}Waiting for loopback instances 0/{nb_instances}{tcolors.ENDC}", spinner="dots")
        logger_spinner.start()
        logger_instances_connected = 0

        while logger_instances_connected != nb_instances:
            loopback, addr = logger_int_socket.accept()
            th.Thread(target=loopback_2_logs, args=(loopback, logs_dict, addr, config, stop_logger, lock)).start()
            logger_instances_connected += 1
            logger_spinner.text = f"{tcolors.WARNING}Waiting for loopback instances {logger_instances_connected}/{nb_instances}{tcolors.ENDC}"

        logger_spinner.succeed(f"{tcolors.OKGREEN}All loopback instances are connected ~(^-^)~{tcolors.ENDC}")
    except KeyboardInterrupt:
            logger_spinner.fail(f"{tcolors.FAIL}Only {logger_instances_connected} loopback instance/s are connected (╯°□°）╯︵ ┻━┻{tcolors.ENDC}")

def generate_pages(instance_log:dict, terminal_height:int = 42):
    pages = {}
    max_instances_per_page = (terminal_height - 7)//2 + 1
    instances_id = list(instance_log.keys())

    n_page = 1
    for i in range(0, len(instances_id), max_instances_per_page):
        pages[str(n_page)] = instances_id[i:i + max_instances_per_page]
        n_page += 1

    return pages

def read_key_nonblocking():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select([sys.stdin], [], [], 0)  # timeout=0 for non-blocking
        if rlist:
            c = sys.stdin.read(1)
            if c == '\x1b':  # Escape sequence (arrow keys)
                # Read next two chars if available
                rlist2, _, _ = select([sys.stdin], [], [], 0)
                if rlist2:
                    c += sys.stdin.read(1)
                rlist3, _, _ = select([sys.stdin], [], [], 0)
                if rlist3:
                    c += sys.stdin.read(1)
            return c
        else:
            return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_max_pages(logs_dict, logs_lock):
    instance_log = {}
    current_logs = {}

    with logs_lock:
        current_logs = dict(logs_dict)

    if current_logs != {}:
        instance_log = current_logs
    else:
        instance_log = {
            "N/A": {
                "game": -1,
                "game_time": -1,
                "stage": "N/A",
                "agent": "N/A",
                "agent_percent": -1,
                "agent_stock": -1,
                "CPU": "N/A",
                "CPU_percent": -1,
                "CPU_stock": -1,
                "score": -1,
                "avg_score": -1,
                "learn_iters": -1,
                "epoch": -1,
                "max_epoch": -1,
                "victory": -1,
                "defeat": -1,
                "cpu_level": -1
            },
        }
    
    return (len(instance_log) + max_per_page - 1) // max_per_page

def make_dirs(path:str):
    needed_dirs = ["backup", "logs"]

    for ndir in needed_dirs:
        os.makedirs(f"{path}/{ndir}", exist_ok=True)

def get_logfiles(path:str, logs_filename:str, nb_instances:int, config:dict):
    log_files_buffer = {}
    log_file_buffer = None
    instances_config = config["instances"]
    instances_base_port = config["network-config"]["envs_base_port"]
    instance_ids_mode = config["backup-logs"]["show_instances_id_as"]
    instances_filter_mode = config["backup-logs"]["instances_filter_mode"]
    instances_filtered = config["backup-logs"]["instances_filtered"]

    local_time = time.localtime()
    
    for i in range(nb_instances):
        csv_filename = logs_filename.replace("[d]", str(local_time.tm_mday)).replace("[mo]", str(local_time.tm_mon)).replace("[y]", str(local_time.tm_year))
        if instances_filter_mode == "WHITELIST":
            if i not in instances_filtered:
                continue
        elif instances_filter_mode == "BLACKLIST":
            if i in instances_filtered:
                continue

        if csv_filename.find("[ID]") == -1:
            csv_filename = f"{csv_filename}{i}"
        else:
            if instance_ids_mode == "IP":
                csv_filename = csv_filename.replace("[ID]", instances_config[str(i)]["ip"])
            elif instance_ids_mode == "PORT":
                csv_filename = csv_filename.replace("[ID]", str(instances_base_port+i))
            else: # ID
                csv_filename = csv_filename.replace("[ID]", str(i))
        
        if os.path.exists(f"{path}/logs/{csv_filename}.csv"):
            log_file_buffer = open(f"{path}/logs/{csv_filename}.csv", 'a')
        else:
            log_file_buffer = open(f"{path}/logs/{csv_filename}.csv", 'w')
        
        if instance_ids_mode == "IP":
            log_files_buffer[instances_config[str(i)]["ip"]] = log_file_buffer
        elif instance_ids_mode == "PORT":
            log_files_buffer[str(instances_base_port+i)] = log_file_buffer
        else: # ID
            log_files_buffer[str(i)] = log_file_buffer

    return log_files_buffer

def close_logfiles(log_files_buffer:dict):
    for buffer in log_files_buffer.values():
        buffer.close()

def save_to_csv(logs_dict, logs_lock, log_files_buffer):
    if log_files_buffer != {}:
        current_logs = {}
        with logs_lock:
            current_logs = dict(logs_dict)

        for instance_id, buffer in log_files_buffer.items():
            log = current_logs.get(instance_id, False)
            if log:
                if not log["learn_mode"] and not log["done"]:
                    log_data = {
                        "game": log["game"],
                        "game_time": log["game_time"],
                        "stage": log["stage"],
                        "agent": log["agent"],
                        "agent_percent": log["agent_percent"],
                        "agent_stock": log["agent_stock"],
                        "CPU": log["CPU"],
                        "CPU_percent": log["CPU_percent"],
                        "CPU_stock": log["CPU_stock"],
                        "score": log["score"],
                        "avg_score": log["avg_score"],
                        "learn_iters": log["learn_iters"],
                        "victory": log["victory"],
                        "defeat": log["defeat"],
                        "cpu_level": log["cpu_level"],
                    }

                    df = pd.DataFrame(log_data, index=[log["frame"]])
                    df.to_csv(buffer, mode='a', header=not buffer.tell(), index=False)

if __name__ == "__main__":
    console = Console()
    print(Figlet(font="larry3d").renderText("J0HN Melee"))
    print()

    config = tomli.load(open(f"{pathlib.Path(__file__).parent.resolve()}/config.toml", "rb"))

    network_config = config["network-config"]
    instances_configuration = config["instances"]
    nb_projectiles = config["training-config"]["n_projectiles"]
    nb_instances = len(instances_configuration)
    n_epochs = config["training-config"]["n_epochs"]
    logs_path = config["backup-logs"]["data_path"]
    logs_filename = config["backup-logs"]["csv_logs_filename"]

    stop_routers = th.Event()

    stop_loggers = mp.Event()
    loggers_manager = mp.Manager()
    logs_dict = loggers_manager.dict()
    logs_lock = loggers_manager.Lock()
    current_page = 1

    routing_table = setup_rooting_table(config)

    loopback_int_socket = setup_loopback_int(network_config, nb_instances)
    socket_table = setup_loopback_connections(loopback_int_socket, routing_table, nb_instances)

    logger_int_socket = setup_logger_int(network_config, nb_instances)
    setup_logger_connections(logger_int_socket, logs_dict, config, stop_loggers, logs_lock, nb_instances)
    
    client_int_socket = setup_client_int(network_config, nb_instances)
    setup_client_connections(client_int_socket, socket_table, stop_routers, nb_projectiles, nb_instances)

    make_dirs(logs_path)
    log_files_buffer = get_logfiles(logs_path, logs_filename, nb_instances, config)

    now = time.time()
    
    try:
        with Live(logs_2_terminal(logs_dict, logs_lock, current_page),  screen=True, refresh_per_second=60) as live:
            while True:
                terminal_height = live.console.height
                max_per_page = (terminal_height - 7) // 2 + 1
                max_pages = get_max_pages(logs_dict, logs_lock)
                
                key = read_key_nonblocking()
                if key:
                    if ord(key[0]) == 100:  # d
                        current_page = current_page + 1 if current_page < max_pages else 1
                    elif ord(key[0]) == 97:  # a
                        current_page = current_page - 1 if current_page > 1 else max_pages
                
                live.update(logs_2_terminal(logs_dict, logs_lock, current_page, terminal_height))

                if (now + config["backup-logs"]["save_logs_every"]) <= time.time():
                    now = time.time()
                    save_to_csv(logs_dict, logs_lock, log_files_buffer)

    except KeyboardInterrupt:
        stop_routers.set()
        stop_loggers.set()
        client_int_socket.close()
        loopback_int_socket.close()
        logger_int_socket.close()
        close_logfiles(log_files_buffer)
        del logs_lock