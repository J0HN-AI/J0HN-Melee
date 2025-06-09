import tomli
from pyfiglet import Figlet
import pathlib
import socket
import struct
import threading as th
from halo import Halo

def get_stage(stage):
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
    
def get_player(hex_player):
        match hex_player:
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

def client_2_loopback_router(loopback_socket:socket.socket, client_socket:socket.socket, stop:th.Event, n_projectiles:int):
    payload_char = "iffffiiff??ii??iiii??hhhhhhhh??ffffffff??ffffffffffffffffh" + "ffffhhhh"*n_projectiles
    payload_size = struct.calcsize(payload_char)

    while not stop.is_set():
        payload = client_socket.recv(payload_size)
        loopback_socket.send(payload)
    client_socket.close()

def loopback_2_client_router(loopback_socket:socket.socket, client_socket:socket.socket, stop:th.Event, n_projectiles:int):
    #payload_char = "iffffiiff??ii??iiii??hhhhhhhh??ffffffff??ffffffffffffffffh" + "ffffhhhh"*n_projectiles
    payload_char = "i"
    payload_size = struct.calcsize(payload_char)
    
    while not stop.is_set():
        payload = loopback_socket.recv(payload_size)
        client_socket.send(payload)
    loopback_socket.close()

if __name__ == "__main__":
    print(Figlet(font="larry3d").renderText("J0HN Melee"))
    print()

    config = tomli.load(open(f"{pathlib.Path(__file__).parent.resolve()}/config.toml", "rb"))

    network_config = config["network-config"]
    instances_configuration = config["instances"]
    nb_projectiles = config["training-config"]["n_projectiles"]
    nb_instances = len(instances_configuration)

    routing_table = {}
    socket_table = {}
    stop_routers = th.Event()

    for i, instance in instances_configuration.items():
        routing_table[config["network-config"]["envs_base_port"] + int(i)] = instance["ip"]


    loopback_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    loopback_int_socket.bind(("127.0.0.1", network_config["router_port"]))
    loopback_int_socket.listen(nb_instances)
    
    try:
        loopback_spinner = Halo(f"Waiting for loopback instances 0/{nb_instances}", spinner="dots")
        loopback_spinner.start()
        loopback_instances_connected = 0

        while loopback_instances_connected != nb_instances:
            loopback, addr = loopback_int_socket.accept()
            socket_table[routing_table.get(addr[1])] = loopback
            loopback_instances_connected += 1
            loopback_spinner.text = f"Waiting for loopback instances {loopback_instances_connected}/{nb_instances}"

        loopback_spinner.succeed("All instances loopback connected ~(^-^)~")
    except KeyboardInterrupt:
            loopback_spinner.fail(f"Only {loopback_instances_connected} loopback instance/s are connected (╯°□°）╯︵ ┻━┻")
    

    client_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_int_socket.bind((network_config["HPC_ip"], network_config["HPC_port"]))
    client_int_socket.listen(nb_instances)

    try:
        clients_spinner = Halo(f"Waiting for client instances 0/{nb_instances}", spinner="dots")
        clients_spinner.start()
        clients_instances_connected = 0

        while clients_instances_connected != nb_instances:
            client, addr = client_int_socket.accept()
            th.Thread(target=client_2_loopback_router, args=(socket_table.get(addr[0]), client, stop_routers, nb_projectiles)).start()
            th.Thread(target=loopback_2_client_router, args=(socket_table.get(addr[0]), client, stop_routers, nb_projectiles)).start()

            clients_instances_connected += 1
            clients_spinner.text = f"Waiting for client instances {clients_instances_connected}/{nb_instances}"

        clients_spinner.succeed("All client instances connected ~(^-^)~")
    except KeyboardInterrupt:
            clients_spinner.fail(f"Only {clients_instances_connected} client instance/s are connected (╯°□°）╯︵ ┻━┻")

    try:
        print("Router Test is UP")
        while True:
            ()
    except KeyboardInterrupt:
        stop_routers.set()
        client_int_socket.close()
        loopback_int_socket.close()