from pyfiglet import Figlet
import threading as th
from halo import Halo

import pathlib
import socket
import struct
import tomli


def client_2_loopback_router(loopback_socket:socket.socket, client_socket:socket.socket, stop:th.Event, n_projectiles:int):
    payload_char = "l?ffffiiff??ii??iiii??iiiiiiii??ffffffff??ffffffffffffffffi" + "ffffiiii"*n_projectiles
    payload_size = struct.calcsize(payload_char)

    while not stop.is_set():
        payload = client_socket.recv(payload_size)
        loopback_socket.send(payload)
    client_socket.close()

def loopback_2_client_router(loopback_socket:socket.socket, client_socket:socket.socket, stop:th.Event):
    payload_char = "iiiiiiiiiiffffff"
    payload_size = struct.calcsize(payload_char)
    
    while not stop.is_set():
        payload = loopback_socket.recv(payload_size)
        client_socket.send(payload)
    loopback_socket.close()

def setup_rooting_table(config:dict):
    routing_table = {}
    for i, instance in instances_configuration.items():
        routing_table[config["network-config"]["envs_base_port"] + int(i)] = instance["ip"]
    
    return routing_table

def setup_loopback_int(network_config:dict, nb_instances:int):
    loopback_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    loopback_int_socket.bind(("127.0.0.1", network_config["router_port"]))
    loopback_int_socket.listen(nb_instances)

    return loopback_int_socket

def setup_loopback_connections(loopback_int_socket: socket.socket, routing_table: dict, nb_instances:int):
    socket_table = {}
    try:
        loopback_spinner = Halo(f"Waiting for loopback instances 0/{nb_instances}", spinner="dots")
        loopback_spinner.start()
        loopback_instances_connected = 0

        while loopback_instances_connected != nb_instances:
            loopback, addr = loopback_int_socket.accept()
            socket_table[routing_table.get(addr[1])] = loopback
            loopback_instances_connected += 1
            loopback_spinner.text = f"Waiting for loopback instances {loopback_instances_connected}/{nb_instances}"

        loopback_spinner.succeed("All loopback instances are connected ~(^-^)~")
    except KeyboardInterrupt:
            loopback_spinner.fail(f"Only {loopback_instances_connected} loopback instance/s are connected (╯°□°）╯︵ ┻━┻")
    
    return socket_table

def setup_client_int(network_config:dict, nb_instances:int):
    client_int_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_int_socket.bind((network_config["HPC_ip"], network_config["HPC_port"]))
    client_int_socket.listen(nb_instances)
    
    return client_int_socket

def setup_client_connections(client_int_socket:socket.socket, socket_table:dict, stop_routers:th.Event, nb_projectiles:int, nb_instances:int):
    try:
        clients_spinner = Halo(f"Waiting for client instances 0/{nb_instances}", spinner="dots")
        clients_spinner.start()
        clients_instances_connected = 0

        while clients_instances_connected != nb_instances:
            client, addr = client_int_socket.accept()
            th.Thread(target=client_2_loopback_router, args=(socket_table.get(addr[0]), client, stop_routers, nb_projectiles)).start()
            th.Thread(target=loopback_2_client_router, args=(socket_table.get(addr[0]), client, stop_routers)).start()

            clients_instances_connected += 1
            clients_spinner.text = f"Waiting for client instances {clients_instances_connected}/{nb_instances}"

        clients_spinner.succeed("All client instances are connected ~(^-^)~")
    except KeyboardInterrupt:
            clients_spinner.fail(f"Only {clients_instances_connected} client instance/s are connected (╯°□°）╯︵ ┻━┻")

if __name__ == "__main__":
    print(Figlet(font="larry3d").renderText("J0HN Melee"))
    print()

    config = tomli.load(open(f"{pathlib.Path(__file__).parent.resolve()}/config.toml", "rb"))

    network_config = config["network-config"]
    instances_configuration = config["instances"]
    nb_projectiles = config["training-config"]["n_projectiles"]
    nb_instances = len(instances_configuration)

    stop_routers = th.Event()

    routing_table = setup_rooting_table(config)

    loopback_int_socket = setup_loopback_int(network_config, nb_instances)
    socket_table = setup_loopback_connections(loopback_int_socket, routing_table, nb_instances)
    
    client_int_socket = setup_client_int(network_config, nb_instances)
    setup_client_connections(client_int_socket, socket_table, stop_routers, nb_projectiles, nb_instances)
    

    try:
        print("Router Test is UP")
        while True:
            pass
    except KeyboardInterrupt:
        stop_routers.set()
        client_int_socket.close()
        loopback_int_socket.close()