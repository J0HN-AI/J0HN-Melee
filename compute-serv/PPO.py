import numpy as np
import torch.nn as nn
import torch as T
import torch.optim as optim
import torch.distributed as distrib
import torch.multiprocessing as mp
from torch.distributions import Beta
from torch.nn.parallel import DistributedDataParallel as DDP
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pandas as pd

import tomli
import pathlib
import os
import gym_env.melee_env
import time

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
                np.array(self.probs), np.array(self.vals), \
                np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, config:dict):
        super(ActorNetwork, self).__init__()
        activation_fcts = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "ReLU6": nn.ReLU6,
            "LeakyReLU": nn.LeakyReLU,
            "Softplus": nn.Softplus
        }
        
        activation_name = config["actor-model"]["activation"]
        hidden_layers = config["actor-model"]["hidden_layers"]
        add_LayerNorm = config["actor-model"]["add_LayerNorm"]

        hidden_input_size = hidden_layers[0]
        
        layers = []
        layers.append(nn.Linear(*input_dims, hidden_input_size))
        if add_LayerNorm:
            layers.append(nn.LayerNorm(hidden_input_size))
        layers.append(activation_fcts[activation_name]())
        
        for layer_size in hidden_layers:
            layers.append(nn.Linear(hidden_input_size, layer_size))
            if add_LayerNorm:
                layers.append(nn.LayerNorm(layer_size))
            layers.append(activation_fcts[activation_name]())
            hidden_input_size = layer_size

        self.shared = nn.Sequential(*layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_input_size, n_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_input_size, n_actions), nn.Softplus())

    def forward(self, state):
        x = self.shared(state)

        alpha = T.clamp(self.alpha_head(x), min=1e-5, max=20) 
        beta = T.clamp(self.beta_head(x), min=1e-5, max=20)

        if not T.isfinite(alpha).all():
            alpha = T.nan_to_num(alpha)
        
        if not T.isfinite(beta).all():
            beta = T.nan_to_num(beta)

        return Beta(alpha, beta)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, config:dict):
        super(CriticNetwork, self).__init__()
        activation_fcts = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "ReLU6": nn.ReLU6,
            "LeakyReLU": nn.LeakyReLU,
            "Softplus": nn.Softplus
        }
        
        activation_name = config["critic-model"]["activation"]
        hidden_layers = config["critic-model"]["hidden_layers"]
        add_LayerNorm = config["critic-model"]["add_LayerNorm"]

        hidden_input_size = hidden_layers[0]
        
        layers = []
        layers.append(nn.Linear(*input_dims, hidden_input_size))
        if add_LayerNorm:
            layers.append(nn.LayerNorm(hidden_input_size))
        layers.append(activation_fcts[activation_name]())
        
        for layer_size in hidden_layers:
            layers.append(nn.Linear(hidden_input_size, layer_size))
            if add_LayerNorm:
                layers.append(nn.LayerNorm(layer_size))
            layers.append(activation_fcts[activation_name]())
            hidden_input_size = layer_size
        
        layers.append(nn.Linear(hidden_input_size, 1))
        
        self.critic = nn.Sequential(*layers)

    def forward(self, state):
        value = self.critic(state)

        return value

def save_weights(actor, critic, config:dict, last=False):
    weights_dir_name = str(config["backup-logs"]["weights_dir_name"])
    data_path = config["backup-logs"]["data_path"]

    local_time = time.localtime()
    weights_dir_name = weights_dir_name.replace("[d]", str(local_time.tm_mday)).replace("[mo]", str(local_time.tm_mon)).replace("[y]", str(local_time.tm_year))
    weights_dir_name = weights_dir_name.replace("[h]", str(local_time.tm_hour)).replace("[mi]", str(local_time.tm_min)).replace("[s]", str(local_time.tm_sec))
    
    if last:
        weights_dir_name += "_LAST"

    os.makedirs(f"{data_path}/backup/{weights_dir_name}", exist_ok=True)
    T.save(actor.state_dict(), f"{data_path}/backup/{weights_dir_name}/actor.pt")
    T.save(critic.state_dict(), f"{data_path}/backup/{weights_dir_name}/critic.pt")

def load_weights(actor:ActorNetwork, critic:CriticNetwork, config:dict, gpu:int):
    data_path = config["backup-logs"]["data_path"]
    weights_to_load = config["backup-logs"]["weights_to_load"]

    list_subdirs = [file for file in os.scandir(f"{data_path}/backup") if file.is_dir()]

    if list_subdirs == []:
        return actor, critic
    else:
        if weights_to_load == "LAST":
            weights_dir = sorted(list_subdirs, key=lambda folder: folder.stat().st_mtime, reverse=True)[0]

            actor.load_state_dict(T.load(f"{weights_dir.path}/actor.pt", map_location=f"cuda:{gpu}"))
            critic.load_state_dict(T.load(f"{weights_dir.path}/critic.pt", map_location=f"cuda:{gpu}"))

            return actor, critic
        elif weights_to_load == "FIRST":
            weights_dir = sorted(list_subdirs, key=lambda folder: folder.stat().st_mtime, reverse=False)[1]

            actor.load_state_dict(T.load(f"{weights_dir.path}/actor.pt", map_location=f"cuda:{gpu}"))
            critic.load_state_dict(T.load(f"{weights_dir.path}/critic.pt", map_location=f"cuda:{gpu}"))

            return actor, critic
        else:
            if os.path.exists(f"{data_path}/backup/{weights_to_load}"):
                actor.load_state_dict(T.load(f"{data_path}/backup/{weights_to_load}/actor.pt", map_location=f"cuda:{gpu}"))
                critic.load_state_dict(T.load(f"{data_path}/backup/{weights_to_load}/critic.pt", map_location=f"cuda:{gpu}"))

                return actor, critic
            else:
                print("Weights not found; check your PATH. Using LAST weights instead")
                weights_dir = sorted(list_subdirs, key=lambda folder: folder.stat().st_mtime, reverse=True)[0]

                actor.load_state_dict(T.load(f"{weights_dir.path}/actor.pt", map_location=f"cuda:{gpu}"))
                critic.load_state_dict(T.load(f"{weights_dir.path}/critic.pt", map_location=f"cuda:{gpu}"))

                return actor, critic

def save_checkpoint(actor, critic, actor_optim, critic_optim, scaler, config:dict, last=False):
    checkpoint_dir_name = str(config["backup-logs"]["checkpoint_dir_name"])
    data_path = config["backup-logs"]["data_path"]

    local_time = time.localtime()
    checkpoint_dir_name = checkpoint_dir_name.replace("[d]", str(local_time.tm_mday)).replace("[mo]", str(local_time.tm_mon)).replace("[y]", str(local_time.tm_year))
    checkpoint_dir_name = checkpoint_dir_name.replace("[h]", str(local_time.tm_hour)).replace("[mi]", str(local_time.tm_min)).replace("[s]", str(local_time.tm_sec))
    
    if last:
        checkpoint_dir_name += "_LAST"

    os.makedirs(f"{data_path}/checkpoints/{checkpoint_dir_name}", exist_ok=True)

    model_ckpt = {
        "actor_model": actor.state_dict(),
        "critic_model": critic.state_dict(),
        "actor_optim": actor_optim.state_dict(),
        "critic_optim": critic_optim.state_dict(),
        "grad_scaler": scaler.state_dict(),
    }

    T.save(model_ckpt, f"{data_path}/checkpoints/{checkpoint_dir_name}/checkpoint.pt")

def load_checkpoint(actor:ActorNetwork, critic:CriticNetwork, actor_optim:optim.Adam, critic_optim:optim.Adam, scaler:T.amp.GradScaler, config:dict, gpu:int):
    data_path = config["backup-logs"]["data_path"]
    checkpoint_to_load = config["backup-logs"]["checkpoint_to_load"]

    list_subdirs = [file for file in os.scandir(f"{data_path}/checkpoints") if file.is_dir()]
    
    model_ckpt = None

    if list_subdirs == []:
        return actor, critic, actor_optim, critic_optim, scaler
    else:
        if checkpoint_to_load == "LAST":
            weights_dir = sorted(list_subdirs, key=lambda folder: folder.stat().st_mtime, reverse=True)[0]

            model_ckpt = T.load(f"{weights_dir.path}/checkpoint.pt", map_location=f"cuda:{gpu}")
        elif checkpoint_to_load == "FIRST":
            weights_dir = sorted(list_subdirs, key=lambda folder: folder.stat().st_mtime, reverse=False)[1]

            model_ckpt = T.load(f"{weights_dir.path}/checkpoint.pt", map_location=f"cuda:{gpu}")
        else:
            if os.path.exists(f"{data_path}/checkpoints/{checkpoint_to_load}"):
                model_ckpt = T.load(f"{data_path}/checkpoints/{checkpoint_to_load}/checkpoint.pt", map_location=f"cuda:{gpu}")
            else:
                print("Weights not found; check your PATH. Using LAST weights instead")
                weights_dir = sorted(list_subdirs, key=lambda folder: folder.stat().st_mtime, reverse=True)[0]

                model_ckpt = T.load(f"{weights_dir.path}/checkpoint.pt", map_location=f"cuda:{gpu}")

    actor.load_state_dict(model_ckpt["actor_model"])
    critic.load_state_dict(model_ckpt["critic_model"])
    actor_optim.load_state_dict(model_ckpt["actor_optim"])
    critic_optim.load_state_dict(model_ckpt["critic_optim"])
    scaler.load_state_dict(model_ckpt["grad_scaler"])

    return actor, critic, actor_optim, critic_optim, scaler

def save_score_history_csv(config:dict, score_history:list, rank:int, checkpoint=False):
    data_path = config["backup-logs"]["data_path"]
    score_history_csv_name = config["backup-logs"]["score_history_csv_name"]
    n_games = config["training-config"]["n_games"]

    local_time = time.localtime()
    score_history_csv_name = score_history_csv_name.replace("[d]", str(local_time.tm_mday)).replace("[mo]", str(local_time.tm_mon)).replace("[y]", str(local_time.tm_year))
    score_history_csv_name = score_history_csv_name.replace("[h]", str(local_time.tm_hour)).replace("[mi]", str(local_time.tm_min)).replace("[s]", str(local_time.tm_sec))
    
    csv_path = ""
    if checkpoint:
        csv_path = f"{data_path}/checkpoints/R{rank}_{score_history_csv_name}.csv"
    else:
        csv_path = f"{data_path}/logs/{score_history_csv_name}.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[f"RANK_{rank}"] = score_history[:n_games]

        with open(csv_path, "w", newline="") as f:
            df.to_csv(f, index=False)
    else:
        df = pd.DataFrame({f"RANK_{rank}": score_history[:n_games]})

        with open(csv_path, "w", newline="") as f:
            df.to_csv(f, index=False)

def remember(memory:PPOMemory, state, action, probs, vals, reward, done):
    memory.store_memory(state, action, probs, vals, reward, done)
            
def choose_action(observation, gpu, actor, critic):
    state = T.tensor([np.array(observation).tolist()]).cuda(gpu, non_blocking=True)

    dist = actor(state)
    value = critic(state)
    action = dist.sample()
    probs = dist.log_prob(action).sum(dim=-1) # sum log probs of all the buttons
    buttons_action = action.detach().cpu().numpy().flatten()

    return buttons_action, probs.item(), value.item()

def learn(gpu, actor, actor_optim, critic, critic_optim, memory:PPOMemory, config:dict, sync_loops:T.Tensor, scaler:T.amp.GradScaler, env):
    for i in range(config["training-config"]["n_epochs"]):
        distrib.barrier()
        env.unwrapped.send_logs(epoch=i+1, max_epoch=config["training-config"]["n_epochs"], learn_mode=True, done=False)

        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = memory.generate_batches()

        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + config["agent"]["gamma"]*values[k+1]*(1-int(dones_arr[k])) - values[k])
                discount *= config["agent"]["gamma"]*config["agent"]["gae"]
            advantage[t] = a_t
        advantage = T.tensor(advantage, dtype=T.float32).cuda(gpu, non_blocking=True)
        values = T.tensor(values).cuda(gpu, non_blocking=True)

        for batch in batches:
            with T.autocast(device_type="cuda", dtype=T.float16):  
                states = T.tensor(state_arr[batch], dtype=T.float32).cuda(gpu, non_blocking=True)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).cuda(gpu, non_blocking=True)
                actions = T.tensor(action_arr[batch], dtype=T.float32).cuda(gpu, non_blocking=True)
                
                dist = actor(states)
                critic_value = critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum(dim=1)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-config["agent"]["policy_clip"], 1+config["agent"]["policy_clip"])*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

            scaler.scale(total_loss).backward()

            scaler.unscale_(actor_optim)
            scaler.unscale_(critic_optim)

            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=config["agent"]["gradient_clip"])
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=config["agent"]["gradient_clip"])

            scaler.step(actor_optim)
            scaler.step(critic_optim)
            scaler.update()

            actor_optim.zero_grad()
            critic_optim.zero_grad()

    env.unwrapped.send_logs(epoch=-1, max_epoch=config["training-config"]["n_epochs"], learn_mode=True, done=False)
    memory.clear_memory()
    distrib.barrier()

def workers(gpu, rank_node, config:dict, sync_loops:T.Tensor, stop_ppo:T.Tensor, sync_steps:T.Tensor, sync_save_checkpoint:T.Tensor):
    rank = rank_node * config["hardware-config"]["n_gpus"] + gpu

    distrib.init_process_group(backend="gloo", init_method="env://", world_size=len(config["instances"]), rank=rank)
    T.cuda.set_device(gpu)
    env = FlattenObservation(gym.make('melee_env/MeleeEnv-v0', config=config, rank=rank, debug=True, disable_env_checker=True))

    memory = PPOMemory(config["training-config"]["batch_size"])

    actor = ActorNetwork(env.action_space.shape[0], env.observation_space.shape, config)
    actor.cuda(gpu)
    
    critic = CriticNetwork(env.observation_space.shape, config)
    critic.cuda(gpu)

    if config["backup-logs"]["load_weights_on_startup"] and not config["backup-logs"]["load_checkpoint_on_startup"]:
        actor, critic = load_weights(actor, critic, config, gpu)
        distrib.barrier()
        print(f"Loaded weights Rank:{rank}")

    actor_optim = optim.Adam(actor.parameters(), config["agent"]["actor_lr"])
    critic_optim = optim.Adam(critic.parameters(), config["agent"]["critic_lr"])
    scaler = T.amp.GradScaler("cuda")

    if config["backup-logs"]["load_checkpoint_on_startup"]:
        actor, critic, actor_optim, critic_optim, scaler = load_checkpoint(actor, critic, actor_optim, critic_optim, scaler, config, gpu)
        distrib.barrier()
        print(f"Loaded weights Rank:{rank}")
    
    if rank == 0:
        total_actor = sum(pa.numel() for pa in actor.parameters())
        total_critic = sum(pc.numel() for pc in critic.parameters())
        print(f"Total ACTOR parameters: {total_actor}")
        print(f"Total CRITIC parameters: {total_critic}")
    
    actor = DDP(actor, device_ids=[gpu])
    critic = DDP(critic, device_ids=[gpu])

    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    sync_steps[0][rank] = 0
    game = 0
    now_weights = time.time()
    now_checkpoint = time.time()

    while (not T.all(sync_loops >= config["training-config"]["n_games"]).item()) and stop_ppo.item() == 0:
        observation, info = env.reset()
        game += 1
        done = False
        score = 0

        while not done:
            action, prob, val = choose_action(observation, gpu, actor, critic)
            observation_, reward, term, trunc, info = env.step(action)
            done = term or trunc
            n_steps += 1
            sync_steps[0][rank] += 1
            score += np.array(reward).item()

            remember(memory, observation, action, prob, val, np.array(reward).item(), done)

            if (n_steps % config["training-config"]["learn_steps"] == 0):
                env.unwrapped.pause_game()
                while True:
                    if T.all(sync_steps % config["training-config"]["learn_steps"] == 0).item() and T.all(sync_steps != 0).item():
                        break
                
                learn(gpu, actor, actor_optim, critic, critic_optim, memory, config, sync_loops, scaler, env)

                if (now_weights + config["backup-logs"]["save_weights_every"]) <= time.time():
                    now_weights = time.time()
                    if rank == 0:
                        print("SAVING WEIGHTS !!!!")
                        save_weights(actor.module, critic.module, config)
                
                if (now_checkpoint + config["backup-logs"]["save_checkpoint_every"]) <= time.time():
                    now_checkpoint = time.time()
                    sync_save_checkpoint[0] = 0
                    if rank == 0:
                        print("SAVING CHECKPOINT !!!!")
                        save_checkpoint(actor.module, critic.module, actor_optim, critic_optim, scaler, config)
                        save_score_history_csv(config, score_history, rank, True)
                        sync_save_checkpoint[0] = 1
                    else:
                        save_score_history_csv(config, score_history, rank, True)
                        while True:
                            if sync_save_checkpoint[0].item() == 1:
                                break
                            
                env.unwrapped.resume_game()
                learn_iters += 1
                
            observation = observation_
            env.unwrapped.send_logs(game, score, avg_score, learn_iters, learn_mode=False, done=done)
        score_history.append(score)
        avg_score = np.mean(score_history)
        sync_loops[0][rank] = game
    

    if rank == 0:
        save_weights(actor.module, critic.module, config, True)
    save_score_history_csv(config, score_history, rank)
    
    env.close()
    distrib.destroy_process_group()

def setup_envs(config:dict, stop_ppo:T.Tensor):
    network_config = config["network-config"]
    n_gpus = config["hardware-config"]["n_gpus"]
    world_size = len(config["instances"])

    sync_loops = T.full((1, world_size), 0).share_memory_()
    sync_steps = T.full((1, world_size), 0).share_memory_()
    sync_save_checkpoint = T.Tensor([0]).share_memory_()
    nodes = world_size // n_gpus
    
    os.environ['MASTER_ADDR'] = network_config["DDP_ip"]              
    os.environ['MASTER_PORT'] = str(network_config["DDP_port"])

    for rank_node in range(nodes):
        p = mp.Process(target=lambda rank_node, n_gpus, config, sync_loops, stop_ppo, sync_steps, sync_save_checkpoint: 
                       (mp.spawn(workers, nprocs=n_gpus, args=(rank_node, config, sync_loops, stop_ppo, sync_steps, sync_save_checkpoint))), 
                       args=(rank_node, n_gpus, config, sync_loops, stop_ppo, sync_steps, sync_save_checkpoint))
        p.start()
   
if __name__ == "__main__":
    config = tomli.load(open(f"{pathlib.Path(__file__).parent.resolve()}/config.toml", "rb"))
    stop_ppo = T.tensor([0]).share_memory_()
    try:
        setup_envs(config, stop_ppo)
    except KeyboardInterrupt:
        stop_ppo[0] = 1