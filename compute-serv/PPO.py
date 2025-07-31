import numpy as np
import torch.nn as nn
import torch as T
import torch.optim as optim
import torch.distributed as distrib
import torch.multiprocessing as mp
import gymnasium as gym
from torch.distributions import Beta
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import tomli
import pathlib
import os

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
        
        activation_name = config["actor_model"]["activation"]
        hidden_layers = config["actor_model"]["hidden_layers"]
        hidden_input_size = hidden_layers[0]
        
        layers = []
        layers.append(nn.Linear(*input_dims, hidden_input_size))
        layers.append(activation_fcts[activation_name]())
        
        for layer_size in hidden_layers:
            layers.append(nn.Linear(hidden_input_size, layer_size))
            layers.append(activation_fcts[activation_name]())
            hidden_input_size = layer_size

        self.shared = nn.Sequential(*layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_input_size, n_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_input_size, n_actions), nn.Softplus())

    def forward(self, state):
        x = self.shared(state)
        alpha = self.alpha_head(x) + 1e-5 
        beta = self.beta_head(x) + 1e-5

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
        
        activation_name = config["critic_model"]["activation"]
        hidden_layers = config["critic_model"]["hidden_layers"]
        hidden_input_size = hidden_layers[0]
        
        layers = []
        layers.append(nn.Linear(*input_dims, hidden_input_size))
        layers.append(activation_fcts[activation_name]())
        
        for layer_size in hidden_layers:
            layers.append(nn.Linear(hidden_input_size, layer_size))
            layers.append(activation_fcts[activation_name]())
            hidden_input_size = layer_size
        
        layers.append(nn.Linear(hidden_input_size, 1))
        
        self.critic = nn.Sequential(*layers)

    def forward(self, state):
        value = self.critic(state)

        return value


def remember(memory, state, action, probs, vals, reward, done):
    memory.store_memory(state, action, probs, vals, reward, done)
            
def choose_action(observation, gpu, actor, critic):
    state = T.tensor([np.array(observation).tolist()], dtype=T.float).cuda(gpu, non_blocking=True)

    dist = actor(state)
    value = critic(state)
    action = dist.sample()
    probs = dist.log_prob(action).sum(dim=-1) # sum log probs of all the buttons
    buttons_action = action.detach().cpu().numpy().flatten()

    return buttons_action, probs.item(), value.item()

def learn(gpu, actor, actor_optim, critic, critic_optim, memory:PPOMemory, config, rank, progress_bar, sync_loops, scaler):
    for i in range(config["training-config"]["n_epochs"]):
        if rank == 0:
            progress_bar.update(1)
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

            if T.all(sync_loops >= config["training-config"]["n_games"]).item():
                break

            scaler.scale(total_loss).backward()
            scaler.step(actor_optim)
            scaler.step(critic_optim)
            scaler.update()
            actor_optim.zero_grad()
            critic_optim.zero_grad()

    memory.clear_memory()

def workers(gpu, rank_node, config, sync_loops):
    global progress_bar
    progress_bar = "ola"
    rank = rank_node * config["hardware-config"]["n_gpus"] + gpu
    if rank == 0:
        progress_bar = tqdm(total=config["training-config"]["n_epochs"])

    distrib.init_process_group(backend="gloo", init_method="env://", world_size=config["training-config"]["world_size"], rank=rank)
    T.cuda.set_device(gpu)
    env = gym.make("CartPole-v1")

    memory = PPOMemory(config["training-config"]["batch_size"])

    actor = ActorNetwork(env.action_space.n, env.observation_space.shape)
    actor.cuda(gpu)
    actor_optim = optim.Adam(actor.parameters(), config["agent"]["actor_lr"])
    
    critic = CriticNetwork(env.observation_space.shape)
    critic.cuda(gpu)
    critic_optim = optim.Adam(critic.parameters(), config["agent"]["critic_lr"])

    actor = DDP(actor, device_ids=[gpu])
    critic = DDP(critic, device_ids=[gpu])
    scaler = T.amp.GradScaler("cuda")

    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    game = 0

    while (not T.all(sync_loops >= config["training-config"]["n_games"]).item()):
        observation, info = env.reset()
        game += 1
        done = False
        score = 0

        while not done:
            action, prob, val = choose_action(observation, gpu, actor, critic)
            observation_, reward, term, trunc, info = env.step(action)
            done = term or trunc
            n_steps += 1
            score += reward

            remember(memory, observation, action, prob, val, reward, done)

            if (n_steps % config["training-config"]["learn_steps"] == 0):
                if rank == 0:
                    progress_bar.reset()
                learn(gpu, actor, actor_optim, critic, critic_optim, memory, config, rank, progress_bar, sync_loops, scaler)
                learn_iters += 1
                
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        sync_loops[0][rank] = game
        

        if rank == 0:
            print('episode', game, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

    if rank == 0:
        T.save(actor, "./actor.pt")
        T.save(critic, "./critic.pt")
    distrib.destroy_process_group()

def train(config):
    network_config = config["network-config"]
    n_gpus = config["hardware-config"]["n_gpus"]
    os.environ['MASTER_ADDR'] = network_config["DDP_ip"]              
    os.environ['MASTER_PORT'] = str(network_config["DDP_port"])
    sync_loops = T.full((1, config["training-config"]["world_size"]), 0).share_memory_()

    nodes = config["training-config"]["world_size"] // n_gpus
    for rank_node in range(nodes):
        p = mp.Process(target=lambda rank_node, n_gpus, config, sync_loops: (mp.spawn(workers, nprocs=n_gpus, args=(rank_node, config, sync_loops))), args=(rank_node, n_gpus, config, sync_loops))
        p.start()
   
if __name__ == "__main__":
    config = tomli.load(open(f"{pathlib.Path(__file__).parent.resolve()}/config.toml", "rb"))
    train(config)