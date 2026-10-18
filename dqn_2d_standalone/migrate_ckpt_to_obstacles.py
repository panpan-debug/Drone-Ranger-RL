"""
Migrate a pure-navigation DQN checkpoint (num_obstacles=0, state_dim=2)
to a wider network for obstacle training (num_obstacles=N, state_dim=2+3*K).

Why zero-padding works: dqn_2d_env.py pads unused/invisible obstacle slots
with a constant sentinel (sense_range, 0, 0) / state_scale. Setting the new
fc1 columns to zero means the obstacle inputs contribute exactly nothing at
initialization, so the migrated network's policy is IDENTICAL to the
navigation policy on obstacle-free states. Gradients still flow into the
new columns (dL/dW = delta * input), so obstacle avoidance is learned
on top of the intact navigation behavior instead of from scratch.

Usage (from the dqn_2d_standalone directory):
    python3 migrate_ckpt_to_obstacles.py \
        --in=checkpoints/nav100_final.pt \
        --out=checkpoints/nav100_3obs_init.pt \
        --num_obstacles=3

Then start curriculum training with:
    python3 train_dqn.py --num_envs=64 --num_episodes=3000 \
        --num_obstacles=3 --replay_buffer_size=5000000 \
        --resume_from=checkpoints/nav100_3obs_init.pt \
        --run_name=curriculum_3obs
"""
import argparse

import torch

from dqn_config import DQNConfig

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="src", required=True, help="source checkpoint (num_obstacles=0)")
parser.add_argument("--out", dest="dst", required=True, help="output checkpoint path")
parser.add_argument("--num_obstacles", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="must match the checkpoint's hidden_dim (default 128)")
args = parser.parse_args()

cfg_old = DQNConfig(num_obstacles=0, hidden_dim=args.hidden_dim)
cfg_new = DQNConfig(num_obstacles=args.num_obstacles, hidden_dim=args.hidden_dim)
print(f"state_dim: {cfg_old.state_dim} -> {cfg_new.state_dim} "
      f"(k_nearest={cfg_new.k_nearest}, {3 * min(cfg_new.k_nearest, args.num_obstacles)} new obstacle columns)")

sd_old = torch.load(args.src, map_location="cpu")
fc1_w_old = sd_old["fc1.weight"]  # (hidden, old_state_dim)
assert fc1_w_old.shape == (args.hidden_dim, cfg_old.state_dim), (
    f"fc1.weight shape {tuple(fc1_w_old.shape)} != ({args.hidden_dim}, {cfg_old.state_dim}); "
    "check --hidden_dim matches the training run")

sd_new = {k: v.clone() for k, v in sd_old.items()}
fc1_w_new = torch.zeros(args.hidden_dim, cfg_new.state_dim)
fc1_w_new[:, :cfg_old.state_dim] = fc1_w_old   # keep dx/dy columns, zero the obstacle columns
sd_new["fc1.weight"] = fc1_w_new

torch.save(sd_new, args.dst)

# Sanity check: policy output must be identical on obstacle-free states
from dqn_action_space import QNetwork
net_old = QNetwork(state_dim=cfg_old.state_dim, action_dim=cfg_old.num_actions, hidden_dim=args.hidden_dim)
net_new = QNetwork(state_dim=cfg_new.state_dim, action_dim=cfg_new.num_actions, hidden_dim=args.hidden_dim)
net_old.load_state_dict(sd_old)
net_new.load_state_dict(sd_new)
net_old.eval(); net_new.eval()

with torch.no_grad():
    s2 = torch.randn(256, cfg_old.state_dim)
    pad = torch.tensor([cfg_new.sense_range / cfg_new.state_scale, 0.0, 0.0])
    s11 = torch.cat([s2, pad.repeat(256, (cfg_new.state_dim - 2) // 3)], dim=1)
    same = torch.equal(net_old(s2).argmax(-1), net_new(s11).argmax(-1))
print("policy preserved on obstacle-free states:", "OK" if same else "MISMATCH (should not happen)")
print(f"saved: {args.dst}")
print("NOTE: optimizer state is NOT migrated (Adam moments have the old shapes); "
      "train_dqn.py will start Adam fresh, and epsilon restarts at resume_epsilon=0.3.")
