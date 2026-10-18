"""
Batch evaluation with FAILURE-MODE CLASSIFICATION for the 2D standalone env.

Extends dqn_diagnose_2d.py's outcome table (reached / out_of_bounds /
timed_out) by splitting the timed_out bucket into the two failure modes
the team is working on (see handoff doc, section 2.1 conclusion 6):

    stuck    : timed out, ended far from the target, and barely moved
               over the last --stuck_window_s seconds (typically parked
               in front of an obstacle, "don't know turn left or right")
    orbiting : timed out, but ended within --orbit_radius of the target
               (close enough to see it, never settling inside the 0.75 m
               reach threshold)
    timed_out: everything else that timed out (wandering, slow detours)

crashed (obstacle contact) is also reported separately -- the env exposes
it via _last_any_contact, the old script lumped it into timed_out.

Classification uses per-step position history recorded during the greedy
rollout; thresholds are CLI-tunable. THIS SCRIPT IS EVALUATION-ONLY: it
loads the checkpoint read-only and never touches training code.

--seed (default 0) makes the scenario stream deterministic: two runs with
the same seed evaluate on the IDENTICAL set of episodes, which is what
makes living_cost A/B comparisons (handoff doc P1-6) cleanly attributable.
Use --seed=-1 to fall back to non-deterministic sampling.

Example:
    python3 dqn_diagnose_2d_classify.py \
        --checkpoint=checkpoints/curriculum_3obs_ep1800.pt \
        --num_obstacles=3 --num_envs=64 --num_episodes=100 \
        --csv_out=eval_1800_classified.csv
"""
import argparse
import csv

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import DroneNav2DEnv, REACHED_THRESHOLD

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--csv_out", type=str, default=None)
parser.add_argument("--seed", type=int, default=0,
                    help="scenario-stream seed; -1 = non-deterministic")
parser.add_argument("--orbit_radius", type=float, default=2.0,
                    help="m; timed-out episode ending within this distance "
                         "of the target is 'orbiting'")
parser.add_argument("--stuck_window_s", type=float, default=4.0,
                    help="seconds of trailing history used for the "
                         "no-movement test")
parser.add_argument("--stuck_disp", type=float, default=0.3,
                    help="m; displacement below this over the trailing "
                         "window (and not orbiting) is 'stuck'")
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

if args.seed >= 0:
    torch.manual_seed(args.seed)

device = "cpu"
env = DroneNav2DEnv(num_envs=args.num_envs, cfg=cfg, device=device)
q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions,
                 hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()
print(f"[CLASSIFY] loaded checkpoint: {args.checkpoint}")
print(f"[CLASSIFY] seed={args.seed}  orbit_radius={args.orbit_radius}m  "
      f"stuck_window={args.stuck_window_s}s  stuck_disp={args.stuck_disp}m")

n = args.num_envs
max_steps = env.max_steps
stuck_window_steps = int(args.stuck_window_s / 0.02)  # DT = 0.02 s

obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]

cum_reward = torch.zeros(n)
ep_step = torch.zeros(n, dtype=torch.long)
# Per-env ring buffer of this episode's positions, indexed by ep_step.
pos_buf = torch.zeros(n, max_steps, 2)

episodes = []

while len(episodes) < args.num_episodes:
    with torch.no_grad():
        action_idx = q_net(state).argmax(dim=-1)
    actions_continuous = discrete_action_to_continuous(action_idx, device)
    obs_dict, reward, terminated, truncated, _ = env.step(actions_continuous)
    next_state = obs_dict["dqn_state"]
    done = terminated | truncated
    cum_reward += reward
    ep_step += 1

    # _pre_reset[0] holds post-integration positions BEFORE finished envs
    # were resampled -- the true trajectory point for every env this step.
    pos_buf[torch.arange(n), (ep_step - 1).clamp(max=max_steps - 1)] = env._pre_reset[0]

    if done.any():
        done_idx = done.nonzero(as_tuple=True)[0].tolist()
        final_pos, _, final_target, final_obs = env._pre_reset
        for i in done_idx:
            steps = int(ep_step[i].item())
            hist = pos_buf[i, :steps]                       # (steps, 2)
            tgt = final_target[i]
            dist_hist = torch.norm(hist - tgt, dim=-1)
            final_dist = float(dist_hist[-1])
            min_dist = float(dist_hist.min())
            w = min(stuck_window_steps, steps - 1) if steps > 1 else 0
            recent_disp = float(torch.norm(hist[-1] - hist[-1 - w])) if w > 0 else 0.0
            if env.num_obstacles > 0:
                obs_d = torch.norm(final_obs[i] - final_pos[i], dim=-1)
                nearest_obs_surf = float(obs_d.min() - cfg.obstacle_radius)
            else:
                nearest_obs_surf = float("nan")

            # Priority order matches the env's own tie-breaking:
            # reached > crashed > out_of_bounds > timeout sub-classes.
            if env._last_reached_flag[i].item():
                outcome = "reached"
            elif env._last_any_contact[i].item():
                outcome = "crashed"
            elif env._last_out_of_bounds[i].item():
                outcome = "out_of_bounds"
            elif final_dist < args.orbit_radius:
                outcome = "orbiting"
            elif recent_disp < args.stuck_disp:
                outcome = "stuck"
            else:
                outcome = "timed_out"

            episodes.append({
                "episode_id": len(episodes), "outcome": outcome,
                "steps": steps, "total_reward": round(cum_reward[i].item(), 3),
                "final_dist": round(final_dist, 3), "min_dist": round(min_dist, 3),
                "recent_disp": round(recent_disp, 3),
                "nearest_obs_surf": round(nearest_obs_surf, 3),
            })
        cum_reward[done_idx] = 0
        ep_step[done_idx] = 0

    state = next_state

print(f"\n[CLASSIFY] collected {len(episodes)} episodes\n")
outcome_counts = {}
for e in episodes:
    outcome_counts[e["outcome"]] = outcome_counts.get(e["outcome"], 0) + 1
total = len(episodes)
print(f"[CLASSIFY] Outcome distribution (n={total}):")
for k in ["reached", "crashed", "out_of_bounds", "stuck", "orbiting", "timed_out"]:
    v = outcome_counts.get(k, 0)
    print(f"  {k:<14}: {v:>4} ({v/total*100:>5.1f}%)")

mean_reward = sum(e["total_reward"] for e in episodes) / total
mean_steps = sum(e["steps"] for e in episodes) / total
print(f"\n[CLASSIFY] mean_reward={mean_reward:.2f}  mean_episode_length={mean_steps:.1f}")

if args.csv_out:
    fields = ["episode_id", "outcome", "steps", "total_reward",
              "final_dist", "min_dist", "recent_disp", "nearest_obs_surf"]
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    print(f"[CLASSIFY] CSV written to {args.csv_out}")
