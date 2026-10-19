"""
Mechanism test for the near-target ORBITING failure mode (E4 prerequisite).

Hypothesis under test (from dqn_2d_env.py's reward comment):
    "the turn action's fixed angular step size prevents settling exactly
     at zero heading error" -- i.e. orbiting is an ACTION-SPACE defect.

Method: replace the learned policy with an IDEAL GEOMETRIC CONTROLLER
that uses the exact same 3 discrete actions and the exact same env
physics (rotate-in-place 0.8 rad/s * DT, forward 1.5 m/s * DT):

    if |heading_error| > half of one angular step:  turn toward target
    else:                                           go forward

If this controller reaches ~100%, the action space is provably NOT the
bottleneck -- orbiting must be a learning/policy problem (wrong Q-values
near the target), and changing turn_rate (E4) would treat the wrong cause.
If even the ideal controller orbits, the action space is genuinely
defective and E4 (turn_rate ablation) is justified.

Two test conditions:
  A) standard episodes      -- full-range starts, obstacle-free
  B) near-target stress     -- drone spawned at radius d of the target
     with adversarial headings (the regime where orbiting was observed)

No training, no checkpoint, no network -- pure kinematics + geometry.
"""
import math

import torch

from dqn_config import DQNConfig
from dqn_2d_env import DroneNav2DEnv, DT, REACHED_THRESHOLD

cfg = DQNConfig(num_obstacles=0)
ANGULAR_STEP = cfg.turn_rate * DT          # rad per turn step  (~0.9 deg)
LINEAR_STEP = cfg.forward_speed * DT       # m per forward step (0.03 m)

device = "cpu"


def ideal_actions(env):
    """Bang-bang geometric controller from env ground-truth state."""
    rel = env.target - env.pos
    desired = torch.atan2(rel[:, 1], rel[:, 0])
    err = desired - env.yaw
    err = torch.atan2(torch.sin(err), torch.cos(err))          # wrap to [-pi, pi]
    action = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    action[err > ANGULAR_STEP / 2] = 1                          # turn left
    action[err < -ANGULAR_STEP / 2] = 2                         # turn right
    return action


def run_episodes(env, n_episodes, label, setup=None):
    """Roll out the ideal controller. setup(env, env_ids) is applied after
    every reset/resample so adversarial initial conditions persist across
    episode boundaries (finished envs get re-placed, not random-sampled)."""
    from dqn_action_space import discrete_action_to_continuous
    outcomes, steps_list = {"reached": 0, "timed_out": 0, "out_of_bounds": 0}, []
    done_count = 0
    obs_dict, _ = env.reset()
    if setup is not None:
        setup(env, torch.arange(env.num_envs))
    ep_step = torch.zeros(env.num_envs, dtype=torch.long)
    while done_count < n_episodes:
        action_idx = ideal_actions(env)
        cont = discrete_action_to_continuous(action_idx, device)
        obs_dict, reward, terminated, truncated, _ = env.step(cont)
        done = terminated | truncated
        ep_step += 1
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            for i in done_idx.tolist():
                if env._last_reached_flag[i].item():
                    outcomes["reached"] += 1
                elif env._last_out_of_bounds[i].item():
                    outcomes["out_of_bounds"] += 1
                else:
                    outcomes["timed_out"] += 1
                steps_list.append(int(ep_step[i].item()))
                done_count += 1
            ep_step[done_idx] = 0
            if setup is not None:
                setup(env, done_idx)   # keep finished envs adversarial
    n = sum(outcomes.values())
    print(f"[{label}] n={n}")
    for k, v in outcomes.items():
        print(f"    {k:<14}: {v:>4} ({v/n*100:5.1f}%)")
    if steps_list:
        mean_steps = sum(steps_list) / len(steps_list)
        print(f"    mean_episode_length={mean_steps:.1f} steps")
    return outcomes


print("=" * 60)
print("Turn-granularity mechanism test (ideal geometric controller)")
print(f"physics: turn {ANGULAR_STEP:.4f} rad/step ({math.degrees(ANGULAR_STEP):.2f} deg), "
      f"forward {LINEAR_STEP*1000:.1f} mm/step, reach radius {REACHED_THRESHOLD} m")
print("=" * 60)

# Condition A: standard obstacle-free episodes
env = DroneNav2DEnv(num_envs=64, cfg=cfg, device=device)
torch.manual_seed(0)
run_episodes(env, 500, "A) standard episodes, obstacles=0")

# Condition B: near-target stress -- spawn at radius d with worst-case headings
print()
for d in [0.8, 1.0, 1.5, 2.0]:
    env = DroneNav2DEnv(num_envs=64, cfg=cfg, device=device)
    torch.manual_seed(1)

    def make_setup(dist):
        def setup(env, env_ids):
            m = len(env_ids)
            angles = torch.rand(m) * 2 * math.pi - math.pi
            env.target[env_ids] = torch.tensor([0.0, 0.0])
            env.pos[env_ids, 0] = dist * torch.cos(angles)
            env.pos[env_ids, 1] = dist * torch.sin(angles)
            # facing AWAY from the target (outward) plus small jitter
            env.yaw[env_ids] = angles + (torch.rand(m) - 0.5) * 0.2
        return setup

    run_episodes(env, 128, f"B) near-target start d={d}m, facing away",
                 setup=make_setup(d))
