import os
import argparse
import json
import torch
import numpy as np
from stable_baselines3 import PPO
from server.gym_env import ThermalGridGymEnv

def evaluate(task_id: str, model_path: str, steps: int = 20):
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        return

    print(f"Evaluating RL Model: {model_path} on Task: {task_id}")
    
    env = ThermalGridGymEnv(task_id=task_id)
    model = PPO.load(model_path)

    obs, info = env.reset()
    total_reward = 0.0
    
    print("-" * 50)
    print(f"{'Step':<5} | {'Reward':<8} | {'PUE':<6} | {'MaxCPU':<8} | {'Action Summary'}")
    print("-" * 50)

    for i in range(1, steps + 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Extract metadata from info
        summary = info.get("summary", "")
        # Parsed manually for display if needed or extract from obs vector
        # obs vector indices: [pue: 52, max_cpu: 20-29]
        max_cpu = np.max(obs[20:30])
        pue = obs[52]

        print(f"{i:<5} | {reward:<8.3f} | {pue:<6.3f} | {max_cpu:<8.1f} | {np.round(action, 2)}")
        total_reward += reward

        if done or truncated:
            break

    print("-" * 50)
    print(f"Evaluation Complete. Average Reward: {total_reward/i:.3f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Agent")
    parser.add_argument("--task", type=str, default="baseline", help="Task ID")
    parser.add_argument("--model", type=str, default="models/ppo_baseline_final", help="Path to model (without .zip)")
    parser.add_argument("--steps", type=int, default=20, help="Steps to run")

    args = parser.parse_args()
    evaluate(args.task, args.model, args.steps)
