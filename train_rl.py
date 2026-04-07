import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from server.gym_env import ThermalGridGymEnv

def train(task_id: str, steps: int):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"Starting RL Training for Task: {task_id}")
    print(f"Training for {steps} steps...")

    # Align max_episode_steps and n_steps (1024 is power of 2, good for PPO)
    n_steps = 1024
    env = ThermalGridGymEnv(task_id=task_id, max_steps=n_steps)
    
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment check passed!")
    
    # Initialize PPO model
    # MLP Policy (Multi-Layer Perceptron) is standard for vector observations
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/ppo_thermal_grid_tensorboard/",
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        gamma=0.99,
    )

    # Save checkpoints every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix=f"ppo_{task_id}"
    )

    # Train
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    model_path = f"models/ppo_{task_id}_final"
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO RL Agent for Thermal Grid Environment")
    parser.add_argument("--task", type=str, default="baseline", help="Task ID (baseline, load_shift, grid_stress)")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    
    args = parser.parse_args()
    train(args.task, args.steps)
