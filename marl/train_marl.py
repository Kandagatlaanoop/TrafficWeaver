# marl/train_marl.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
# Import the environment class directly
from marl_env import TrafficEnv

def train_hive_mind():
    """
    Initializes Ray and trains the multi-agent "hive mind".
    """
    print("--- Initializing Ray for Multi-Agent Training ---")
    ray.init(logging_level="ERROR")

    # Configure the PPO algorithm
    config = (
        PPOConfig()
        # Pass the environment class directly to the config
        .environment(TrafficEnv)
        .framework("torch")
        .multi_agent(
            # All agents will share a single policy
            policies={"shared_policy"},
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
    )

    print("--- Starting Hive Mind Training ---")
    
    trainer = config.build()
    
    # Run training for your desired number of iterations
    for i in range(300):
        result = trainer.train()
        mean_reward = result.get("episode_reward_mean", float('nan'))
        print(f"Iteration: {i+1}/700, Mean Reward: {mean_reward}")

    checkpoint_dir = trainer.save()
    print(f"\nTraining complete. Checkpoint saved in {checkpoint_dir}")

    ray.shutdown()

if __name__ == "__main__":
    train_hive_mind()