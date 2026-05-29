import numpy as np
from env import make_causal_env

def sample_trajectories(episodes=1000, confound_strength=0.8, filepath='train_trajectories.npz'):
    env = make_causal_env(confound_strength=confound_strength)
    all_obs, all_actions, all_rewards, all_goals, all_colors = [], [], [], [], []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() 
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            all_obs.append(obs['image']) 
            all_actions.append(action)
            all_rewards.append(reward)
            all_goals.append(step_info['goal_direction'])
            all_colors.append(step_info['floor_color'])
            
            obs = next_obs
            done = terminated or truncated

    env.close()
    np.savez_compressed(
        filepath,
        obs=np.array(all_obs, dtype=np.uint8), actions=np.array(all_actions, dtype=np.int32),
        rewards=np.array(all_rewards, dtype=np.float32), goals=np.array(all_goals, dtype=str),
        colors=np.array(all_colors, dtype=str)
    )
    print(f"Saved {len(all_actions)} steps across {episodes} episodes to {filepath}")

if __name__ == "__main__":
    sample_trajectories(episodes=1000, confound_strength=0.8, filepath='train_data.npz')
    sample_trajectories(episodes=200, confound_strength=0.0, filepath='test_data.npz')
