import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.world_object import Floor, Goal
from minigrid.envs.empty import EmptyEnv
from minigrid.wrappers import RGBImgObsWrapper

class CausalMiniGridEnv(EmptyEnv):
    def __init__(self, confound_strength=0.8, size=8, **kwargs):
        self.confound_strength = confound_strength
        
        self.goal_positions = {
            'north': (3, 1), 'south': (3, 6),
            'east': (6, 3), 'west': (1, 3)
        }
        self.causal_mapping = {
            'north': 'red', 'south': 'blue',
            'east': 'yellow', 'west': 'purple'
        }
        
        self.current_goal = None
        self.current_color = None
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.current_goal = self.np_random.choice(list(self.goal_positions.keys()))

        # THE FIX: Exact probability distribution
        all_colors = ['red', 'blue', 'yellow', 'purple']
        correct_color = self.causal_mapping[self.current_goal]
        
        if self.confound_strength == 0.0:
            probs = [0.25, 0.25, 0.25, 0.25]
        else:
            probs = [self.confound_strength if c == correct_color else (1.0 - self.confound_strength)/3 for c in all_colors]
            
        self.current_color = self.np_random.choice(all_colors, p=probs)

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                self.grid.set(i, j, Floor(self.current_color))

        self.put_obj(Goal(), *self.goal_positions[self.current_goal])
        self.agent_pos = (width // 2, height // 2)
        self.agent_dir = self.np_random.integers(0, 4)
        self.mission = "reach the goal"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['goal_direction'] = self.current_goal
        info['floor_color'] = self.current_color
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['goal_direction'] = self.current_goal
        info['floor_color'] = self.current_color
        return obs, info

def make_causal_env(confound_strength=0.8):
    env = CausalMiniGridEnv(confound_strength=confound_strength, render_mode="rgb_array")
    env = RGBImgObsWrapper(env, tile_size=8)
    return env
