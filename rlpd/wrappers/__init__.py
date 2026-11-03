import gymnasium as gym


def seed_env(env: gym.Env, seed: int):
    """Seeds the env's dynamics and its spaces' samplers; returns the first obs.

    Gymnasium seeds through reset(), so this replaces the old env.seed().
    """
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    return observation
