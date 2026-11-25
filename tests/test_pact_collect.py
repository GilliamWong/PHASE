import numpy as np
import torch

from pact.collect import collect_episode
from rlif.human import OfficialRandomController


def test_collect_episode_records_task_rewards_and_fresh_takeover_sample():
    class Actor(torch.nn.Linear):
        calls = 0

        def sample(self, observation):
            self.calls += 1
            return observation.new_full((1, 1), self.calls / 10), observation.new_zeros(1)

    class Env:
        def reset(self):
            self.step_index = 0
            return np.array([0.8], dtype=np.float32), {}

        def step(self, action):
            self.step_index += 1
            done = self.step_index == 3
            return np.array([0.2], dtype=np.float32), float(done), done, False, {"is_success": done}

    class Progress(torch.nn.Linear):
        def forward(self, observation):
            return observation[:, 0]

    class Replay:
        def add_episode(self, batch, annotations):
            self.batch = batch
            self.annotations = annotations

    actor = Actor(1, 1)
    agent = type("Agent", (), {"actor": actor})()
    gate = OfficialRandomController(lambda obs: np.array([-0.5]), np.random.default_rng(42), 0.5, 4)
    replay = Replay()
    metrics = collect_episode(Env(), agent, gate, Progress(1, 1), replay)
    assert metrics["intervention_events"] == 1
    assert metrics["intervention_steps"] == 2
    torch.testing.assert_close(replay.batch.rewards, torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(replay.batch.masks, torch.tensor([1., 1., 0.]))
    assert replay.annotations.preference_indices.tolist() == [0, -1, -1]
    np.testing.assert_allclose(replay.annotations.preferences[0].policy_action, [0.3])
    np.testing.assert_allclose(replay.annotations.preferences[0].human_action, [-0.5])
