"""Collect a PACT episode using PHASE's frozen simulated corrector."""

import numpy as np
import torch

from pact.replay import annotate_episode
from rlif.data import Batch
from rlif.human import OfficialRandomController
from rlif.train import load_agent


def random_gate(checkpoint, obs_dim, action_dim, device, rng,
                *, probability=0.5, takeover_steps=4):
    """Use the same deterministic corrector and random gate as RLIF."""
    corrector = load_agent(checkpoint, obs_dim, action_dim, device)
    return OfficialRandomController(
        lambda obs: corrector.act(obs, deterministic=True),
        rng, probability, takeover_steps,
    )


@torch.no_grad()
def collect_episode(env, agent, gate, progress_model, replay):
    """Collect task rewards and write annotated transitions after the rollout."""
    observation, _ = env.reset()
    gate.reset()
    device = next(agent.actor.parameters()).device
    rows, interventions, alternatives = [], [], {}
    while True:
        state = torch.as_tensor(observation, dtype=torch.float32, device=device)[None]
        policy_action, _ = agent.actor.sample(state)
        policy_action = policy_action[0].cpu().numpy()
        correction = gate.act(observation, policy_action)
        intervening = correction is not None
        if intervening and (not interventions or not interventions[-1]):
            alternative, _ = agent.actor.sample(state)
            alternatives[len(rows)] = alternative[0].cpu().numpy()
        action = np.array(correction if intervening else policy_action, copy=True)
        previous = np.array(observation, copy=True)
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append((previous, action, reward, float(not terminated), np.array(observation, copy=True)))
        interventions.append(intervening)
        if terminated or truncated:
            break

    batch = Batch(*(torch.as_tensor(np.asarray(field), dtype=torch.float32)
                    for field in zip(*rows)))
    success = bool(info.get("is_success", False))
    progress = np.zeros(len(rows), dtype=np.float32)
    if success and alternatives:
        progress_device = next(progress_model.parameters()).device
        progress = progress_model(batch.observations.to(progress_device)).cpu().numpy()
    annotations = annotate_episode(
        batch.observations.numpy(), batch.actions.numpy(), interventions,
        alternatives, progress, success=success,
    )
    replay.add_episode(batch, annotations)
    return {
        "steps": len(rows),
        "success": success,
        "intervention_steps": sum(interventions),
        "intervention_events": len(alternatives),
        "return": float(batch.rewards.sum()),
    }
