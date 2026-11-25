"""Fit and save PACT progress supervision from 20 successful demos."""

import argparse
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

from pact.progress import (
    ProgressModel, demonstration_targets, load_successful_demo_observations, progress_loss,
)
from rlif.data import split_demo_names
from rlpd.env import download_dataset


def progress_dataset(path, *, seed: int, terminal_steps: int):
    """Select successful demos deterministically and build aligned labels."""
    if terminal_steps < 1:
        raise ValueError("terminal_steps must be positive")
    # Use the same numeric demo ordering as the shared loader, then shuffle
    # at trajectory level before filtering for success.
    with h5py.File(path, "r") as file:
        count = len(file["data"])
    candidates = split_demo_names(path, count, 0, 0, seed)["train"]
    selected, examples = [], []
    for name, observations in load_successful_demo_observations(path, candidates):
        obs = torch.from_numpy(observations)
        labels, returns = demonstration_targets(len(obs))
        mask = torch.arange(len(obs)) >= len(obs) - terminal_steps
        examples.append((obs, labels, returns, mask))
        selected.append(name)
        if len(selected) == 20:
            break
    if len(selected) < 20:
        raise ValueError(f"expected 20 successful demonstrations, found {len(selected)}")
    return TensorDataset(*(torch.cat(field) for field in zip(*examples))), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lift/ph", "can/ph"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--terminal-steps", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.terminal_steps, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("epochs, terminal steps, batch size, and learning rate must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    torch.manual_seed(args.seed)
    dataset, demo_names = progress_dataset(
        download_dataset(args.dataset), seed=args.seed, terminal_steps=args.terminal_steps,
    )
    obs_dim = dataset.tensors[0].shape[1]
    model = ProgressModel(obs_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        running_loss = torch.zeros((), device=args.device)
        for obs, linear, returns, mask in loader:
            obs, linear, returns, mask = (x.to(args.device) for x in (obs, linear, returns, mask))
            loss = progress_loss(model(obs), linear, returns, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * len(obs)
        print(f"epoch {epoch + 1}/{args.epochs} loss={running_loss.item() / len(dataset):.6f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        torch.save({
            "model": model.cpu().state_dict(),
            "observation_dim": obs_dim,
            "demo_names": demo_names,
            "settings": {key: value for key, value in vars(args).items() if key != "output"},
        }, output)


if __name__ == "__main__":
    main()
