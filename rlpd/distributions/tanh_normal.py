import functools
import math
from typing import Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from rlpd.networks import default_init

# These replace the tensorflow_probability distributions used upstream; TFP's
# JAX substrate no longer imports with current JAX releases.

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class MultivariateNormalDiag:
    """Gaussian with diagonal covariance over the last axis."""

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray):
        self.loc = loc
        self.scale_diag = scale_diag

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        noise = jax.random.normal(seed, self.loc.shape, self.loc.dtype)
        return self.loc + self.scale_diag * noise

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        z = (x - self.loc) / self.scale_diag
        return jnp.sum(-0.5 * z**2 - jnp.log(self.scale_diag) - _HALF_LOG_2PI, axis=-1)

    def sample_and_log_prob(self, seed: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.sample(seed)
        return x, self.log_prob(x)

    def mode(self) -> jnp.ndarray:
        return self.loc


def _tanh_log_det_jacobian(x: jnp.ndarray) -> jnp.ndarray:
    # log(1 - tanh(x)^2), written in a form that stays finite for large |x|.
    return 2.0 * (math.log(2.0) - x - jax.nn.softplus(-2.0 * x))


class TanhMultivariateNormalDiag:
    """A diagonal Gaussian whose samples are squashed into (-1, 1) by tanh.

    There is deliberately no log_prob(action): recovering the pre-tanh value
    with arctanh is inaccurate for saturated actions, so log-probabilities are
    computed together with the sample from the pre-tanh value. (Upstream got
    the same effect from TFP's bijector cache.)
    """

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray):
        self.base = MultivariateNormalDiag(loc, scale_diag)

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        return jnp.tanh(self.base.sample(seed))

    def sample_and_log_prob(self, seed: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.base.sample(seed)
        log_prob = self.base.log_prob(x) - jnp.sum(_tanh_log_det_jacobian(x), axis=-1)
        return jnp.tanh(x), log_prob

    def mode(self) -> jnp.ndarray:
        return jnp.tanh(self.base.mode())


class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(
        self, inputs, *args, **kwargs
    ) -> Union[MultivariateNormalDiag, TanhMultivariateNormalDiag]:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        if self.squash_tanh:
            return TanhMultivariateNormalDiag(means, jnp.exp(log_stds))
        else:
            return MultivariateNormalDiag(means, jnp.exp(log_stds))


TanhNormal = functools.partial(Normal, squash_tanh=True)
