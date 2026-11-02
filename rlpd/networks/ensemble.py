from typing import Optional, Type

import flax.linen as nn
import jax
import jax.numpy as jnp


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


def subsample_ensemble(
    key: jax.Array, params, num_sample: Optional[int], num_qs: int
):
    """Picks num_sample random ensemble members (REDQ); None keeps all."""
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)
        params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params
