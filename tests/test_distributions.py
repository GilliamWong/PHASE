"""Checks the local tanh-Gaussian that replaced tensorflow_probability."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from rlpd.distributions.tanh_normal import TanhMultivariateNormalDiag


def test_log_prob_matches_change_of_variables_reference():
    # Catches a wrong Jacobian term or Gaussian density in the SAC entropy terms.
    loc = jnp.array([[0.3, -0.5, 0.1], [0.0, 0.2, -0.4]])
    scale = jnp.array([[0.5, 0.2, 0.8], [0.3, 0.6, 0.4]])
    dist = TanhMultivariateNormalDiag(loc, scale)

    actions, log_probs = dist.sample_and_log_prob(seed=jax.random.PRNGKey(0))

    a = np.asarray(actions, dtype=np.float64)
    pre_tanh = np.arctanh(a)
    expected = norm.logpdf(pre_tanh, np.asarray(loc), np.asarray(scale)).sum(-1)
    expected -= np.log1p(-(a**2)).sum(-1)
    np.testing.assert_allclose(np.asarray(log_probs), expected, rtol=1e-4, atol=1e-4)


def test_log_prob_stays_finite_for_saturated_actions():
    # tanh(10) rounds to 1.0 in float32; arctanh-based log_prob would be inf/nan.
    dist = TanhMultivariateNormalDiag(jnp.full((4, 2), 10.0), jnp.full((4, 2), 0.1))
    actions, log_probs = dist.sample_and_log_prob(seed=jax.random.PRNGKey(1))
    assert np.all(np.abs(np.asarray(actions)) <= 1.0)
    assert np.all(np.isfinite(np.asarray(log_probs)))


def test_mode_is_squashed_mean():
    loc = jnp.array([[-2.0, 0.0, 0.7]])
    dist = TanhMultivariateNormalDiag(loc, jnp.ones_like(loc))
    np.testing.assert_allclose(np.asarray(dist.mode()), np.tanh(np.asarray(loc)))
