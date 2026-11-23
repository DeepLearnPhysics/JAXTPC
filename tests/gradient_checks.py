"""AD-vs-finite-difference checks of SimParams gradients (not a test module)."""

import jax
import jax.numpy as jnp
import numpy as np


def param_gradients(sim, deposits):
    # Grouping ids stay integer; only the physics columns are promoted
    to64 = lambda a: (a.astype(jnp.float64)
                      if hasattr(a, 'dtype') and jnp.issubdtype(a.dtype, jnp.floating)
                      else a)
    deposits = deposits._replace(volumes=jax.tree.map(to64, deposits.volumes))
    params = jax.tree.map(to64, sim.default_sim_params)

    @jax.jit
    def loss(p):
        return sum(jnp.sum(s ** 2) for s in sim.forward(p, deposits))
    return loss, jax.jit(jax.grad(loss))(params), params, float(loss(params))


def check_param_gradient(result, field, eps=1e-8, tol=1e-3, min_sensitivity=1e-4):
    """AD == central FD for one (possibly nested) SimParams field.

    The logarithmic sensitivity p/L dL/dp must exceed ``min_sensitivity``, so a
    parameter that has dropped out of the model (both gradients zero) fails. ``eps`` is
    relative and small so that no arrival time crosses a tick edge within the step.
    """
    loss, grads, params, value = result
    if field.startswith('recomb_params.'):
        name = field.split('.')[1]
        get = lambda t: getattr(t.recomb_params, name)
        put = lambda v: params._replace(
            recomb_params=params.recomb_params._replace(**{name: v}))
    else:
        get = lambda t: getattr(t, field)
        put = lambda v: params._replace(**{field: v})
    p0 = get(params)
    ad = float(get(grads))
    fd = float(loss(put(p0 * (1 + eps))) - loss(put(p0 * (1 - eps)))) / float(2 * eps * p0)
    sens = float(p0) * fd / value
    msg = f"{field}: AD={ad:.6e} FD={fd:.6e} dlnL/dlnp={sens:.3e}"
    assert np.isfinite(ad), msg
    assert abs(sens) > min_sensitivity, msg
    assert abs(ad - fd) < tol * abs(fd), msg
