from jax import jit, lax, random
from jax.example_libraries import stax
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

def affine_grid(thetas, size):
    N, C, H, W = size

    xs, ys = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    xs, ys = 2 * xs / W - 0.5, 2 * ys / H - 0.5
    xs, ys = xs[:, jnp.newaxis], ys[:, jnp.newaxis]
    ones = jnp.ones_like(xs)
    grids = jnp.concatenate((xs, ys, ones), axis=-1)
    grids = jnp.expand_dims(jnp.reshape(grids, (-1, 3, 1)), 0)
    grids = jnp.tile(grids, (N, 1, 1, 1)).squeeze(axis=-1)

    transforms = jnp.reshape(thetas, (-1, 2, 3))
    grids = jnp.matmul(transforms, jnp.matrix_transpose(grids))
    # transform grid range from [-1,1) to the range of [0,1)
    grids = (jnp.matrix_transpose(grids) + 1) / 2
    return jnp.multiply(grids, [W, H])

def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )

def decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()),
        stax.Sigmoid,
    )

def mnist_model(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim), (batch_dim, z_dim))
    with numpyro.plate("batch", batch_dim):
        z = numpyro.sample("z", dist.Normal(0, 1).expand([z_dim]).to_event(1))
        img_loc = decode(z)
        return numpyro.sample("obs", dist.Bernoulli(img_loc).to_event(1), obs=batch)

def mnist_guide(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    with numpyro.plate("batch", batch_dim):
        return numpyro.sample("z", dist.Normal(z_loc, z_std).to_event(1))

@jit
def binarize(rng_key, batch):
    return random.bernoulli(rng_key, batch).astype(batch.dtype)
