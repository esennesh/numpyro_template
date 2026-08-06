from jax import jit, lax
from jax.example_libraries import stax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

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
    import math

    batch_dim, image_shape = batch.shape[0], batch.shape[1:]
    out_dim = math.prod(image_shape)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim),
                            (batch_dim, z_dim))
    with numpyro.plate("batch", batch_dim):
        z = numpyro.sample("z", dist.Normal(0, 1).expand([z_dim]).to_event(1))
        img_loc = decode(z).reshape((batch_dim,) + image_shape)
        return numpyro.sample("obs", dist.Bernoulli(img_loc).to_event(3),
                              obs=batch)

def mnist_guide(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim),
                            (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    with numpyro.plate("batch", batch_dim):
        return numpyro.sample("z", dist.Normal(z_loc, z_std).to_event(1))
