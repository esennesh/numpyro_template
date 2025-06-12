from functools import partial
from jax import Array, jit
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import SVI, Trace_ELBO
from numpyro import optim
from typing import Any, Dict

from trainer.para import ParaMonad

class SviPara(ParaMonad):
    def __init__(self, data_shape, guide, lr, model, num_particles, rng):
        if not isinstance(rng, Array):
            rng = random.key(rng)
        self.optimizer = optim.Adam(step_size=lr)
        self.num_particles = num_particles
        self.svi = SVI(model, guide, self.optimizer,
                       Trace_ELBO(num_particles))
        self.svi_state = self.svi.init(rng, jnp.zeros((1,) + data_shape))

    def load(self, checkpoint: Dict[str, Any]):
        self.svi_state = checkpoint["svi_state"]

    @property
    def parameters(self):
        return self.optimizer.get_params(self.svi_state.optim_state)

    def save(self):
        return {"svi_state": self.svi_state}

    @staticmethod
    @partial(jit, static_argnums=0)
    def svi_evaluate(svi, state, data):
        return svi.evaluate(state, data)

    @staticmethod
    @partial(jit, static_argnums=0)
    def svi_update(svi, state, data):
        return svi.update(state, data)

    def test_step(self, data, *args):
        return {"loss": self.svi_evaluate(self.svi, self.svi_state, data)}

    def train_step(self, data, *args):
        self.svi_state, loss = self.svi_update(self.svi, self.svi_state, data)
        return {"loss": loss}

    def valid_step(self, data, *args) -> Dict[str, float]:
        return {"loss": self.svi_evaluate(self.svi, self.svi_state, data)}
