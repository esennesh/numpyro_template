from contextlib import ExitStack
import functools
import jax
from jax import Array
import jax.numpy as jnp
import math
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.util import log_density
from pytrie import SortedStringTrie as Trie
from typing import Tuple

class AutoMeanFieldProposal(AutoGuide):
    def __init__(self, model, *, create_plates=None, prefix="auto"):
        self._event_dims = {}
        super().__init__(model, init_loc_fn=init_to_sample, prefix=prefix,
                         create_plates=create_plates)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = self._event_dims[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                batch_shape = site["value"].shape[:-event_dim]
                site_dist = site["fn"]
                while hasattr(site_dist, "base_dist"):
                    site_dist = site_dist.base_dist
                params = {}
                for param, constraint in site_dist.arg_constraints.items():
                    if isinstance(constraint, dist.constraints._Dependent):
                        constraint = dist.constraints.real
                    transform = biject_to(constraint)
                    init_value = transform.inv(getattr(site_dist, param))
                    if init_value.shape[:len(batch_shape)] != batch_shape:
                        init_value = jax.lax.broadcast(init_value, batch_shape)
                    params[param] = transform(numpyro.primitives.param(
                        "{}_{}_{}".format(name, self.prefix, param),
                        init_value=init_value
                    ))
                ind_event_dim = event_dim - site_dist.event_dim
                q = site_dist.__class__(**params).to_event(ind_event_dim)

                result[name] = numpyro.sample(name, q)

        return result

    @property
    def event_dims(self):
        return self._event_dims

    @property
    def num_particles(self):
        return self._num_particles

    def sample_posterior(self, rng_key, params, *args, sample_shape=(),
                         **kwargs):
        samples = {}
        with numpyro.infer.handlers.seed(rng_seed=rng_key):
            for site in self.prototype_trace:
                if site["type"] != "sample" or site["is_observed"]:
                    continue

                q = self._mixture(name, site["fn"])
                samples[site] = numpyro.sample(site, q.expand_by(sample_shape))

        return self._constrain(samples)

    def _setup_prototype(self, *args, **kwargs):
        from numpyro.handlers import block, trace
        with block(expose_types=["param"]):
            self.prototype_trace = trace(self.model).get_trace(*args, **kwargs)

        for name, site in self.prototype_trace.items():
            if site["type"] == "sample":
                if site["is_observed"]:
                    continue

                self._event_dims[name] = site["fn"].event_dim
                # If subsampling, repeat init_value to full size.
                for frame in site["cond_indep_stack"]:
                    if frame.name in self._prototype_frames:
                        assert frame == self._prototype_frames[frame.name], (
                            f"The plate {frame.name} has inconsistent dim or size. Please check your model again."
                        )
                    else:
                        self._prototype_frames[frame.name] = frame
            elif site["type"] == "plate":
                self._prototype_frame_full_sizes[name] = site["args"][0]
