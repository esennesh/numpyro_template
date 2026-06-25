from abc import ABC
from collections import defaultdict
from functools import cached_property
import functools
import math
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer.elbo import MultiFrameTensor
from numpyro.infer.util import compute_log_probs, get_importance_trace
from numpyro._typing import Message
from numpyro.util import _validate_model, check_model_guide_match

from typing import Callable, Dict, Optional

def broadcast_observed(observed, target_shape):
    while observed.ndim < len(target_shape):
        observed = jnp.expand_dims(observed, axis=-1)
    return jnp.broadcast_to(observed, target_shape)

def configure_sample(msg: Message, /, **kwargs) -> Dict:
    return kwargs

def expected_value(site: Dict):
    fn = site.get("fn", None)
    mean = getattr(fn, "mean", None)
    if mean is None:
        return site["value"]
    return mean() if callable(mean) else mean

def logmeanexp(value, axis=None, keepdims=False):
    if axis is None:
        size = value.size
    elif isinstance(axis, tuple):
        size = math.prod(value.shape[idx] for idx in axis)
    else:
        size = value.shape[axis]
    return jax.scipy.special.logsumexp(value, axis=axis, keepdims=keepdims) -\
        jnp.log(size)

def trace_entry(site: Dict, log_p, log_q, observed):
    return {
        "value": site["value"],
        "log_p": log_p,
        "log_q": log_q,
        "observed": observed,
        "ev": expected_value(site),
    }

class VariationalMixin(ABC):
    def log_weights(self, traces, mutables):
        raise NotImplementedError

    def loss_fn(self, log_ws, traces):
        raise NotImplementedError

class ELBOMixin(VariationalMixin):
    def log_weights(self, traces, mutables):
        log_ws = 0.
        beta = getattr(self, "beta", 1.)
        for name, site in traces.items():
            term = site["log_p"] - site["log_q"]
            log_ws = log_ws + jnp.where(site["observed"], term, beta * term)
        return log_ws

    def loss_fn(self, log_ws, traces):
        return -jnp.mean(log_ws, axis=0).sum()

class IwaeMixin(ELBOMixin):
    def loss_fn(self, log_ws, traces):
        return -jax.nn.logmeanexp(log_ws)

class ParticleTracer(ELBOMixin):
    def __init__(self, beta: float=1., num_particles: int=1):
        self.beta = beta
        self.num_particles = num_particles

    def __call__(self, rng_key, param_map, particle_params, model, guide,
                 *args, **kwargs):
        param_map = param_map.copy()
        particle_params = self._broadcast_particle_params(particle_params)
        rng_key, guide = self._prepare_guide(rng_key, param_map, guide, args,
                                             kwargs)

        def single_trace(rng_key, pwise_params, param_map=param_map,
                         particle=None):
            return self._single_trace(rng_key, pwise_params, param_map,
                                      particle, model, guide, args, kwargs)

        rng_keys = random.split(rng_key, self.num_particles)
        particles = jnp.arange(self.num_particles)
        particle_traces = jax.vmap(single_trace)
        return particle_traces(rng_keys, particle_params, particle=particles)

    def _broadcast_particle_params(self, particle_params):
        return jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (self.num_particles,
                                                 *leaf.shape))
            if leaf.shape[0] != self.num_particles else leaf,
            particle_params
        )

    def _build_graph_state(self, model_trace, guide_trace):
        graph_state = {
            name: trace_entry(
                site,
                site["log_prob"],
                guide_trace[name]["log_prob"] if name in guide_trace
                else jnp.zeros_like(site["log_prob"]),
                site["is_observed"],
            )
            for name, site in model_trace.items()
            if site["type"] == "sample"
        }
        graph_state.update({
            name: trace_entry(
                site,
                jnp.zeros_like(site["log_prob"]),
                site["log_prob"],
                False,
            )
            for name, site in guide_trace.items()
            if site["type"] == "sample" and name not in graph_state
        })
        graph_state.update({
            name: trace_entry(site, 0., 0., False)
            for name, site in model_trace.items()
            if site["type"] == "deterministic"
        })
        graph_state.update({
            name: trace_entry(site, 0., 0., False)
            for name, site in guide_trace.items()
            if site["type"] == "deterministic"
        })
        return graph_state

    def _prepare_guide(self, rng_key, param_map, guide, args, kwargs):
        return rng_key, guide

    def _single_trace(self, rng_key, pwise_params, param_map, particle, model,
                      guide, args, kwargs):
        param_map = param_map | pwise_params
        particle_guide, particle_model = guide, model

        model_seed, guide_seed = random.split(rng_key)
        if particle is not None:
            particle_guide = numpyro.handlers.infer_config(
                particle_guide,
                functools.partial(configure_sample, k=particle)
            )
            particle_model = numpyro.handlers.infer_config(
                particle_model,
                functools.partial(configure_sample, k=particle)
            )
        seeded_model = numpyro.handlers.seed(particle_model, model_seed)
        seeded_guide = numpyro.handlers.seed(particle_guide, guide_seed)
        model_trace, guide_trace = get_importance_trace(seeded_model,
                                                        seeded_guide, args,
                                                        kwargs, param_map)

        check_model_guide_match(model_trace, guide_trace)
        _validate_model(model_trace, plate_warning="loose")

        graph_state = self._build_graph_state(model_trace, guide_trace)
        mutables = {name: site["value"] for name, site in model_trace.items()
                    if site["type"] == "mutable"}
        return graph_state, mutables

    def loss(self, *args, **kwargs):
        traces, mutables = self(*args, **kwargs)
        for k, v in traces.items():
            is_observed = broadcast_observed(v["observed"],
                                             v["log_p"].shape)
            traces[k] = v | {"observed": is_observed}
        log_ws = self.log_weights(traces, mutables)
        return self.loss_fn(log_ws, traces), {"log_w": log_ws.sum(axis=-1),
                                              "mutables": mutables,
                                              "trace": traces}

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        pass

class ELBOTracer(ParticleTracer):
    def __init__(self, num_particles: int=1):
        super().__init__(num_particles=num_particles)
        self._guide_deps, self._model_deps = None, None
        self._guide_properties, self._model_properties = {}, {}

    def log_weights(self, traces, mutables):
        if jax.tree.leaves(mutables):
            return super().log_weights(traces, mutables)
        log_ws = jnp.array(0.0)
        # mapping from non-reparameterizable sample sites to cost terms
        # influenced by each of them
        downstream_costs: Dict[str, MultiFrameTensor] =\
            defaultdict(lambda: MultiFrameTensor())
        for name, site in traces.items():
            log_ws = log_ws + site["log_p"]
            for key in self._model_deps.get(name, []):
                downstream_costs[key].add((
                    self._model_properties[name]["cond_indep_stack"],
                    site["log_p"]
                ))
            if name in self._guide_properties:
                log_q = site["log_q"]
                if not self._guide_properties[name]["reparameterized"]:
                    log_q = jax.lax.stop_gradient(log_q)
                log_ws = log_ws - log_q
                for key in self._guide_deps[name]:
                    downstream_costs[key].add((
                        self._guide_properties[name]["cond_indep_stack"],
                        -site["log_q"]
                    ))

        for node, cost in downstream_costs.items():
            downstream_cost = cost.sum_to(
                self._guide_properties[node]["cond_indep_stack"]
            )
            advantage = downstream_cost - downstream_cost.mean(axis=0)
            surrogate = traces[node]["log_q"] * jax.lax.stop_gradient(
                advantage
            )
            log_ws = log_ws + surrogate - jax.lax.stop_gradient(surrogate)
        return log_ws

    def loss_fn(self, log_ws, traces):
        reparameterized = all(site["reparameterized"] for site
                              in self._guide_properties.values())
        if reparameterized:
            return super().loss_fn(log_ws, traces)
        return -(jnp.sum(log_ws, axis=0) / (log_ws.shape[0] - 1)).sum()

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        self._guide_deps, self._model_deps = guide_deps, model_deps
        for name, site in guide_trace.items():
            if site["type"] != "sample":
                continue

            self._guide_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
                "reparameterized": site["fn"].has_rsample
            }

        for name, site in model_trace.items():
            if site["type"] != "sample":
                continue

            self._model_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
            }

class OvisTracer(ParticleTracer):
    def __init__(self, beta=1., include_aux=True, num_particles: int=1,
                 num_auxiliary: Optional[int]=None):
        self._guide_deps, self._model_deps = None, None
        self._guide_properties, self._model_properties = {}, {}
        self._include_aux = include_aux
        if not num_auxiliary:
            num_auxiliary = num_particles
        self._num_aux = num_auxiliary
        super().__init__(beta=beta, num_particles=num_particles + num_auxiliary)

    @cached_property
    def control_variate(self):
        def fn(log_ws, log_aux):
            # log_ws: K x B
            # log_aux: S x B
            B, K, S = log_ws.shape[-1], log_ws.shape[0], log_aux.shape[0]

            log_ws = jnp.expand_dims(log_ws, (0, 1)) # -> 1 x 1 X K x B
            log_ws = jnp.broadcast_to(log_ws, (S, K, K, B))
            log_aux = jnp.expand_dims(log_aux, (1, 2)) # -> S x 1 x 1 x B
            log_aux = jnp.broadcast_to(log_aux, (S, K, K, B))

            mask = jnp.expand_dims(jnp.identity(K), (0, -1)) # -> 1 x K x K x 1
            log_w_hat = (1 - mask) * log_ws + mask * log_aux # S x K x K x B
            # S x K x K x B -> S x B x K x K
            objectives = jnp.moveaxis(self.objective(log_w_hat, axis=-2), -1, 1)
            # S x B x K x K -> S x B x K -> B x K
            results = jnp.diagonal(objectives, axis1=-2, axis2=-1).mean(axis=0)
            # B x K -> K x B
            return jnp.moveaxis(results, 0, -1)
        return fn

    def loss_fn(self, log_ws, traces):
        num_particles = self.num_particles - self._num_aux
        log_weights, log_aux = log_ws[:num_particles], log_ws[num_particles:]

        rewards = self.objective(log_weights, axis=0)
        values = self.control_variate(log_weights, log_aux)
        advantages = rewards - values

        if self._include_aux:
            log_evidence = jax.nn.logmeanexp(log_ws, axis=0)
        else:
            log_evidence = jax.nn.logmeanexp(log_weights, axis=0)

        surrogates = jnp.zeros_like(log_weights)
        for name, site in traces.items():
            if name in self._guide_properties and\
               not self._guide_properties[name]["reparameterized"]:
                log_q = site["log_q"][:num_particles]
                surrogate = log_q * jax.lax.stop_gradient(advantages)
                surrogates = surrogates + surrogate
        surrogates = surrogates.sum(axis=0)
        loss = -(log_evidence + surrogates - jax.lax.stop_gradient(surrogates))
        return loss.sum()

    @cached_property
    def objective(self):
        def fn(log_ws, axis=0):
            return jax.nn.logmeanexp(log_ws, axis=axis, keepdims=True) -\
                   jax.nn.softmax(log_ws, axis=axis)
        return fn

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        self._guide_deps, self._model_deps = guide_deps, model_deps
        for name, site in guide_trace.items():
            if site["type"] != "sample":
                continue

            self._guide_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
                "reparameterized": site["fn"].has_rsample
            }

        for name, site in model_trace.items():
            if site["type"] != "sample":
                continue

            self._model_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
            }

class VarGradMixin(VariationalMixin):
    def log_weights(self, traces, mutables):
        return sum(jnp.sum(site["log_p"], axis=-1) -
                   jnp.sum(site["log_q"], axis=-1)
                   for name, site in traces.items())

    def loss_fn(self, log_ws, traces):
        return jnp.var(-log_ws, axis=0, ddof=1.).sum() / 2

class VarGradTracer(VarGradMixin, ParticleTracer):
    pass

class OnlineWeightMixin(VariationalMixin):
    def log_weights(self, traces, mutables):
        log_likelihood = sum(jnp.where(site["observed"], site["log_p"],
                                       jnp.zeros_like(site["observed"]))
                             for name, site in traces.items())
        log_q = sum(site["log_q"] for site in traces.values())
        return log_likelihood - log_q

class OnlineVarGradTracer(OnlineWeightMixin, VarGradTracer):
    pass

class AdaptiveParticleTracer(ParticleTracer):
    def _prepare_guide(self, rng_key, param_map, guide, args, kwargs):
        if hasattr(guide, "adapt") and isinstance(guide.adapt, Callable):
            adaptation_rng, rng_key = random.split(rng_key)

            adapt = numpyro.handlers.seed(guide.adapt, adaptation_rng)
            with numpyro.handlers.substitute(data=param_map):
                adapted_params = adapt(*args, **kwargs)
            guide = functools.partial(guide, adaptation=adapted_params)
        return rng_key, guide

class AdaptiveElboTracer(AdaptiveParticleTracer, ELBOTracer):
    pass
