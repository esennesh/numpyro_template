from functools import cached_property
import itertools
import jax
import jax.random as random
import networkx as nx
import numpyro

from numpyro.infer.elbo import get_nonreparam_deps
from numpyro.infer import Predictive
from omegaconf.dictconfig import DictConfig
import optax

from typing import Any, Dict

from .learner import ParamLearner
from src.data import DataModule
from src.inference.tracer import ParticleTracer
from src.utils import initialize_traces, is_autoguide, reconstruct

class GraphicalModelLearner(ParamLearner):
    def __init__(self, data_shape, guide, model, optim, rng,
                 tracer: ParticleTracer,
                 scheduler: optax.GradientTransformation=None):
        if is_autoguide(guide):
            guide = guide(model)
        if not isinstance(rng, jax.Array):
            rng = random.key(rng)
        self._buffer_state = {}
        self._constrain_fn = None
        self._graph = nx.DiGraph()
        self._guide = guide
        self._model = model
        self.optim_state = None
        if isinstance(optim, numpyro.optim._NumPyroOptim):
            self.optimizer = optim
        else:
            if isinstance(optim, dict) or isinstance(optim, DictConfig):
                optim = optax.chain(*optim.values())
            self.optimizer = numpyro.optim.optax_to_numpyro(optim)
        self._particle_params = set({})
        self._relations = {}
        self._rng = rng
        self.scheduler = scheduler
        self.schedule_state = None
        self._tracer = tracer

    @property
    def buffer_state(self):
        return self._buffer_state

    def __call__(self, *args, stage="train", **kwargs):
        self._rng, rng = random.split(self.rng)
        particle_params = jax.lax.stop_gradient(self.buffer_state)
        particle_params.update({
            param: value for param, value in self.parameters.items()
            if param in self._particle_params
        })
        params = {param: value for param, value in self.parameters.items()
                  if param not in self._particle_params}
        trace, mutables = self.tracer(rng, params, particle_params,
                                      reconstruct(self.model),
                                      self.guide, *args, **kwargs)
        return {k: v[0] for k, v in trace.items()}

    @cached_property
    def _evaluate(self):
        @jax.jit
        def fn(data, params, rng):
            next_rng, rng = random.split(rng)
            particle_params = jax.lax.stop_gradient(self.buffer_state)
            particle_params.update({
                param: value for param, value in params.items()
                if param in self._particle_params
            })
            params = {param: value for param, value in params.items()
                      if param not in self._particle_params}
            loss, state = self.tracer.loss(rng, params, particle_params,
                                           self.model, self.guide, data)
            return loss, next_rng, state
        return fn

    @property
    def guide(self):
        return self._guide

    def load(self, checkpoint: Dict[str, Any]):
        self._buffer_state = checkpoint["buffer_state"]
        self.optim_state = checkpoint["optim_state"]

    @property
    def tracer(self):
        return self._tracer

    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return self._constrain_fn(self.optimizer.get_params(self.optim_state))

    def render_model(self, filename=None, render_distributions=False,
                     render_params=False):
        from numpyro.infer.inspect import (generate_graph_specification,
                                           render_graph)
        graph_spec = generate_graph_specification(self._relations,
                                                  render_params=render_params)
        graph = render_graph(graph_spec,
                             render_distributions=render_distributions)

        if filename is not None:
            filename = Path(filename)
            # remove leading period from suffix
            filename_without_suffix = filename.with_suffix("")
            graph.render(
                filename_without_suffix,
                view=False,
                cleanup=True,
                format=filename.suffix[1:],
            )

        return graph

    @property
    def rng(self):
        return self._rng

    def save(self):
        return {"buffer_state": self.buffer_state,
                "optim_state": self.optim_state}

    def _setup_graph(self, *args, **kwargs):
        state = initialize_traces(self.model, self.guide, self._rng, {}, *args,
                                  **kwargs)
        self._constrain_fn, self._buffer_state, self._rng =\
            state.constrain_fn, state.mutables, state.rng
        guide_trace, model_trace = state.guide_trace, state.model_trace

        latents = {}
        for name, site in guide_trace.items():
            if site["type"] == "sample" and not site.get("is_observed", False):
                latents[name] = site["value"]

        from numpyro.handlers import replay, seed
        self._rng, model_seed, guide_seed = random.split(self._rng, 3)
        init_guide = replay(seed(self.guide, guide_seed), guide_trace)
        init_model = replay(seed(self.model, model_seed), model_trace)
        model_deps, guide_deps = get_nonreparam_deps(init_model, init_guide,
                                                     args, kwargs, state.params,
                                                     latents=latents)
        self.tracer.setup(guide_deps, model_deps, guide_trace, model_trace)

        from src.utils import get_model_relations
        self._relations = get_model_relations(init_model, args, kwargs)
        for var, parents in self._relations["sample_sample"].items():
            self._graph.add_node(var)
            for par in parents:
                self._graph.add_edge(par, var)
        return state

    def setup_step(self, datamodule: DataModule):
        for batch in datamodule.test_dataloader():
            data = batch[0]
            break

        state = self._setup_graph(data)
        buffers = state.mutables
        params = {}
        for param, value in state.params.items():
            site = state.guide_trace.get(param, None)
            if not site:
                site = state.model_trace[param]
            if site["kwargs"].get("particle", False):
                self._particle_params.add(param)
            if site["kwargs"].get("requires_grad", True):
                params[param] = value
            else:
                buffers[param] = value

        if not self.optim_state:
            self._buffer_state = buffers
            self.optim_state = self.optimizer.init(params)

        if self.scheduler and not self.schedule_state:
            self.schedule_state = self.scheduler.init(params)

        return state.guide_trace, state.model_trace

    @cached_property
    def _update(self):
        @jax.jit
        def fn(data, optim_state, rng):
            next_rng, rng = random.split(rng)
            def loss_fn(params):
                particle_params = jax.lax.stop_gradient(self.buffer_state)
                particle_params.update({
                    param: value for param, value in params.items()
                    if param in self._particle_params
                })
                params = {param: value for param, value in params.items()
                          if param not in self._particle_params}
                return self.tracer.loss(rng, params, particle_params,
                                        self.model, self.guide, data)

            # Replicating the Numpyro eval_and_update() method.
            (loss, state), grads = numpyro.optim._value_and_grad(
                loss_fn, x=self.optimizer.get_params(optim_state)
            )
            # Intervene by scaling the grads according to LR scheduler
            if self.scheduler and self.schedule_state:
                grads = optax.tree.scale(self.schedule_state.scale, grads)
            # Actually update
            optim_state = self.optimizer.update(grads, optim_state, value=loss)
            return loss, optim_state, next_rng, state
        return fn

    def test_step(self, data, *args, **kwargs):
        loss, self._rng, state = self._evaluate(data, self.parameters, self.rng)
        self._buffer_state.update(state["mutables"])
        return {"loss": loss, "log_w": state["log_w"]}

    def train_step(self, data, *args, **kwargs):
        loss, self.optim_state, self._rng, state = self._update(
            data, self.optim_state, self.rng
        )
        self._buffer_state.update(state["mutables"])
        return {"loss": loss, "log_w": state["log_w"]}

    def validate(self, loss: float):
        if self.scheduler:
            _, self.schedule_state = self.scheduler.update(
                updates=self.parameters, state=self.schedule_state, value=loss
            )

    def valid_step(self, data, *args, **kwargs):
        loss, self._rng, state = self._evaluate(data, self.parameters, self.rng)
        self._buffer_state.update(state["mutables"])
        return {"loss": loss, "log_w": state["log_w"]}
