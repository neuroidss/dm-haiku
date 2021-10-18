# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lifting parameters in Haiku."""

import types
from typing import Any, Callable, Mapping, MutableMapping, Tuple, TypeVar
import warnings

from haiku._src import base
from haiku._src import data_structures
from haiku._src import module
from haiku._src import transform

T = TypeVar("T")
MutableBundle = MutableMapping[str, MutableMapping[str, Any]]

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Params = base.Params
hk.State = base.State
hk.Module = module.Module
hk.running_init = transform.running_init
hk.data_structures = data_structures
del module, data_structures, transform


def pack_into_dict(src: hk.Params,
                   dst: MutableMapping[str, Any],
                   prefix: str,
                   state: bool = False):
  """Puts items from src into dst, with an added prefix."""
  for key, value in src.items():
    new_key = f"{prefix}/{key}"
    assert new_key not in dst
    value = dict(value)
    if state:
      value = {k: base.StatePair(v, v) for k, v in value.items()}
    dst[new_key] = value


def unpack_from_dict(src: hk.Params, prefix: str) -> MutableBundle:
  """Returns pairs from src where key begins with prefix, cutting off prefix."""
  l = len(prefix)
  return {k[l:]: dict(v) for k, v in src.items() if k.startswith(prefix)}


def add_state_to_init_fn(
    init_fn: Callable[..., hk.Params],
) -> Callable[..., Tuple[hk.Params, hk.State]]:
  def wrapped_init_fn(*a, **k):
    params = init_fn(*a, **k)
    if not isinstance(params, Mapping):
      raise ValueError("For stateful lifted functions use `hk.lift_with_state`")
    return params, {}
  return wrapped_init_fn


# TODO(tycai): Make sure transformed functions have better names.
class LiftingModule(hk.Module):
  """See :func:`lift`."""

  def __init__(self, init_fn, name="lifted"):
    super().__init__(name=name)
    self._init_fn = init_fn

  def __call__(self, *args, **kwargs):
    frame = base.current_frame()
    outer_params = frame.params
    outer_state = frame.state
    if hk.running_init():
      inner_params, inner_state = self._init_fn(*args, **kwargs)
      # Lift parameters into this transform's params_dict.
      pack_into_dict(inner_params, outer_params, self.module_name)
      pack_into_dict(inner_state, outer_state, self.module_name, state=True)
      return inner_params, inner_state
    else:
      prefix = f"{self.module_name}/"
      inner_params = unpack_from_dict(outer_params, prefix)
      inner_state = unpack_from_dict(outer_state, prefix)
      inner_state = base.extract_state(inner_state, initial=False)
      inner_params = hk.data_structures.to_haiku_dict(inner_params)
      inner_state = hk.data_structures.to_haiku_dict(inner_state)
      return inner_params, inner_state


def lift(
    init_fn: Callable[..., hk.Params],
    name: str = "lifted",
) -> Callable[..., hk.Params]:
  r"""Lifts the given init fn to a function in the current Haiku namespace.

  During init, the returned callable will run the given ``init_fn``, and include
  the resulting params in the outer transform's dictionaries.
  During ``apply``, the returned callable will instead pull the relevant
  parameters from the outer transform's dictionaries.

  Must be called inside :func:`transform`\ , and be passed the ``init``
  member of a :class:`Transformed`\ .

  The user must ensure that the given ``init`` does not accidentally catch
  modules from an outer :func:`transform` via functional closure.

  Example:

    >>> def g(x):
    ...   return hk.Linear(1, name='g_linear')(x)
    >>> g = hk.transform(g)
    >>> init_rng = hk.next_rng_key() if hk.running_init() else None
    >>> x = jnp.ones([1, 1])
    >>> params = hk.lift(g.init, name='f_lift')(init_rng, x)
    >>> out = g.apply(params, None, x)

  Args:
    init_fn: The ``init`` function from an :class:`Transformed`\ .
    name: A string name to prefix parameters with.

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` reuses parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.
  """
  base.assert_context("lift")
  init_fn = add_state_to_init_fn(init_fn)
  lifted = LiftingModule(init_fn, name=name)
  return lambda *a, **k: lifted(*a, **k)[0]


class LiftWithStateUpdater:
  """Handles updating the state for a `lift_with_state` computation."""

  __slots__ = ("_called", "_name")

  def __init__(self, name: str):
    self._called = False
    self._name = name

  def __del__(self):
    if not self._called:
      warnings.warn("StateUpdater (from `lift_with_state`) must be used, call "
                    "`.update(..)` or `.ignore_update()` before it goes out of "
                    "scope.")

  def ignore_update(self):
    self._called = True

  def update(self, state: hk.State):
    """Updates Haiku's internal state to the given state."""
    if self._called:
      raise ValueError("State updater must only be used once.")

    if not base.inside_transform():
      raise ValueError(
          "State updater must be used inside hk.transform_with_state")

    self._called = True
    frame = base.current_frame()
    for mod_name, bundle in state.items():
      mod_name = f"{self._name}/{mod_name}"
      for name, value in bundle.items():
        initial_pair = base.StatePair(value, value)
        initial = frame.state[mod_name].get(name, initial_pair).initial
        frame.state[mod_name][name] = base.StatePair(initial, value)


def _to_callable(f: Callable[..., T]) -> Callable[..., T]:
  return lambda *a, **k: f(*a, **k)  # pylint: disable=unnecessary-lambda


def lift_with_state(
    init_fn: Callable[..., Tuple[hk.Params, hk.State]],
    name: str = "lifted",
) -> Tuple[Callable[..., Tuple[hk.Params, hk.State]], LiftWithStateUpdater]:
  r"""Lifts the given init fn to a function in the current Haiku namespace.

  During init, the returned callable will run the given ``init_fn``, and include
  the resulting params in the outer transform's dictionaries.
  During ``apply``, the returned callable will instead pull the relevant
  parameters from the outer transform's dictionaries.

  Must be called inside :func:`transform`\ , and be passed the ``init``
  member of a :class:`Transformed`\ .

  The user must ensure that the given ``init`` does not accidentally catch
  modules from an outer :func:`transform` via functional closure.

  Example:

    >>> def g(x):
    ...   return hk.Linear(1, name='g_linear')(x)
    >>> g = hk.transform_with_state(g)
    >>> init_rng = hk.next_rng_key() if hk.running_init() else None
    >>> x = jnp.ones([1, 1])
    >>> init_fn, updater = hk.lift_with_state(g.init, name='f_lift')
    >>> params, state = init_fn(init_rng, x)
    >>> out, state = g.apply(params, state, None, x)
    >>> updater.update(state)

  Args:
    init_fn: The ``init`` function from an :class:`Transformed`\ .
    name: A string name to prefix parameters with.

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` reuses parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.
  """
  base.assert_context("lift_with_state")
  init_fn = _to_callable(LiftingModule(init_fn, name=name))
  updater = LiftWithStateUpdater(name)
  return init_fn, updater
