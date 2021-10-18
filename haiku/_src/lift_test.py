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

import warnings

from absl.testing import absltest
from haiku._src import base
from haiku._src import lift
from haiku._src import module
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


class Bias(module.Module):

  def __call__(self, x):
    b = base.get_parameter("b", (), init=jnp.ones)
    return x + b


class LiftTest(absltest.TestCase):

  def test_lift_with_vmap(self):
    def inner_fn(x):
      assert x.ndim == 1
      return Bias()(x)

    def outer_fn(x):
      assert x.ndim == 2
      x = Bias()(x)
      inner = transform.without_apply_rng(transform.transform(inner_fn))
      inner_p = lift.lift(inner.init)(base.next_rng_key(), x[0])
      vmap_inner = jax.vmap(inner.apply, in_axes=(None, 0))
      return vmap_inner(inner_p, x)

    key = jax.random.PRNGKey(428)
    init_key, apply_key = jax.random.split(key)
    data = np.zeros((3, 2))

    outer = transform.transform(outer_fn)
    outer_params = outer.init(init_key, data)
    self.assertEqual(outer_params, {
        "bias": {"b": np.ones(())},
        "lifted/bias": {"b": np.ones(())},
    })

    out = outer.apply(outer_params, apply_key, data)
    np.testing.assert_equal(out, 2 * np.ones((3, 2)))

  @test_utils.transform_and_run
  def test_empty_lift(self):
    f = transform.transform(lambda: None)
    self.assertEmpty(lift.lift(f.init)(None))

  @test_utils.transform_and_run
  def test_empty_lift_with_state(self):
    f = transform.transform_with_state(lambda: None)
    init_fn, updater = lift.lift_with_state(f.init)
    params, state = init_fn(None)
    self.assertEmpty(params)
    self.assertEmpty(state)
    updater.ignore_update()

  @test_utils.transform_and_run
  def test_unused_updater(self):
    def f() -> lift.LiftWithStateUpdater:
      f = transform.transform_with_state(lambda: None)
      return lift.lift_with_state(f.init)[1]

    with self.assertWarnsRegex(Warning, "StateUpdater.*must be used"):
      f()

    with warnings.catch_warnings(record=True) as caught_warnings:
      f().update({})
      f().ignore_update()
    self.assertEmpty(caught_warnings)

  @test_utils.transform_and_run(run_apply=False)
  def test_lift_raises_with_state(self):
    f = transform.transform_with_state(
        lambda: base.get_state("w", [], init=jnp.zeros))
    lifted = lift.lift(f.init)  # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(ValueError, "use.*lift_with_state"):
      lifted(None)

  def test_lift_with_state(self):
    def inner():
      w = base.get_state("w", [], init=jnp.zeros)
      w += 1
      base.set_state("w", w)
      return w

    inner = transform.transform_with_state(inner)

    def outer():
      lifted, updater = lift.lift_with_state(inner.init)
      params, state = lifted(None)
      self.assertEmpty(params)
      # NOTE: Value is always initial.
      self.assertEqual(state, {"~": {"w": 0}})
      out, state = inner.apply(params, state, None)
      updater.update(state)
      return out, state

    outer = transform.transform_with_state(outer)
    params, state = outer.init(None)
    self.assertEmpty(params)
    self.assertEqual(jax.tree_map(int, state), {"lifted/~": {"w": 0}})

    (w, inner_state), state = outer.apply(params, state, None)
    del inner_state
    self.assertEqual(w, 1)
    self.assertEmpty(params)
    self.assertEqual(state, {"lifted/~": {"w": 1}})

if __name__ == "__main__":
  absltest.main()
