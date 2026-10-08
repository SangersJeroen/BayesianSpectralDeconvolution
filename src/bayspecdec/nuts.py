"""
nuts.py — Multinomial No-U-Turn transition for one chain, in pure JAX.

Iterative (non-recursive) formulation after Hoffman & Gelman (2014) and Betancourt (2017),
with the subtree U-turn checks done through momentum checkpoints as in NumPyro/Stan, so the
whole transition is a pair of nested ``lax.while_loop`` s and can be ``vmap`` -ed over
temperatures. Under ``vmap`` the loops run until the slowest chain finishes.

* uniform multinomial sampling inside a subtree, biased progressive sampling across doublings
* generalised U-turn criterion on the diagonal metric: ``v . rho <= 0`` at either end, where
  ``v = M^-1 p`` and ``rho`` is the summed momentum of the (sub)trajectory
* divergence when the Hamiltonian error exceeds ``max_energy_error`` (or is not finite)

The slice variable of the numpy sampler is replaced by multinomial weights; target density and
metric are the same.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp

Array = jax.Array

# ``vg(z, beta) -> ((U, (ll, lp)), dU/dz)``: value-and-grad of the potential with the likelihood and
# prior as auxiliary outputs.
ValueAndGrad = Callable[[Array, Array], tuple[tuple[Array, tuple[Array, Array]], Array]]
# Phase-space edge ``(z, r, grad U)`` and a proposal ``(z, ll, lp)``.
Edge = tuple[Array, Array, Array]
Proposal = tuple[Array, Array, Array]


class Subtree(TypedDict):
    last: Edge
    prop: Proposal
    logw: Array
    r_sum: Array
    n: Array
    acc: Array
    turning: Array
    diverging: Array


class Tree(TypedDict):
    left: Edge
    right: Edge
    prop: Proposal
    logw: Array
    r_sum: Array
    n: Array
    acc: Array
    turning: Array
    diverging: Array
    depth: Array


def _leaf_idx_to_ckpt_idxs(n: Array) -> tuple[Array, Array]:
    """Checkpoint slots ``[idx_min, idx_max]`` to test after adding 0-based leaf ``n``."""
    idx_max = jax.lax.population_count(n >> 1)
    num_subtrees = jax.lax.population_count((~n & (n + 1)) - 1)
    return idx_max - num_subtrees + 1, idx_max


def nuts_transition(
    vg: ValueAndGrad,
    key: Array,
    z: Array,
    ll: Array,
    lp: Array,
    beta: Array,
    eps: Array,
    inv_m: Array,
    max_depth: int,
    max_energy_error: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """
    One NUTS transition. ``vg(z, beta) -> ((U, (ll, lp)), dU/dz)`` is the value-and-grad of
    the potential. Returns ``(z, ll, lp, mean_accept_prob, moved)``.
    """
    d = z.shape[0]
    i32 = jnp.int32

    def kinetic(r: Array) -> Array:
        return 0.5 * jnp.sum(inv_m * r**2)

    def turning(r_a: Array, r_b: Array, rho: Array) -> Array:
        return (jnp.dot(inv_m * r_a, rho) <= 0.0) | (jnp.dot(inv_m * r_b, rho) <= 0.0)

    key, k_mom = jax.random.split(key)
    r0 = jax.random.normal(k_mom, z.shape) / jnp.sqrt(inv_m)
    (U0, _), g0 = vg(z, beta)
    H0 = U0 + kinetic(r0)

    def new_leaf(
        zc: Array, rc: Array, gc: Array, going_right: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        e = jnp.where(going_right, eps, -eps)
        r = rc - 0.5 * e * gc
        zn = zc + e * inv_m * r
        (U, (ll_n, lp_n)), g = vg(zn, beta)
        r = r - 0.5 * e * g
        dH = U + kinetic(r) - H0
        dH = jnp.where(jnp.isnan(dH), jnp.inf, dH)
        diverging = dH > max_energy_error
        logw = jnp.where(diverging, -jnp.inf, -dH)
        acc = jnp.exp(jnp.minimum(0.0, -dH))
        return zn, r, g, ll_n, lp_n, logw, acc, diverging

    # -- build a subtree of 2**depth leaves, starting next to the edge (z_e, r_e, g_e) -----

    def build_subtree(
        depth: Array, edge: Edge, going_right: Array, key: Array
    ) -> Subtree:
        z_e, r_e, g_e = edge
        n_max = jnp.left_shift(i32(1), depth.astype(i32))

        sub0 = Subtree(
            last=edge,
            prop=(z_e, ll, lp),
            logw=jnp.asarray(-jnp.inf),
            r_sum=jnp.zeros(d),
            n=i32(0),
            acc=jnp.asarray(0.0),
            turning=jnp.asarray(False),
            diverging=jnp.asarray(False),
        )
        state0 = (sub0, key, jnp.zeros((max_depth, d)), jnp.zeros((max_depth, d)))

        State = tuple[Subtree, Array, Array, Array]

        def cond(state: State) -> Array:
            sub = state[0]
            return (sub["n"] < n_max) & ~sub["turning"] & ~sub["diverging"]

        def body(state: State) -> State:
            sub, key, r_ck, rs_ck = state
            key, k_pick = jax.random.split(key)
            zn, rn, gn, ll_n, lp_n, logw, acc, div = new_leaf(*sub["last"], going_right)

            logw_tot = jnp.logaddexp(sub["logw"], logw)
            p = jnp.where(jnp.isfinite(logw_tot), jnp.exp(logw - logw_tot), 0.0)
            take = jax.random.uniform(k_pick) < p
            prop = jax.tree_util.tree_map(
                lambda new, old: jnp.where(take, new, old), (zn, ll_n, lp_n), sub["prop"]
            )
            r_sum = sub["r_sum"] + rn

            idx_min, idx_max = _leaf_idx_to_ckpt_idxs(sub["n"])
            even = (sub["n"] % 2) == 0
            r_ck = jnp.where(even, r_ck.at[idx_max].set(rn), r_ck)
            rs_ck = jnp.where(even, rs_ck.at[idx_max].set(r_sum), rs_ck)

            def tcond(s: tuple[Array, Array]) -> Array:
                i, t = s
                return (i >= idx_min) & ~t

            def tbody(s: tuple[Array, Array]) -> tuple[Array, Array]:
                i, _ = s
                rho = r_sum - rs_ck[i] + r_ck[i]
                return i - 1, turning(r_ck[i], rn, rho)

            _, is_turn = jax.lax.while_loop(tcond, tbody, (idx_max, jnp.asarray(False)))

            sub = Subtree(
                last=(zn, rn, gn),
                prop=prop,
                logw=logw_tot,
                r_sum=r_sum,
                n=sub["n"] + 1,
                acc=sub["acc"] + acc,
                turning=is_turn,
                diverging=div,
            )
            return sub, key, r_ck, rs_ck

        return jax.lax.while_loop(cond, body, state0)[0]

    # -- repeatedly double the trajectory ---------------------------------------------------

    tree0 = Tree(
        left=(z, r0, g0),
        right=(z, r0, g0),
        prop=(z, ll, lp),
        logw=jnp.asarray(0.0),
        r_sum=r0,
        n=i32(0),
        acc=jnp.asarray(0.0),
        turning=jnp.asarray(False),
        diverging=jnp.asarray(False),
        depth=i32(0),
    )

    def cond(carry: tuple[Tree, Array]) -> Array:
        tree = carry[0]
        return (tree["depth"] < max_depth) & ~tree["turning"] & ~tree["diverging"]

    def body(carry: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, key = carry
        key, k_dir, k_sub, k_acc = jax.random.split(key, 4)
        going_right = jax.random.bernoulli(k_dir)
        edge: Edge = jax.tree_util.tree_map(
            lambda a, b: jnp.where(going_right, a, b), tree["right"], tree["left"]
        )
        sub = build_subtree(tree["depth"], edge, going_right, k_sub)

        ok = ~sub["turning"] & ~sub["diverging"]
        p = jnp.exp(jnp.minimum(0.0, sub["logw"] - tree["logw"]))
        take = ok & (jax.random.uniform(k_acc) < p)
        prop = jax.tree_util.tree_map(
            lambda new, old: jnp.where(take, new, old), sub["prop"], tree["prop"]
        )

        sub_edge = sub["last"]
        left = jax.tree_util.tree_map(
            lambda a, b: jnp.where(going_right, a, b), tree["left"], sub_edge
        )
        right = jax.tree_util.tree_map(
            lambda a, b: jnp.where(going_right, a, b), sub_edge, tree["right"]
        )
        r_sum = tree["r_sum"] + sub["r_sum"]
        tree = Tree(
            left=left,
            right=right,
            prop=prop,
            logw=jnp.logaddexp(tree["logw"], sub["logw"]),
            r_sum=r_sum,
            n=tree["n"] + sub["n"],
            acc=tree["acc"] + sub["acc"],
            turning=sub["turning"] | turning(left[1], right[1], r_sum),
            diverging=sub["diverging"],
            depth=tree["depth"] + 1,
        )
        return tree, key

    tree, _ = jax.lax.while_loop(cond, body, (tree0, key))
    z_new, ll_new, lp_new = tree["prop"]
    alpha = tree["acc"] / jnp.maximum(tree["n"], 1).astype(float)
    moved = jnp.any(z_new != z)
    return z_new, ll_new, lp_new, alpha, moved
