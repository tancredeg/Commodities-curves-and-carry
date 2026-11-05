"""Intrinsic value of a storage contract: a linear programme on today's forward curve.

The intrinsic value locks in the best injection/withdrawal schedule given the
forward curve ``F_1..F_M`` observed today, hedged with futures::

    I = max_{i, w}  Σ_m D_m [ F_m (w_m − i_m) − c_i i_m − c_w w_m ]
        s.t.  V_m = V_{m-1} + i_m − w_m,  0 ≤ V_m ≤ V̄,  0 ≤ i_m ≤ Ī,  0 ≤ w_m ≤ W̄,
              V_0 = initial,  V_M = terminal.

Two solvers are provided and tested against each other:

* :func:`intrinsic_lp` — ``scipy.optimize.linprog`` (HiGHS) on the problem above.
* :func:`intrinsic_grid` — backward dynamic programming on an inventory grid of
  step δ, vectorised over many curves at once (used by the Monte Carlo of the
  rolling-intrinsic valuation). The inventory constraints are partial sums of
  the decisions (an interval matrix, totally unimodular), so when V̄, Ī, W̄ and
  the boundary inventories are multiples of δ the LP has an optimal vertex on
  the grid and the DP is exact, not an approximation.

Sign convention of plans: ``net_m = i_m − w_m`` (positive = injection, i.e. a
purchase). With positive costs, injecting and withdrawing in the same month is
never optimal, so a plan is fully described by ``net``.

References: Secomandi (2010); Lai, Margot & Secomandi (2010).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from energycurves.storage.contract import StorageContract


@dataclass(frozen=True)
class IntrinsicPlan:
    """Optimal schedule on one curve: value, monthly injections/withdrawals, inventory."""

    value: float
    injection: np.ndarray
    withdrawal: np.ndarray
    inventory: np.ndarray  # end-of-month inventory V_1..V_M

    @property
    def net(self) -> np.ndarray:
        """Net injection ``i − w`` per month."""
        out: np.ndarray = self.injection - self.withdrawal
        return out


def _check_inputs(contract: StorageContract, forwards: np.ndarray, discount: np.ndarray) -> None:
    if forwards.shape[-1] != contract.n_months or discount.shape != (contract.n_months,):
        raise ValueError("forwards and discount factors must have one value per contract month")


def plan_value(
    net: np.ndarray, forwards: np.ndarray, discount: np.ndarray, contract: StorageContract
) -> np.ndarray:
    """Discounted cash flow ``Σ D_m [−F_m net_m − c_i (net_m)^+ − c_w (net_m)^−]``.

    Broadcasts over leading axes: ``net`` and ``forwards`` of shape ``(..., M)``.
    """
    inj = np.maximum(net, 0.0)
    wd = np.maximum(-net, 0.0)
    flows = -forwards * net - contract.cost_injection * inj - contract.cost_withdrawal * wd
    out: np.ndarray = (flows * discount).sum(axis=-1)
    return out


def intrinsic_lp(
    contract: StorageContract, forwards: np.ndarray, discount: np.ndarray
) -> IntrinsicPlan:
    """Solve the intrinsic LP with HiGHS. Variables ``x = [i_1..i_M, w_1..w_M]``."""
    forwards = np.asarray(forwards, dtype=float)
    discount = np.asarray(discount, dtype=float)
    _check_inputs(contract, forwards, discount)
    M = contract.n_months
    # minimise −profit: buying costs D(F + c_i), selling earns D(F − c_w)
    c = np.concatenate(
        [
            discount * (forwards + contract.cost_injection),
            -discount * (forwards - contract.cost_withdrawal),
        ]
    )
    lower = np.tril(np.ones((M, M)))  # V_m − V_0 = Σ_{k≤m} (i_k − w_k)
    a_ub = np.vstack([np.hstack([lower, -lower]), np.hstack([-lower, lower])])
    b_ub = np.concatenate(
        [np.full(M, contract.capacity - contract.initial), np.full(M, contract.initial)]
    )
    a_eq = np.concatenate([np.ones(M), -np.ones(M)])[None, :]
    b_eq = np.array([contract.terminal - contract.initial])
    bounds = [(0.0, contract.max_injection)] * M + [(0.0, contract.max_withdrawal)] * M
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"intrinsic LP failed: {res.message}")
    inj = np.clip(res.x[:M], 0.0, None)
    wd = np.clip(res.x[M:], 0.0, None)
    inventory = contract.initial + np.cumsum(inj - wd)
    return IntrinsicPlan(value=float(-res.fun), injection=inj, withdrawal=wd, inventory=inventory)


@dataclass(frozen=True)
class GridSolution:
    """DP solution for a batch of curves: values ``(P,)`` and net plans ``(P, M)``."""

    value: np.ndarray
    net: np.ndarray


def intrinsic_grid(
    contract: StorageContract,
    forwards: np.ndarray,
    discount: np.ndarray,
    first_month: int = 0,
    start_inventory: np.ndarray | float | None = None,
) -> GridSolution:
    """Backward DP on the inventory grid, vectorised over ``P`` curves.

    ``forwards`` has shape ``(P, M)`` (or ``(M,)``). Months before ``first_month``
    are already decided: the DP optimises months ``first_month..M−1`` starting
    from ``start_inventory`` (per curve, default ``contract.initial``) and
    returns their discounted value; ``net`` is zero on the earlier months.
    Raises if the contract has no exact grid (see ``StorageContract.grid_step``).
    """
    fwd = np.atleast_2d(np.asarray(forwards, dtype=float))
    discount = np.asarray(discount, dtype=float)
    _check_inputs(contract, fwd, discount)
    step = contract.grid_step()
    if step is None:
        raise ValueError("contract parameters are not multiples of a common grid step")
    P, M = fwd.shape
    n_states = round(contract.capacity / step) + 1
    n_wd, n_inj = round(contract.max_withdrawal / step), round(contract.max_injection / step)
    actions = np.arange(-n_wd, n_inj + 1)
    terminal = round(contract.terminal / step)
    start = (
        np.full(P, contract.initial)
        if start_inventory is None
        else np.broadcast_to(np.asarray(start_inventory, dtype=float), (P,))
    )
    s0 = np.rint(start / step).astype(int)
    if np.any(np.abs(s0 * step - start) > 1e-9) or np.any((s0 < 0) | (s0 >= n_states)):
        raise ValueError("start inventory is not on the grid")

    value_next = np.full((n_states, P), -np.inf)
    value_next[terminal] = 0.0
    best = np.zeros((M, n_states, P), dtype=np.int16)
    states = np.arange(n_states)
    for m in range(M - 1, first_month - 1, -1):
        volume = actions * step  # (A,)
        reward = discount[m] * (
            -fwd[:, m][None, :] * volume[:, None]
            - contract.cost_injection * np.maximum(volume, 0.0)[:, None]
            - contract.cost_withdrawal * np.maximum(-volume, 0.0)[:, None]
        )  # (A, P)
        cand = np.full((len(actions), n_states, P), -np.inf)
        for ai, a in enumerate(actions):
            nxt = states + a
            ok = (nxt >= 0) & (nxt < n_states)
            cand[ai, ok] = reward[ai][None, :] + value_next[nxt[ok]]
        best[m] = cand.argmax(axis=0)
        value_next = cand.max(axis=0)

    paths = np.arange(P)
    value = value_next[s0, paths]
    if not np.all(np.isfinite(value)):
        raise ValueError("terminal inventory is unreachable from the start inventory")
    net = np.zeros((P, M))
    s = s0.copy()
    for m in range(first_month, M):
        a = actions[best[m, s, paths]]
        net[:, m] = a * step
        s = s + a
    return GridSolution(value=value, net=net)
