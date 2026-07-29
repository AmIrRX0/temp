"""Memoisation of access plans.

Callers typically gather from the same table many times -- one call per batch,
per layer, per step -- so re-deriving the plan on every call is pure overhead.
Plans are immutable, so they are safe to hand out repeatedly.
"""

from typing import Dict, Tuple

import torch

from .config import GatherConfig
from .plan import AccessPlan, build_plan

_PLANS: Dict[Tuple, AccessPlan] = {}


def plan_key(table: torch.Tensor, config: GatherConfig) -> Tuple:
    """Cache identity for the plan belonging to ``table`` under ``config``.

    Geometry and config are the inputs to :func:`gatherlib.plan.build_plan`, so
    they both belong in the key. Layout is in there twice over: whether the
    table is packed, since a freshly allocated table and a view of a larger one
    are addressed differently even when they agree on shape, and the pitch of
    the trailing axis, which is what a transpose or a column slice changes.
    """
    return (
        tuple(int(s) for s in table.shape),
        table.dtype,
        table.device,
        table.is_contiguous(),
        int(table.stride(-1)),
        config,
    )


def plan_for(table: torch.Tensor, config: GatherConfig) -> AccessPlan:
    """Return the plan for ``table``, deriving it only on a cache miss."""
    key = plan_key(table, config)
    plan = _PLANS.get(key)
    if plan is None:
        plan = build_plan(table, config)
        _PLANS[key] = plan
    return plan


def cached_plan_count() -> int:
    """How many distinct plans are currently held."""
    return len(_PLANS)


def clear_plans() -> None:
    """Drop every cached plan.

    Worth calling after the tables a process was working on have gone away,
    since the cache holds their dtypes and devices alive in its keys.
    """
    _PLANS.clear()
