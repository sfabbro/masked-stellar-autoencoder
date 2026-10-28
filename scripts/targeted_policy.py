"""Explicit per-arm controls, preserving all old numerical defaults."""


def resolve_policy(suite, arm):
    changes = arm.get("policy_overrides", {})
    allowed = {"checkpoint", "checkpoint_sha256", "encoder_lr", "reconstruction_weight"}
    if set(changes) - allowed:
        raise ValueError("Unsupported per-arm policy override")
    if ("checkpoint" in changes) != ("checkpoint_sha256" in changes):
        raise ValueError("A new parent requires both path and digest")
    return {**suite, **changes}


def experiment_seed(suite, arm):
    return int(arm.get("experiment_seed", suite.get("continuation_seed", 42)))


def dropout_offset(seed):
    return (seed - 42) * 10_000_000
