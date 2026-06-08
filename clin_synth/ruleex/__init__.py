"""Association-rule mining for clinical seed data — see arm.mine_rules."""

from __future__ import annotations

from clin_synth.ruleex.arm import generate_soft_rules_instruction_block, mine_rules

__all__ = ["mine_rules", "generate_soft_rules_instruction_block"]
