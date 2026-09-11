"""Evaluation: the harness, its statistics, its registry and its controls.

Deliberately separate from `core/`. Core answers "what is my portfolio worth
and how risky is it"; this package answers "would I have been able to tell
whether a strategy worked", which is a different question with different
failure modes. Nothing here touches the network, a broker, or live money.
"""
