"""Decision policies. An agent here is a function with an interface, not a
language model: it observes a point-in-time view of the world and returns a
`Proposal`. Nothing in this package calls an API or costs anything per
decision.
"""
