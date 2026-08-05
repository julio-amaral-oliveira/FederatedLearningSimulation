"""E08: partial drift and collective trigger policy.

E08 characterizes the coverage line of the collective trigger policy.
Only a subset of clients drifts. The policy keeps the E07 design: local
detection per client, quorum at the server, global retraining.

This experiment does not try to make the policy work under partial drift.
It measures the boundary: the minimum fleet fraction that triggers, and
the cost of the mixture (downtime, clean retention, retraining cost).
"""
