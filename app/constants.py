"""Application constants (e.g. alert activities for dashboard)."""

# Activity labels shown as alert on the Alerts / Critical Events page (falling only).
# Accept both "falling_down" and "falling down" to match label_map.json (id_to_name often uses space).
ALERT_ACTIVITIES = frozenset({
    "falling_down",
    "falling down",
})
