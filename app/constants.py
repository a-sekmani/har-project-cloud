"""Application constants (e.g. alert activities for dashboard)."""

# Activity labels considered as alerts / critical for dashboard Fall Alerts and Recent Important Events
ALERT_ACTIVITIES = frozenset({
    "falling_down",
    "chest_pain",
    "nausea_vomiting",
    "headache",
    "back_pain",
})
