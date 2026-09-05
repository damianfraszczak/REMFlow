"""End-to-end misinformation propagation example."""

from remflow import MisinformationModel

events = [
    (1.2, "u1", "u5", "retweet", "support"),
    (1.8, "u5", "u8", "reply", "deny"),
    (2.1, "u2", "u5", "mention", "question"),
    (2.7, "u8", "u1", "retweet", "support"),
    (3.4, "u5", "u2", "reply", "deny"),
]

model = MisinformationModel(
    effects=(
        "reciprocity",
        "sender_activity",
        "receiver_popularity",
        "triadic_closure",
        "recent_exposure",
        "stance_similarity",
    ),
    backend="numpy",
    ordinal=True,
).fit(events)

print(model.summary())
print(model.predict_next_events(top_k=5))
print(model.actor_roles())
print(model.detect_sources(top_k=3))
print(model.echo_chamber_metrics())
print(model.simulate_intervention(blocked_actors=["u5"]))
