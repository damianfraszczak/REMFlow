"""Actor-oriented and tie-oriented statistic consistency."""

from tests.remstats.test_aomstats import (
    test_actor_decay_memory_matches_tie_kernels,
    test_actor_receiver_typed_slices_match_tie_slices,
    test_actor_sender_and_receiver_statistics_equal_tie_kernels,
)

__all__ = [
    "test_actor_sender_and_receiver_statistics_equal_tie_kernels",
    "test_actor_receiver_typed_slices_match_tie_slices",
    "test_actor_decay_memory_matches_tie_kernels",
]
