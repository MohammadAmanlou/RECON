"""RECON-Ω: calibrated synthetic research pilot, not neural-model evidence."""
from .core import (B, B4, World, make_world, hyperedges, feature_matrix,
                   select_packets, sample_contrasts, group_lasso,
                   reconstruct_theta, support_f1, exact_identification_witness,
                   confirm_interaction)
__all__ = ["B", "B4", "World", "make_world", "hyperedges", "feature_matrix",
           "select_packets", "sample_contrasts", "group_lasso", "reconstruct_theta",
           "support_f1", "exact_identification_witness", "confirm_interaction"]