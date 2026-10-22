"""Upstream DrivoR model (valeoai/DrivoR, navsim/agents/drivoR): model.py, model_utils/ and
the agent config drivor_original_agent.yaml. Only the network is here: no features, loss or
scoring. Local change, marked ALPASIM-HPC PATCH: ImgEncoder honours image_backbone.pretrained,
so the driver builds the DINOv2 encoder without its pretrained weights (the checkpoint sets them).
"""
