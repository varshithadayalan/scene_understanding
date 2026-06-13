import math
import os

import torch
import torch.nn as nn

class HybridMatcher(nn.Module):
    """
    Microsoft-Level Research Implementation: Uncertainty-Aware Hybrid Matcher.
    Combines Appearance Embeddings and Motion Prediction.
    """
    def __init__(self, emb_dim=16, hidden_dim=64):
        super().__init__()
        
        # Branch 1: Appearance/Spatial Embedding
        # Inputs: x, y, vx, vy, confidence, uncertainty
        self.embedding_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, emb_dim) 
        )
        
        # Branch 2: Motion Scoring (Binary Classifier)
        # Inputs: dx, dy, dist_norm, same_label, c1, c2, u1, u2, vx1, vy1, v_diff
        self.motion_net = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward_embedding(self, feat_vec):
        return self.embedding_net(feat_vec)

    def forward_motion(self, motion_feat):
        return torch.sigmoid(self.motion_net(motion_feat))

    @staticmethod
    def compute_distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def load_hybrid_matcher(checkpoint_path=None, emb_dim=16, hidden_dim=64):
    """Build a HybridMatcher, loading trained weights if a checkpoint exists.

    If ``checkpoint_path`` is falsy or missing, returns a randomly-initialized
    model (legacy behavior) so existing scripts keep working unchanged.
    """
    model = HybridMatcher(emb_dim=emb_dim, hidden_dim=hidden_dim)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[model] loaded trained checkpoint: {checkpoint_path}")
    else:
        if checkpoint_path:
            print(f"[model] checkpoint not found ({checkpoint_path}); using random init")
    model.eval()
    return model
