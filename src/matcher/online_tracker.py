"""Honest online multi-object tracker (no ground-truth leakage).

Unlike ResearchEngine (which keys tracks by the ground-truth ``instance_token``
and adds a persistence penalty when a candidate detection's GT identity differs
from the track's GT identity), this tracker:

  * assigns its OWN arbitrary integer track IDs,
  * never reads ``instance`` during matching (it is used only by the evaluator
    for scoring),
  * associates detections to tracks with the Hungarian algorithm over a cost of
    trained embedding distance + motion (trajectory-buffer) cost + uncertainty,
    with an observable category gate and a distance gate.

This is what makes a metric like IDF1 / IDSW meaningful: the tracker has to keep
IDs consistent on its own.
"""
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from matcher.hybrid_core import embedding_distance

_BIG = 1e6


def _feat6(d, pos=None):
    """Build the 6-D embedding feature for a detection node or a track dict."""
    p = pos if pos is not None else d.get("position", d.get("pos"))
    conf = d.get("confidence", d.get("conf"))
    unc = d.get("uncertainty", d.get("unc"))
    return [p[0], p[1], d["vx"], d["vy"], conf, unc]


class OnlineTracker:
    def __init__(self, model=None, *, use_embeddings=True,
                 alpha=0.4, beta=0.4, gamma=0.2,
                 max_age=10, gate=4.0, age_gate_step=1.5,
                 motion_clip=10.0, use_buffer=True):
        self.model = model
        self.use_embeddings = use_embeddings and (model is not None)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.max_age = max_age
        self.gate = gate
        self.age_gate_step = age_gate_step
        self.motion_clip = motion_clip
        self.use_buffer = use_buffer
        self.tracks = {}          # track_id -> dict(pos,vx,vy,label,conf,unc,age)
        self._next_id = 0

    def _new_track(self, node):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {
            "pos": [node["position"][0], node["position"][1]],
            "vx": node["vx"], "vy": node["vy"], "label": node["label"],
            "conf": node["confidence"], "unc": node["uncertainty"], "age": 0,
        }
        return tid

    def _predicted_pos(self, track):
        if self.use_buffer and track["age"] > 0:
            return [track["pos"][0] + track["vx"] * track["age"],
                    track["pos"][1] + track["vy"] * track["age"]]
        return track["pos"]

    def _emb_for(self, feat6):
        with torch.no_grad():
            return self.model.forward_embedding(
                torch.tensor(feat6, dtype=torch.float32)).numpy()

    def update(self, detections):
        """Advance one frame. Returns a list: track_id assigned to each detection."""
        if not detections:
            for t in self.tracks.values():
                t["age"] += 1
            self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}
            return []

        if not self.tracks:
            return [self._new_track(n) for n in detections]

        track_ids = list(self.tracks.keys())
        n_t, n_d = len(track_ids), len(detections)
        cost = np.full((n_t, n_d), _BIG, dtype=float)

        det_emb = [self._emb_for(_feat6(d)) for d in detections] if self.use_embeddings else None

        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            pred = self._predicted_pos(tr)
            gate_i = self.gate + tr["age"] * self.age_gate_step
            tr_emb = self._emb_for(_feat6(tr, pred)) if self.use_embeddings else None
            for j, d in enumerate(detections):
                raw_dist = math.hypot(d["position"][0] - pred[0], d["position"][1] - pred[1])
                if raw_dist > gate_i or tr["label"] != d["label"]:
                    continue  # disallowed: out of gate or different observable category
                motion_cost = min(raw_dist, self.motion_clip) / self.motion_clip
                emb_dist = embedding_distance(tr_emb, det_emb[j]) if self.use_embeddings else 0.0
                unc = (tr["unc"] + d["uncertainty"]) / 2.0
                cost[i, j] = self.alpha * emb_dist + self.beta * motion_cost + self.gamma * unc

        row, col = linear_sum_assignment(cost)
        assigned_det = {}        # det_idx -> track_id
        matched_tracks = set()
        for r, c in zip(row, col):
            if cost[r, c] >= _BIG:
                continue
            tid = track_ids[r]
            d = detections[c]
            tr = self.tracks[tid]
            tr["vx"] = d["position"][0] - tr["pos"][0]
            tr["vy"] = d["position"][1] - tr["pos"][1]
            tr["pos"] = [d["position"][0], d["position"][1]]
            tr["label"], tr["conf"], tr["unc"], tr["age"] = d["label"], d["confidence"], d["uncertainty"], 0
            assigned_det[c] = tid
            matched_tracks.add(tid)

        # Age / retire unmatched tracks.
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["age"] += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}

        # New tracks for unmatched detections.
        result = []
        for j, d in enumerate(detections):
            result.append(assigned_det[j] if j in assigned_det else self._new_track(d))
        return result
