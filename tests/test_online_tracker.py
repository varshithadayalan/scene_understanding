from matcher.online_tracker import OnlineTracker


def test_ids_stable_across_frames(frames):
    # Motion-only tracker (no model) must keep a consistent synthetic ID per
    # instance across the 3 synthetic frames (linear, well-separated motion).
    tr = OnlineTracker(model=None, use_embeddings=False)
    id_by_instance = {}
    for frame in frames:
        ids = tr.update(frame)
        assert len(ids) == len(frame)
        for node, tid in zip(frame, ids):
            inst = node["instance"]
            if inst in id_by_instance:
                assert id_by_instance[inst] == tid, f"ID switch for {inst}"
            else:
                id_by_instance[inst] = tid
    # Two distinct instances -> two distinct stable track IDs.
    assert len(set(id_by_instance.values())) == 2


def test_no_gt_leak_in_track_state(frames):
    # The tracker must never store the ground-truth instance token in its state.
    tr = OnlineTracker(model=None, use_embeddings=False)
    tr.update(frames[0])
    instances = {n["instance"] for f in frames for n in f}
    for track in tr.tracks.values():
        assert "instance" not in track
        assert track.get("label") not in instances  # label is category, not GT id


def test_new_tracks_get_unique_ids(frames):
    tr = OnlineTracker(model=None, use_embeddings=False)
    ids = tr.update(frames[0])
    assert len(set(ids)) == len(ids)
