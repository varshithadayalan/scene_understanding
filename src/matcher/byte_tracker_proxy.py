import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilterProxy:
    """
    Simplified Constant Velocity Kalman Filter for BEV tracking.
    State: [x, y, w, l, vx, vy]
    """
    def __init__(self):
        self.dt = 0.5 # 2Hz in nuScenes

    def predict(self, track):
        # x = x + vx * dt
        track['pos'][0] += track['v'][0] * self.dt
        track['pos'][1] += track['v'][1] * self.dt
        return track

    def update(self, track, detection):
        # Simple velocity estimate
        vx = (detection['pos'][0] - track['pos'][0]) / self.dt
        vy = (detection['pos'][1] - track['pos'][1]) / self.dt
        track['pos'] = detection['pos']
        track['size'] = detection['size']
        track['v'] = [vx, vy]
        return track

def iou_bev(box1, box2):
    """
    box = [x, y, w, l] where x,y is center
    """
    x1, y1, w1, l1 = box1
    x2, y2, w2, l2 = box2
    
    # Corners
    b1_x1, b1_y1 = x1 - l1/2, y1 - w1/2
    b1_x2, b1_y2 = x1 + l1/2, y1 + w1/2
    
    b2_x1, b2_y1 = x2 - l2/2, y2 - w2/2
    b2_x2, b2_y2 = x2 + l2/2, y2 + w2/2
    
    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (w1 * l1) + (w2 * l2) - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

class ByteTrackerProxy:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_age=10):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.tracks = [] # list of {id, pos, size, v, age, state}
        self.id_counter = 1
        self.kf = KalmanFilterProxy()

    def update(self, detections):
        """
        detections: list of {pos, size, conf}
        """
        # 1. Predict existing tracks
        for t in self.tracks:
            self.kf.predict(t)
            t['age'] += 1

        # 2. Split detections
        dets_high = [d for d in detections if d['conf'] >= self.track_thresh]
        dets_low = [d for d in detections if d['conf'] < self.track_thresh]

        # 3. Match high-score detections
        # Cost = 1 - IoU
        tracked_indices = [i for i, t in enumerate(self.tracks) if t['state'] == 'tracked']
        lost_indices = [i for i, t in enumerate(self.tracks) if t['state'] == 'lost']
        
        pool = tracked_indices + lost_indices
        matches_1, unmatched_tracks, unmatched_dets_high = self._match(pool, dets_high)

        # 4. Match low-score detections
        # Match remaining tracks (from pool) with low-score detections
        matches_2, unmatched_tracks_final, unmatched_dets_low = self._match(unmatched_tracks, dets_low)

        # 5. Handle matches
        matched_track_ids = set()
        for t_idx, d_idx in matches_1:
            self.tracks[t_idx] = self.kf.update(self.tracks[t_idx], dets_high[d_idx])
            self.tracks[t_idx]['state'] = 'tracked'
            self.tracks[t_idx]['age'] = 0
            matched_track_ids.add(t_idx)
            
        for t_idx, d_idx in matches_2:
            self.tracks[t_idx] = self.kf.update(self.tracks[t_idx], dets_low[d_idx])
            self.tracks[t_idx]['state'] = 'tracked'
            self.tracks[t_idx]['age'] = 0
            matched_track_ids.add(t_idx)

        # 6. Initialize new tracks from unmatched high-score detections
        for d_idx in unmatched_dets_high:
            d = dets_high[d_idx]
            self.tracks.append({
                'id': self.id_counter,
                'pos': d['pos'],
                'size': d['size'],
                'v': [0, 0],
                'age': 0,
                'state': 'tracked'
            })
            self.id_counter += 1

        # 7. Update lost status and remove old tracks
        remaining_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in matched_track_ids:
                if t['state'] == 'tracked':
                    t['state'] = 'lost'
                if t['age'] <= self.max_age:
                    remaining_tracks.append(t)
            else:
                remaining_tracks.append(t)
        self.tracks = remaining_tracks

        return self.tracks

    def _match(self, track_indices, detections):
        if not track_indices or not detections:
            return [], track_indices, list(range(len(detections)))
            
        cost_matrix = np.zeros((len(track_indices), len(detections)))
        for i, t_idx in enumerate(track_indices):
            t = self.tracks[t_idx]
            for j, d in enumerate(detections):
                # 1 - IoU
                box_t = [t['pos'][0], t['pos'][1], t['size'][0], t['size'][1]]
                box_d = [d['pos'][0], d['pos'][1], d['size'][0], d['size'][1]]
                cost_matrix[i][j] = 1.0 - iou_bev(box_t, box_d)

        r_ind, c_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(r_ind, c_ind):
            if cost_matrix[r][c] < self.match_thresh:
                matches.append((track_indices[r], c))
                unmatched_tracks.remove(track_indices[r])
                unmatched_dets.remove(c)
                
        return matches, unmatched_tracks, unmatched_dets
