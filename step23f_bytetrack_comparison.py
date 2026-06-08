import os
import sys
from nuscenes import NuScenes

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from matcher.byte_tracker_proxy import ByteTrackerProxy
from matcher.mot_formatter import MOTFormatter
from matcher.engine import ResearchEngine
from matcher.config import ResearchConfig
from matcher.models import HybridMatcher

def run_bytetrack_benchmark(nusc, scenes):
    formatter = MOTFormatter()
    cfg = ResearchConfig()
    model = HybridMatcher()
    engine = ResearchEngine(nusc, model, cfg) # only used for node extraction
    
    tracker_name = "ByteTrack"
    print(f"Running {tracker_name} and saving MOT output...")
    
    for scene_info in scenes:
        tracker = ByteTrackerProxy(track_thresh=0.5, match_thresh=0.8, max_age=10)
        samples = []
        curr = scene_info['first_sample_token']
        while curr != "":
            samples.append(nusc.get('sample', curr))
            curr = samples[-1]['next']
            
        results = []
        for f_idx, sample in enumerate(samples):
            nodes = engine.extract_nodes(sample, samples[f_idx-1] if f_idx > 0 else None)
            
            # Format nodes for ByteTrack
            tracker_inputs = []
            for n in nodes:
                tracker_inputs.append({
                    'pos': n['position'],
                    'size': n['size'],
                    'conf': n['confidence']
                })
            
            online_tracks = tracker.update(tracker_inputs)
            
            for t in online_tracks:
                if t['state'] == 'tracked':
                    # MOT format: frame, id, left, top, width, height, conf, class, visibility, unused
                    width = t['size'][1]
                    height = t['size'][0]
                    left = t['pos'][0] - (width / 2)
                    top = t['pos'][1] - (height / 2)
                    row = f"{f_idx+1},{t['id']},{left:.3f},{top:.2f},{width:.2f},{height:.2f},1.0,1,1,-1\n"
                    results.append(row)
        
        formatter.save_tracker_output(scene_info['name'], results, tracker_name)

if __name__ == "__main__":
    DATAROOT = os.environ.get("NUSCENES_DATAROOT", os.path.expanduser("~/data"))
    nusc = NuScenes(version=os.environ.get("NUSCENES_VERSION", "v1.0-mini"), dataroot=DATAROOT, verbose=False)
    target_scenes = nusc.scene[:5]
    
    run_bytetrack_benchmark(nusc, target_scenes)
    print("ByteTrack benchmarking complete.")
