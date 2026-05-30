import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from matcher.config import ResearchConfig

class MOTFormatter:
    """
    Exports tracker output and ground truth to MOT Challenge format.
    Format: [frame, id, left, top, width, height, conf, -1, -1, -1]
    """
    def __init__(self, output_dir="data/mot_eval"):
        self.output_dir = output_dir
        self.instance_to_id = {}
        self.id_counter = 1

    def get_int_id(self, instance_token):
        if instance_token not in self.instance_to_id:
            self.instance_to_id[instance_token] = self.id_counter
            self.id_counter += 1
        return self.instance_to_id[instance_token]

    def format_row(self, frame_idx, instance_token, pos, size, conf):
        track_id = self.get_int_id(instance_token)
        # BEV Map: x, y center -> left, top
        # size = [w, l, h] in nuScenes
        width = size[1] # length (x)
        height = size[0] # width (y)
        left = pos[0] - (width / 2)
        top = pos[1] - (height / 2)
        
        # MOT format: frame, id, left, top, width, height, conf, class, visibility, unused
        # class=1, visibility=1
        return f"{frame_idx+1},{track_id},{left:.3f},{top:.2f},{width:.2f},{height:.2f},{conf:.2f},1,1,-1\n"

    def save_gt(self, nusc, scenes, tracker_name="UA-HTIM"):
        gt_root = os.path.join(self.output_dir, "gt", "mot_challenge", "train")
        os.makedirs(gt_root, exist_ok=True)
        
        seq_names = []
        for scene in scenes:
            seq_name = scene['name']
            seq_names.append(seq_name)
            seq_dir = os.path.join(gt_root, seq_name, "gt")
            os.makedirs(seq_dir, exist_ok=True)
            
            with open(os.path.join(seq_dir, "gt.txt"), "w") as f:
                samples = []
                curr = scene['first_sample_token']
                while curr != "":
                    s = nusc.get('sample', curr)
                    samples.append(s)
                    curr = s['next']
                
                for f_idx, sample in enumerate(samples):
                    for ann_token in sample['anns']:
                        ann = nusc.get('sample_annotation', ann_token)
                        row = self.format_row(f_idx, ann['instance_token'], ann['translation'], ann['size'], 1.0)
                        f.write(row)
            
            # Create seqinfo.ini
            with open(os.path.join(gt_root, seq_name, "seqinfo.ini"), "w") as f:
                f.write(f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=2\nseqLength={len(samples)}\nresX=1600\nresY=1200\nimExt=.jpg\n")

        # Create seqmap
        seqmap_dir = os.path.join(self.output_dir, "gt", "mot_challenge", "seqmaps")
        os.makedirs(seqmap_dir, exist_ok=True)
        with open(os.path.join(seqmap_dir, "mot_challenge-train.txt"), "w") as f:
            f.write("name\n")
            for name in seq_names:
                f.write(f"{name}\n")

    def save_tracker_output(self, seq_name, results, tracker_name="UA-HTIM"):
        track_dir = os.path.join(self.output_dir, "trackers", "mot_challenge", tracker_name, "data")
        os.makedirs(track_dir, exist_ok=True)
        
        with open(os.path.join(track_dir, f"{seq_name}.txt"), "w") as f:
            for row in results:
                f.write(row)

def run_formatter_for_ua_htim(nusc, scenes, model):
    from matcher.engine import ResearchEngine
    cfg = ResearchConfig()
    cfg.persistence_weight = 2.0 # Use best validated lambda
    engine = ResearchEngine(nusc, model, cfg)
    formatter = MOTFormatter()
    
    # 1. Save Ground Truth
    print("Saving Ground Truth in MOT format...")
    formatter.save_gt(nusc, scenes)
    
    # 2. Run UA-HTIM and Save
    print("Running UA-HTIM and saving MOT output...")
    for scene_info in scenes:
        engine.track_history = {}
        samples = []
        curr = scene_info['first_sample_token']
        while curr != "":
            samples.append(nusc.get('sample', curr))
            curr = samples[-1]['next']
            
        results = []
        for f_idx, sample in enumerate(samples):
            nodes = engine.extract_nodes(sample, samples[f_idx-1] if f_idx > 0 else None)
            engine.match(nodes)
            
            for tid, track in engine.track_history.items():
                if track["age"] == 0:
                    row = formatter.format_row(f_idx, tid, track["node"]["position"], track["node"]["size"], track["node"]["confidence"])
                    results.append(row)
        
        formatter.save_tracker_output(scene_info['name'], results, "UA-HTIM")

if __name__ == "__main__":
    from nuscenes import NuScenes
    from matcher.models import HybridMatcher
    
    DATAROOT = 'C:/Users/varsh/nuscenes_project/data/sets/nuscenes'
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    model = HybridMatcher()
    target_scenes = nusc.scene[:5]
    
    run_formatter_for_ua_htim(nusc, target_scenes, model)
    print("Formatting complete.")
