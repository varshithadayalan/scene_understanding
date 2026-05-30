import sys
import os
import numpy as np
# Fix for TrackEval compatibility with modern numpy
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

import trackeval

def run_eval(tracker_name="UA-HTIM"):
    config = trackeval.Evaluator.get_default_eval_config()
    gt_folder = os.path.join(os.getcwd(), 'data', 'mot_eval', 'gt', 'mot_challenge')
    trackers_folder = os.path.join(os.getcwd(), 'data', 'mot_eval', 'trackers', 'mot_challenge')
    
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = os.path.join(gt_folder, 'train')
    dataset_config['TRACKERS_FOLDER'] = trackers_folder
    dataset_config['TRACKERS_TO_EVAL'] = [tracker_name]
    dataset_config['BENCHMARK'] = 'mot_challenge'
    dataset_config['SPLIT_TO_EVAL'] = ''
    dataset_config['CLASSES_TO_EVAL'] = ['pedestrian']
    dataset_config['SEQMAP_FILE'] = os.path.join(gt_folder, 'seqmaps', 'mot_challenge-train.txt')
    dataset_config['SKIP_SPLIT_FOL'] = True
    
    # In our gt.txt, we didn't specify classes. MOT format uses column 8 as class.
    # Our formatter used -1. We should ideally update formatter to use a valid class ID if needed.
    # Actually, MOT Challenge class 1 is pedestrian.
    
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    
    evaluator = trackeval.Evaluator(config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        metrics_list.append(metric())
    
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    return output_res, output_msg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str, default='UA-HTIM')
    args = parser.parse_args()
    
    print(f"Executing evaluation for {args.tracker}...")
    try:
        res, msg = run_eval(tracker_name=args.tracker)
        print(f"\nEvaluation Results for {args.tracker}:")
        print(msg)
    except Exception as e:
        print(f"Error during evaluation: {e}")
