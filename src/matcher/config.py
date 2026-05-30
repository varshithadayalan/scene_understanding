class ResearchConfig:
    """
    Microsoft-Level Research Configuration System.
    Centralizes hyperparameters for systematic scaling studies.
    """
    def __init__(self):
        # Dataset
        self.dataroot = 'C:/Users/varsh/nuscenes_project/data/sets/nuscenes'
        self.version = 'v1.0-mini' # Change to 'v1.0-trainval' for full scale
        
        # Model Hyperparameters
        self.alpha_base = 0.4    # Embedding trust
        self.beta_base = 0.4     # Motion trust
        self.gamma_base = 0.2    # Uncertainty base
        self.persistence_weight = 2.0 # ID Inertia penalty
        
        # Memory / Temporal Buffer
        self.max_age = 10        # Frames to persist "LOST" tracks
        self.re_id_threshold = 5.0 # Base matching threshold
        self.age_threshold_step = 1.5 # How much threshold grows per frame of age
        
        # Systems
        self.use_gpu = False     # Toggle for large scale runs
