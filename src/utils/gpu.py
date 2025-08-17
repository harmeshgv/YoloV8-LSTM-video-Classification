# backend/utils/gpu.py
import torch
import os

class GPUConfigurator:
    def __init__(self):
        self.device = self._setup_device()
        self._configure_gpu()
    
    def _setup_device(self):
        """Determine and setup the appropriate device (GPU or CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Warm-up GPU
            torch.zeros(1).to(device)
            torch.cuda.synchronize()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        print("No GPU available. Using CPU.")
        return torch.device("cpu")
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'