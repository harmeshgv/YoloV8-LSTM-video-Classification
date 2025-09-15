import torch
import os

class GPUConfigurator:
    def __init__(self):
        self.device = self._setup_device()
        self._configure_gpu()
    
    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.zeros(1).to(device)
            torch.cuda.synchronize()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        print("No GPU available. Using CPU.")
        return torch.device("cpu")
    
    def _configure_gpu(self):
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
