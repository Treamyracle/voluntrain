import torch
import zmq
import pickle
import cloudpickle
import time
from .protocol import decode_id, serialize, deserialize

class ElasticWorker:
    def __init__(self, join_id):
        # 1. Tentukan Device (CUDA jika ada, jika tidak CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Worker started using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

        host_ip, port = decode_id(join_id)
        self.context = zmq.Context()
        self.host_ip = host_ip
        self.port = port
        
        print(f"Connecting to Host at {host_ip}:{port}...")
        
        # --- SOCKET SETUP ---
        # A. Handshake
        self.reg_socket = self.context.socket(zmq.REQ)
        self.reg_socket.connect(f"tcp://{host_ip}:{port+2}")
        
        print("Knocking on door...")
        self.reg_socket.send_string("KNOCK_KNOCK")
        
        resp = self.reg_socket.recv_string() 
        if resp == "WELCOME":
            print("Access Granted! Joined the cluster.")
        else:
            print(f"Warning: Unexpected response: {resp}")
        
        # B. Subscribe (Weights)
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect(f"tcp://{host_ip}:{port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, '')
        
        # C. Push (Gradients)
        self.push = self.context.socket(zmq.PUSH)
        self.push.connect(f"tcp://{host_ip}:{port+1}")

    def _move_to_device(self, data):
        """Fungsi pembantu untuk memindahkan data (List/Tuple/Dict) ke GPU"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self._move_to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for v in data.items()}
        return data

    def start(self):
        print("Waiting for work...")
        while True:
            try:
                # 1. Terima Data
                msg = self.sub.recv() 
                payload = pickle.loads(msg)
                
                # 2. Load Model
                model = cloudpickle.loads(payload["model_structure"])
                model.load_state_dict(payload["state_dict"])
                
                # === UPGRADE: PINDAHKAN MODEL KE GPU ===
                model.to(self.device)
                model.train() # Pastikan mode training
                
                # 3. Siapkan Data Input
                # === UPGRADE: PINDAHKAN INPUT KE GPU ===
                args = self._move_to_device(payload["data_args"])
                kwargs = self._move_to_device(payload["data_kwargs"])
                
                # 4. Forward Pass (Sekarang berjalan di GPU!)
                output = model(*args, **kwargs)
                
                if isinstance(output, torch.Tensor): loss = output
                elif hasattr(output, 'loss'): loss = output.loss
                else: loss = output[0]
                
                # Fix scalar loss issue
                if loss.numel() > 1:
                    loss = loss.mean()
                
                # 5. Backward Pass (Hitung Gradient di GPU)
                loss.backward()
                
                # 6. Ambil Gradient & PINDAHKAN BALIK KE CPU
                # Kita harus pindah ke CPU sebelum dikirim lewat jaringan
                grads = [p.grad.cpu() for p in model.parameters() if p.grad is not None]
                
                # 7. Kirim
                self.push.send(pickle.dumps(grads))
                print(f"Batch processed on {self.device}. Sent to Host.")
                
            except KeyboardInterrupt:
                print("\nStopping worker...")
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Print detail error GPU jika ada
                import traceback
                traceback.print_exc()
                continue