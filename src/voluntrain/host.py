import torch
import zmq
import pickle
import cloudpickle
import time
from .protocol import encode_id, get_local_ip, serialize, deserialize

class ElasticHost:  # <--- Pastikan namanya ini!
    def __init__(self, model, optimizer, port=5555):
        self.model = model
        self.optimizer = optimizer
        self.context = zmq.Context()
        self.active_workers = 0
        
        # 1. PUB Socket: Broadcast Weights
        self.pub = self.context.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{port}")
        
        # 2. PULL Socket: Receive Gradients
        self.pull = self.context.socket(zmq.PULL)
        self.pull.bind(f"tcp://*:{port+1}")
        
        # 3. REP Socket: Registration Desk
        self.reg_socket = self.context.socket(zmq.REP)
        self.reg_socket.bind(f"tcp://*:{port+2}")
        
        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.pull, zmq.POLLIN)
        self.poller.register(self.reg_socket, zmq.POLLIN)
        
        # ID Generation
        self.ip = get_local_ip()
        self.join_id = encode_id(self.ip, port)
        
        print(f"Host ready on port {port}")
        print(f"=== SHARE THIS ID TO WORKERS ===")
        print(f"{self.join_id}")
        print(f"================================")

    def check_for_new_workers(self):
        try:
            while self.reg_socket.poll(0): 
                msg = self.reg_socket.recv_string()
                if msg == "KNOCK_KNOCK":
                    self.active_workers += 1
                    print(f"A new worker joined! Total helpers: {self.active_workers}")
                    self.reg_socket.send_string("WELCOME")
        except zmq.ZMQError:
            pass

    def train_step(self, *args, **kwargs):
        self.check_for_new_workers()
        
# ... (di dalam def train_step) ...
        
        # Local Forward Pass
        self.optimizer.zero_grad()
        output = self.model(*args, **kwargs)
        
        # Handle output types
        if isinstance(output, torch.Tensor): loss = output
        elif hasattr(output, 'loss'): loss = output.loss
        else: loss = output[0]
            
        # === FIX DIMULAI DI SINI ===
        # Jika loss berupa vector (misal [32, 1]), kita harus menjadikannya scalar
        if loss.numel() > 1:
            loss = loss.mean() 
        # === FIX BERAKHIR DI SINI ===

        loss.backward() # Sekarang aman karena loss sudah pasti 1 angka
        
        # ... (lanjut ke Distributed Logic) ...
        
        # Distributed Logic
        if self.active_workers > 0:
            payload = {
                "model_structure": cloudpickle.dumps(self.model),
                "state_dict": self.model.state_dict(),
                "data_args": args,
                "data_kwargs": kwargs
            }
            self.pub.send(pickle.dumps(payload))
            
            collected_grads = 0
            # Wait up to 2 seconds for workers
            while collected_grads < self.active_workers:
                socks = dict(self.poller.poll(2000)) 
                
                if self.pull in socks:
                    msg = self.pull.recv()
                    remote_grads = pickle.loads(msg)
                    
                    for param, r_grad in zip(self.model.parameters(), remote_grads):
                        if param.grad is None:
                            param.grad = r_grad
                        else:
                            param.grad += r_grad
                    
                    collected_grads += 1
                else:
                    print("Worker timeout! Continuing...")
                    break

            # Average gradients
            total_participants = 1 + collected_grads
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= total_participants
                    
        self.optimizer.step()
        
        if self.active_workers > 0:
            print(f"Step complete. Host + {self.active_workers} Workers.")
        else:
            print("Step complete (Host only).")