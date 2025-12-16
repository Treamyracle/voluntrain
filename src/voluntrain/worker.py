import torch
import zmq
import pickle
import cloudpickle
import time
from .protocol import decode_id, serialize, deserialize

class ElasticWorker:
    def __init__(self, join_id):
        # 1. Decode ID menjadi IP dan Port asli
        host_ip, port = decode_id(join_id)
        
        self.context = zmq.Context()
        self.host_ip = host_ip
        self.port = port
        
        print(f"Connecting to Host at {host_ip}:{port}...")
        
        # --- SOCKET SETUP (Harus urutannya sama dengan Host) ---
        
        # A. Handshake Socket (REQ) -> Connect ke Host REP (Port + 2)
        # Gunanya: Bilang "Halo saya mau bantu" ke Host
        self.reg_socket = self.context.socket(zmq.REQ)
        self.reg_socket.connect(f"tcp://{host_ip}:{port+2}")
        
        # Kirim salam
        print("Knocking on door...")
        self.reg_socket.send_string("KNOCK_KNOCK")
        
        # Tunggu balasan "WELCOME" dari Host
        resp = self.reg_socket.recv_string() 
        if resp == "WELCOME":
            print("Access Granted! Successfully joined the training cluster.")
        else:
            print(f"Warning: Received unexpected response: {resp}")
        
        # B. Subscribe Socket (SUB) -> Connect ke Host PUB (Port Utama)
        # Gunanya: Menerima Model dan Data terbaru
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect(f"tcp://{host_ip}:{port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, '') # Subscribe ke semua pesan
        
        # C. Push Socket (PUSH) -> Connect ke Host PULL (Port + 1)
        # Gunanya: Mengirim hasil hitungan (Gradient) balik ke Host
        self.push = self.context.socket(zmq.PUSH)
        self.push.connect(f"tcp://{host_ip}:{port+1}")

    def start(self):
        print("Waiting for work (Model & Data)...")
        while True:
            try:
                # 1. Terima Paket (Weights + Model Code + Data)
                msg = self.sub.recv() 
                payload = pickle.loads(msg)
                
                # 2. Reconstruct Model (Cloudpickle Magic)
                # Worker tidak perlu file model.py, ia membaca definisi class langsung dari bytes
                model = cloudpickle.loads(payload["model_structure"])
                model.load_state_dict(payload["state_dict"])
                
                # 3. Siapkan Data Input
                args = payload["data_args"]
                kwargs = payload["data_kwargs"]
                
                # ... (di dalam while True) ...
                
                # 4. Forward Pass
                output = model(*args, **kwargs)
                
                if isinstance(output, torch.Tensor): loss = output
                elif hasattr(output, 'loss'): loss = output.loss
                else: loss = output[0]
                
                # === FIX DIMULAI DI SINI ===
                # Pastikan loss menjadi scalar sebelum backward
                if loss.numel() > 1:
                    loss = loss.mean()
                # === FIX BERAKHIR DI SINI ===
                
                # 5. Backward Pass
                loss.backward()
                
                # ... (lanjut ke kirim gradient) ...
                
                # 6. Ambil Gradient
                grads = [p.grad for p in model.parameters()]
                
                # 7. Kirim Gradient Balik ke Host
                self.push.send(pickle.dumps(grads))
                print(f"Batch processed. Gradients sent to Host.")
                
            except KeyboardInterrupt:
                print("\nStopping worker...")
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Kita continue agar worker tidak mati total hanya karena 1 batch error
                continue