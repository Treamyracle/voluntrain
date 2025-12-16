# 🚄 VolunTrain

**VolunTrain** adalah library Python ringan untuk melakukan *Distributed Training* (pelatihan AI terdistribusi) secara instan dan desentralisasi.

Ubah laptop teman, komputer kantor, atau warnet menjadi "GPU Farm" dadakan untuk mempercepat pelatihan model AI Anda. Cukup jalankan satu perintah, dan komputer lain bisa bergabung menggunakan **Session ID**.

---

## ✨ Fitur Utama

- **Elastic Scaling:** Worker bisa bergabung (*join*) atau keluar (*leave*) kapan saja tanpa mematikan proses training di Host.
- **Universal Device Support:** Mendukung **NVIDIA (CUDA)**, **AMD (ROCm)**, **Apple Silicon (MPS)**, dan CPU secara otomatis.
- **Zero-Config Discovery:** Menggunakan sistem ID berbasis Base64 sederhana. Tidak perlu setting IP manual yang ribet.
- **Framework Agnostic:** Bekerja dengan model PyTorch apa saja (CNN, RNN, Transformer, LLM, dll).

---

## 📦 Instalasi

### Prasyarat (PENTING!)

Agar sistem berjalan lancar, **Host** dan **Worker** harus menggunakan versi Python yang sama  
(disarankan **Python 3.11**) untuk menghindari error serialisasi model.

---

### 1. Install PyTorch (GPU Support)

Pastikan Anda menginstall PyTorch yang mendukung GPU di komputer Anda sebelum menginstall library ini.

**Untuk Pengguna NVIDIA (Windows/Linux):**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Untuk Pengguna Mac (M1/M2/M3):**

```bash
pip3 install torch torchvision torchaudio
```

---

### 2. Install VolunTrain

Clone repository ini dan install dalam mode editable:

```bash
git clone https://github.com/Treamyracle/voluntrain.git
cd voluntrain
pip install -e .
```

---

## 🚀 Cara Penggunaan

### 1. Menjadi Host (Server)

Host adalah komputer yang memiliki **Model**, **Data**, dan melakukan update bobot.

Buat file `train_host.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from voluntrain import Host

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

host = Host(model, optimizer, port=5555)

print("Mulai training...")

for i in range(1000):
    inputs = torch.randn(32, 10).to(device)
    host.train_step(inputs)
```

---

### 2. Menjadi Worker (Client)

Worker adalah komputer yang menyumbangkan tenaga komputasinya.

#### Cara Cepat (CLI)

```bash
voluntrain join MTkyLjE2OC4xLjMyOjU1NTU=
```

#### Cara Script (Python)

Buat file `run_worker.py`:

```python
from voluntrain import Worker

HOST_ID = "MTkyLjE2OC4xLjMyOjU1NTU="

worker = Worker(join_id=HOST_ID)
worker.start()
```

---

## 🔧 Troubleshooting

### 1. Worker Timeout atau Tidak Bisa Connect

- Pastikan Host dan Worker berada di jaringan yang sama
- Jika beda lokasi, gunakan VPN seperti **ZeroTier**
- Pastikan firewall tidak memblokir Python

---

### 2. Error Perbedaan Versi Python

Gunakan versi Python yang **sama persis** di Host dan Worker  
(disarankan **Python 3.11.x**)

---

### 3. GPU Tidak Terdeteksi

Cek PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Jika `False`, install ulang PyTorch CUDA.

---

## 📜 Lisensi

MIT License — Bebas digunakan dan dimodifikasi.
