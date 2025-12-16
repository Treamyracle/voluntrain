import socket
import base64
import pickle
import zlib  # Opsional: untuk kompresi data agar pengiriman lebih cepat

def get_local_ip():
    """
    Mencoba menghubungkan ke DNS Google (8.8.8.8) untuk mengetahui
    IP Address mana yang digunakan komputer ini untuk akses jaringan.
    Tidak ada data yang dikirim sebenarnya.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # IP ini tidak perlu bisa dijangkau, hanya untuk cek routing
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        # Fallback jika tidak ada koneksi internet/jaringan
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def encode_id(ip: str, port: int) -> str:
    """
    Mengubah IP '192.168.1.5' dan Port 5555 menjadi string acak
    Contoh: 'MTkyLjE2OC4xLjU6NTU1NQ=='
    Ini agar user merasa 'Join ID' lebih rapi daripada mengetik IP.
    """
    raw_str = f"{ip}:{port}"
    # Encode ke bytes, lalu ke base64, lalu decode kembali ke string utf-8
    return base64.b64encode(raw_str.encode()).decode()

def decode_id(token: str):
    """
    Kebalikan dari encode_id.
    Mengubah ID acak kembali menjadi (ip, port).
    """
    try:
        decoded_bytes = base64.b64decode(token.encode())
        decoded_str = decoded_bytes.decode()
        ip, port_str = decoded_str.split(':')
        return ip, int(port_str)
    except Exception as e:
        raise ValueError(f"ID tidak valid: {token}. Pastikan Anda menyalin ID dengan benar.")

def serialize(data):
    """
    Mengubah objek Python (list, dict, tensor) menjadi bytes untuk dikirim.
    Menggunakan pickle protocol tertinggi untuk performa.
    """
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize(data):
    """
    Mengubah bytes kembali menjadi objek Python.
    """
    return pickle.loads(data)