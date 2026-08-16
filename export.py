from ultralytics import YOLO

# def main():
#     # 1. Inisialisasi model YOLO-cls (menggunakan arsitektur Nano untuk klasifikasi, misal: yolov8n-cls.pt)
#     model = YOLO("yolov8n-cls.pt")

#     # 2. Melatih model (Training)
#     # Ganti 'path/ke/dataset' dengan direktori dataset Anda yang terstruktur untuk klasifikasi
#     # Struktur folder dataset: dataset/train/kelas_A, dataset/train/kelas_B, dst.
#     results = model.train(
#         data="path/ke/dataset", 
#         epochs=10,             # Jumlah epoch (sesuaikan kebutuhan)
#         imgsz=224,             # Ukuran gambar standar untuk klasifikasi
#         batch=16,              # Ukuran batch
#         device="cpu"           # Gunakan "0" jika menggunakan GPU (CUDA)
#     )

#     print("Pelatihan selesai!")

#     # 3. Ekspor model ke format TensorFlow (SavedModel)
#     # Format bisa berupa 'savedmodel' atau 'tfjs'
#     print("Mengekspor model ke format TensorFlow...")
#     tf_model_path = model.export(format="savedmodel")
    
#     print(f"Model berhasil diekspor ke: {tf_model_path}")

# if __name__ == "__main__":
#     main()

model = YOLO("./models/yolo.pt")
tf_model_path = model.export(format="saved_model", imgsz=224)
print(f"Lokasi model: {tf_model_path}")