Program ini menggunakan OpenCV dan MediaPipe untuk mendeteksi gerakan tangan yang membentuk sinyal darurat SOS. Proses kerja utama program:

Deteksi Landmark Tangan:
Menggunakan MediaPipe Hands untuk melacak posisi jari dan telapak tangan.
Pengenalan Gerakan SOS:
Program memantau urutan tiga tahap gerakan tangan:
Tahap 1: Semua jari terbuka.
Tahap 2: Jari kelingking ditekuk.
Tahap 3: Jari lain menutup ke atas kelingking.
Jika urutan ini terdeteksi, program mengidentifikasi sinyal sebagai "SOS DETECTION".
Penyimpanan Bukti Gambar:
Saat sinyal SOS terdeteksi, program otomatis mengambil screenshot dan menyimpannya sebagai file gambar.
Menampilkan Video dengan Deteksi Real-Time:
Video dari webcam ditampilkan dengan overlay landmark tangan dan teks status SOS.
