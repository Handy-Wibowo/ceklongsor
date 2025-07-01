# CekLongsor

web app untuk prediksi longsor menggunakan Machine learning Model SVM, Gradient Boosting, Random Forest, dan Logistics Regression
menggunakan Flask dan HTML dan CSS Bootstrap framework

## Prasyarat

Pastikan Anda telah menginstal Python, pip dan virtualenv di sistem Anda. Proyek ini direkomendasikan untuk dijalankan dengan Python 3.8 atau yang lebih baru.

- [pip] (https://pypi.org/project/pip/)
- [Python 3.8+](https://www.python.org/downloads/)
- [virtualenv] (https://virtualenv.pypa.io/en/latest/) pip install virtualenv

## Instalasi

Ikuti langkah-langkah ini untuk menyiapkan lingkungan pengembangan lokal Anda.

1.  **Clone repository** (Jika proyek Anda ada di Git)

    ```bash
    # Ganti dengan URL repository Anda jika ada
    git clone https://github.com/Handy-Wibowo/ceklongsor.git
    cd "ceklongsor"
    ```

2.  **Buat dan aktifkan Virtual Environment**

    Sangat disarankan untuk menggunakan _virtual environment_ untuk mengisolasi dependensi proyek.

    jika belum buat gunakan command "virtualenv venv" di direktori pilihan
    ```bash
    virtualenv venv
    ```
    jika sudah buat, ketik command di bawah untuk mengaktifkan virtualenv
    perlu di note bahwa command ini harus dimasukkan ketika membuka project
    - Di Windows:

      ```bash
      python -m venv .venv
      .\.venv\Scripts\activate
      ```

    - Di macOS/Linux:
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

3.  **Instal dependensi**

    Instal semua paket yang diperlukan yang tercantum dalam file `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    > **Catatan:** Jika file `requirements.txt` tidak ada, Anda dapat membuatnya dari lingkungan yang sudah ada (seperti yang Anda berikan dalam konteks) dengan menjalankan perintah `pip freeze > requirements.txt` di dalam virtual environment yang aktif.

## Menjalankan Proyek

Untuk menjalankan proyek, eksekusi skrip utama dari terminal.

```bash
# Ganti 'main.py' dengan nama file entry point proyek Anda jika berbeda (misal: app.py)
python app.py
```
