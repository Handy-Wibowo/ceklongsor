import os
import sys

# Ganti 'main' dengan nama file Python utama Anda (misal: app.py, run.py)
# tanpa ekstensi .py.
from main import app as application

# Menambahkan direktori proyek ke path Python
# Ini membantu server menemukan modul-modul Anda.
project_directory = os.path.dirname(__file__)
sys.path.insert(0, project_directory)