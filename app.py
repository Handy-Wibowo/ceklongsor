import numpy as np
import pandas as pd
import pickle
import random
from flask import Flask, render_template, request, redirect, url_for, flash
import re
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from lime.lime_tabular import LimeTabularExplainer
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)



#app.secret_key = 'your_secret_key_here'  # Ganti dengan secret key yang aman

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User store sederhana (demo)
users = {
    'guest': {'password': 'guestpass', 'role': 'guest'},
    'premiumuser': {'password': 'premiumpass', 'role': 'premium'}
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.role = users[username]['role']

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

feature_order = [
    'Rainfall_mm',
    'Slope_Angle',
    'Soil_Saturation',
    'Vegetation_Cover',
    'Earthquake_Activity',
    'Proximity_to_Water',
    'Soil_Type_Gravel',
    'Soil_Type_Sand',
    'Soil_Type_Silt'
]

# Load model dan scaler
with open('svm_linear_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('gradient_boosting_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open('regression_logistics_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data latih untuk background LIME
df_train = pd.read_csv('landslide_dataset.csv', encoding='latin1')
X_train_unscaled = df_train[feature_order].values
background_data = X_train_unscaled[:100]

explainer = LimeTabularExplainer(
    training_data=background_data, # LIME harus diinisialisasi dengan data mentah (unscaled)
    feature_names=feature_order,
    class_names=['Ga Longsor', 'Longsor'],
    discretize_continuous=True
)

def interpret_impact(value):
    abs_val = abs(value)
    if abs_val < 0.005:
        strength = "sedikit"
    elif abs_val < 0.02:  # Menurunkan ambang batas untuk 'cukup'
        strength = "cukup"
    else:
        strength = "sangat" # Sekarang, kontribusi >= 0.04 akan dianggap 'sangat'
    direction = "meningkatkan" if value > 0 else "menurunkan"
    return f"{strength.capitalize()} {direction} kemungkinan terjadinya tanah longsor"

def prettify_feature_string(feature_string):
    """Mengubah string fitur LIME menjadi format yang mudah dibaca manusia dalam Bahasa Indonesia."""
    feature_names_map = {
        'Rainfall_mm': 'Curah Hujan (mm)',
        'Slope_Angle': 'Sudut Kemiringan (derajat)',
        'Soil_Saturation': 'Saturasi Tanah',
        'Vegetation_Cover': 'Tingkat Vegetasi',
        'Earthquake_Activity': 'Aktivitas Gempa',
        'Proximity_to_Water': 'Jarak ke Air',
    }
    soil_type_map = {
        'Gravel': 'Kerikil',
        'Sand': 'Pasir',
        'Silt': 'Lanau'
    }
    ops_map = {
        '<=': 'kurang dari atau sama dengan',
        '>=': 'lebih dari atau sama dengan',
        '<': 'kurang dari',
        '>': 'lebih dari',
        '=': 'adalah' # Untuk kasus perbandingan eksak, meskipun LIME jarang menggunakannya
    }

    try:
        # 1. Menangani Jenis Tanah (one-hot encoded)
        # Contoh: 'Soil_Type_Sand > 0.00' atau 'Soil_Type_Gravel <= 0.00'
        soil_type_match = re.match(r'Soil_Type_([a-zA-Z]+)\s*(<=|>)\s*0\.00', feature_string)
        if soil_type_match:
            soil_eng = soil_type_match.group(1)
            op = soil_type_match.group(2)
            soil_id = soil_type_map.get(soil_eng, soil_eng) # Terjemahkan
            if op == '>':
                return f"Jenis tanah adalah {soil_id}"
            elif op == '<=':
                return f"Jenis tanah BUKAN {soil_id}"

        # Menangani kasus di mana LIME hanya mengembalikan 'Soil_Type_X' jika nilainya 1
        if feature_string.startswith('Soil_Type_'):
            soil_eng = feature_string.replace('Soil_Type_', '')
            if soil_eng in soil_type_map:
                return f"Jenis tanah adalah {soil_type_map[soil_eng]}"

        # 2. Menangani Kondisi Rentang (misalnya, "0.50 < Soil_Saturation <= 1.00")
        # Regex ini menangkap: (nilai1) (op1) (nama_fitur) (op2) (nilai2)
        range_match = re.match(r'(\d+\.?\d*)\s*(<|<=|>=|>)\s*([a-zA-Z_]+)\s*(<|<=|>=|>)\s*(\d+\.?\d*)', feature_string)
        if range_match:
            val1, op1, feature_eng, op2, val2 = range_match.groups()

            # Jika fitur adalah jenis tanah, sederhanakan outputnya
            if feature_eng.startswith('Soil_Type_'):
                soil_name_eng = feature_eng.replace('Soil_Type_', '')
                soil_name_id = soil_type_map.get(soil_name_eng, soil_name_eng)
                # Rentang untuk fitur one-hot (misalnya, 0.5 < Soil_Type_Silt <= 1.0)
                # berarti fitur tersebut aktif (bernilai 1).
                return f"Jenis tanah adalah {soil_name_id}"

            readable_feature = feature_names_map.get(feature_eng, feature_eng.replace('_', ' '))

            # Sederhanakan rentang umum menjadi "antara X dan Y"
            if (op1 == '<' or op1 == '<=') and (op2 == '<' or op2 == '<='):
                 return f"{readable_feature} antara {val1} dan {val2}"
            else: # Fallback untuk operasi yang lebih kompleks dalam rentang
                return f"{readable_feature} {ops_map.get(op1, op1)} {val1} dan {ops_map.get(op2, op2)} {val2}"

        # 3. Menangani Kondisi Perbandingan Tunggal (misalnya, "Rainfall_mm <= 175.44")
        parts = re.split(r'(<=|>=|<|>)', feature_string) # Memisahkan berdasarkan operator
        if len(parts) == 3: # Jika berhasil dipisah menjadi fitur, operator, dan nilai
            feature_eng, op, value = [p.strip() for p in parts]
            readable_feature = feature_names_map.get(feature_eng, feature_eng.replace('_', ' '))
            return f"{readable_feature} {ops_map.get(op, op)} {value}"

        # 4. Fallback untuk string lainnya (misalnya, hanya nama fitur atau perbandingan eksak)
        exact_match = re.match(r'([a-zA-Z_]+)=(.*)', feature_string)
        if exact_match:
            feature_eng, value = exact_match.groups()
            readable_feature = feature_names_map.get(feature_eng, feature_eng.replace('_', ' '))
            return f"{readable_feature} adalah {value}"

        # Fallback terakhir: ganti underscore dan gunakan map jika ada
        return feature_names_map.get(feature_string, feature_string.replace('_', ' '))

    except Exception:
        # Jika parsing gagal, kembalikan string asli
        return feature_string.replace('_', ' ')


def generate_random_inputs():
    """Generates random input values for the prediction form based on defined ranges."""
    inputs = {}
    # Ranges from index.html input fields
    inputs['Rainfall_mm'] = round(random.uniform(50.0361507028, 299.919102159), 2)
    inputs['Slope_Angle'] = round(random.uniform(5.0039436145, 59.9667324336), 2)
    inputs['Soil_Saturation'] = round(random.uniform(0.0006522119, 0.9988312628), 4)
    inputs['Vegetation_Cover'] = round(random.uniform(0.1000046539, 0.9998366026), 4)
    inputs['Earthquake_Activity'] = round(random.uniform(0.0016411578, 6.4986697355), 2)
    inputs['Proximity_to_Water'] = round(random.uniform(0.0006533908, 1.9996356254), 4)

    # One-hot encoding for soil types: randomly pick one to be 1, others 0
    soil_types = ['Soil_Type_Gravel', 'Soil_Type_Sand', 'Soil_Type_Silt']
    chosen_soil = random.choice(soil_types)
    for st in soil_types:
        inputs[st] = 1 if st == chosen_soil else 0
    
    return inputs

def predict_and_explain_with_lime(model, input_data, use_scaler=False):
    """
    Menghasilkan prediksi dan penjelasan LIME untuk sebuah model.
    Menangani scaling secara internal berdasarkan flag 'use_scaler'.
    """
    # 1. Siapkan data dan fungsi prediksi berdasarkan apakah scaling diperlukan.
    if use_scaler:
        # Untuk model yang memerlukan data yang di-scale (misalnya, SVM, Regresi Logistik)
        prediction_data = scaler.transform(input_data)
        
        # predict_fn untuk LIME juga harus men-scale data perturbasi yang diterimanya.
        def predict_fn_for_lime(x):
            x_scaled = scaler.transform(x)
            return model.predict_proba(x_scaled)
    else:
        # Untuk model yang bekerja pada data mentah (misalnya, model berbasis tree)
        prediction_data = input_data
        predict_fn_for_lime = model.predict_proba

    # 2. Dapatkan prediksi model pada input yang (mungkin) telah di-scale.
    proba = model.predict_proba(prediction_data)[0, 1]
    label = 'Berpotensi Longsor' if proba >= 0.5 else 'Berpotensi Tidak Longsor'

    # 3. Hasilkan penjelasan LIME.
    # Explainer LIME diinisialisasi dengan data unscaled, jadi ia memerlukan
    # instance unscaled untuk dijelaskan. predict_fn_for_lime menangani scaling yang diperlukan.
    exp = explainer.explain_instance(
        input_data[0],          # Berikan instance mentah (unscaled)
        predict_fn_for_lime,    # Gunakan fungsi prediksi yang sesuai
        num_features=len(feature_order)
    )
    
    feature_impacts = exp.as_list()
    feature_impacts_sorted = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
    interpreted = [
        {
            'feature': prettify_feature_string(feat),
            'contribution': val,
            'interpretation': interpret_impact(val)
        }
        for feat, val in feature_impacts_sorted
    ]
    return label, proba, interpreted

@app.route('/')
@app.route('/index')
def index():
    initial_inputs = generate_random_inputs()
    return render_template('index.html', submitted_inputs=initial_inputs)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('Logged in successfully.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/result', methods=['POST'])
def result():
    to_predict_dict = request.form.to_dict()
    input_list = []
    for feature in feature_order:
        val = to_predict_dict.get(feature)
        if val is None:
            return f"Missing input for {feature}", 400
        try:
            input_list.append(float(val))
        except ValueError:
            return f"Invalid input for {feature}: must be a number", 400

    X = np.array(input_list).reshape(1, -1)

    results = {}

    # Model yang MENGGUNAKAN scaler
    label, proba, impacts = predict_and_explain_with_lime(svm_model, X, use_scaler=True)
    results['SVM'] = {'label': label, 'proba': proba, 'feature_impacts': impacts}

    label, proba, impacts = predict_and_explain_with_lime(lr_model, X, use_scaler=True)
    results['Logistic Regression'] = {'label': label, 'proba': proba, 'feature_impacts': impacts}

    # Model yang TIDAK menggunakan scaler
    label, proba, impacts = predict_and_explain_with_lime(rf_model, X, use_scaler=True)
    results['Random Forest'] = {'label': label, 'proba': proba, 'feature_impacts': impacts}

    label, proba, impacts = predict_and_explain_with_lime(gb_model, X, use_scaler=True)
    results['Gradient Boosting'] = {'label': label, 'proba': proba, 'feature_impacts': impacts}

    # Batasi hasil sesuai status login dan role user
    show_interpretation_guide = False # Default untuk pengguna gratis/tamu
    if not current_user.is_authenticated or current_user.role == 'guest':
        svm_result = results['SVM']
        limited_results = {
            'SVM': {
                'label': svm_result['label'],
                'proba': None,
                'feature_impacts': None
            }
        }
        return render_template('index.html', results=limited_results, user_role='guest', submitted_inputs=to_predict_dict, show_interpretation_guide=show_interpretation_guide)

    elif current_user.role == 'premium':
        show_interpretation_guide = True # Untuk pengguna premium
        return render_template('index.html', results=results, user_role='premium', submitted_inputs=to_predict_dict, show_interpretation_guide=show_interpretation_guide)
    else:
        # Fallback for any other user role, treat as guest
        svm_result = results['SVM']
        limited_results = {
            'SVM': {
                'label': svm_result['label'],
                'proba': None,
                'feature_impacts': None
            }
        }
        return render_template('index.html', results=limited_results, user_role='guest', submitted_inputs=to_predict_dict, show_interpretation_guide=False)


if __name__ == "__main__":
    app.run(debug=True)
