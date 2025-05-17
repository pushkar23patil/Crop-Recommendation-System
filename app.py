from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

# Load models
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle JSON from ESP32
        if request.is_json:
            data = request.get_json()
            N = float(data['Nitrogen'])
            P = float(data['Phosphorous'])
            K = float(data['Potassium'])
            temp = float(data['Temperature'])
            soil_moisture = float(data['soil_moisture'])
        else:
            # Handle form data from HTML
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorous'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            soil_moisture = float(request.form['soil_moisture'])

        feature_list = [N, P, K, temp, soil_moisture]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)
        crop = crop_dict.get(prediction[0], "Unknown")

        if request.is_json:
            return jsonify({"recommended_crop": crop})
        else:
            result = f"{crop} is the best crop to be cultivated right there"
            return render_template("index.html", result=result)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        else:
            return render_template("index.html", result="Error: " + str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
