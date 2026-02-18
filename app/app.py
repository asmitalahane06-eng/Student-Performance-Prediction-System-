from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'student_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect all inputs from the form
    # Ensure the order matches how you trained your model
    input_features = [float(x) for x in request.form.values()]
    final_features = np.array([input_features])

    prediction = model.predict(final_features)[0]
    score = round(prediction, 2)

    # Dynamic Results Logic
    if score >= 75:
        status = "Congratulations"
        style = "high-score"
        msg = "Excellent performance! Keep up the great work."
        img = "https://cdn-icons-png.flaticon.com/512/3112/3112946.png"

    elif score >= 40:
        status = "Good Job"
        style = "mid-score"
        msg = "You are on the right track. Keep practicing!"
        img = "https://cdn-icons-png.flaticon.com/512/2436/2436632.png"

    else:
        status = "Hard Work Needed"
        style = "low-score"
        msg = "Don't lose hope! Focus more on your weak subjects."
        img = "https://cdn-icons-png.flaticon.com/512/10433/10433048.png"

    return render_template(
        'result.html',
        score=score,
        status=status,
        message=msg,
        style_class=style,
        image_url=img
    )


if __name__ == "__main__":
    app.run(debug=True)
