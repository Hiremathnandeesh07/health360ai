from flask import Flask, request, render_template, jsonify, send_from_directory, session
import numpy as np
import pandas as pd
import pickle
from flask import redirect, url_for
import time
from flask_mail import Mail, Message
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
    system_instruction="You are an experienced doctor who provides information about diseases, symptoms, precautions, diets, and workouts to cure diseases. Always respond without using bold letters. (Remember: whatever you are answering is printed in a continuous line without any paragraphs or indentations or any exclamations.)",
)


# flask app
app = Flask(__name__,static_folder='static')
# Serving the datasets directory
@app.route('/datasets/<filename>')
def serve_dataset(filename):
    return send_from_directory('datasets', filename)

# Serving the images directory
@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)


# Global history for chatbot
history = [
    {"role": "user", "parts": ["hi'\n"]},
    {"role": "model", "parts": ["Hello. How can I help you?\n"]},
]


# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
doctor=pd.read_csv("datasets/doctor.csv")

# load model===========================================
svc = pickle.load(open('models/svc.pkl','rb'))


#============================================================
# custome and helping functions
#==========================helper funtions================
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     desc = " ".join([w for w in desc])
#
#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = [col for col in pre.values]
#
#     med = medications[medications['Disease'] == dis]['Medication']
#     med = [med for med in med.values]
#
#     die = diets[diets['Disease'] == dis]['Diet']
#     die = [die for die in die.values]
#
#     wrkout = workout[workout['disease'] == dis] ['workout']
#
#
#     return desc,pre,med,die,wrkout
def helper(dis):
    # Retrieve the description for the disease
    desc = description[description['Disease'].str.strip().str.lower() == dis.strip().lower()]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else "No description found for this disease."

    # Retrieve the precautions
    pre = precautions[precautions['Disease'].str.strip().str.lower() == dis.strip().lower()][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values] if not pre.empty else ["No precautions found for this disease."]

    # Retrieve the medications
    med = medications[medications['Disease'].str.strip().str.lower() == dis.strip().lower()]['Medication']
    med = [med for med in med.values] if not med.empty else ["No medications found for this disease."]

    # Retrieve the diet recommendations
    die = diets[diets['Disease'].str.strip().str.lower() == dis.strip().lower()]['Diet']
    die = [die for die in die.values] if not die.empty else ["No diet recommendations found for this disease."]

    # Retrieve workout recommendations
    wrkout = workout[workout['disease'].str.strip().str.lower() == dis.strip().lower()]['workout']
    wrkout = [wrk for wrk in wrkout.values] if not wrkout.empty else ["No workout found for this disease."]

    # Retrieve doctor information
    doc_info = doctor[doctor['Disease'].str.strip().str.lower() == dis.strip().lower()][['HospitalName', 'DoctorName', 'DoctorName', 'MeetingTiming', 'AssistantPhoneNumber']]
    doc_info = doc_info.to_dict(orient='records') if not doc_info.empty else [{"HospitalName": "N/A", "DoctorName": "N/A",  "MeetingTiming": "N/A", "AssistantPhoneNumber": "N/A"}]

    return desc, pre, med, die, wrkout, doc_info

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]




# creating routes========================================


@app.route("/")
def index():
    return render_template("newindex3.html")


# Chatbot route
@app.route("/chat", methods=["POST"])
def chat():
    global history
    user_input = request.json.get("message")

    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(user_input)

    model_response = response.text

    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [model_response]})

    return jsonify({"response": model_response})








app.secret_key = 'your_secret_key'

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('newindex3.html', message=message)
        else:
            # Store the symptoms in the session
            session['symptoms'] = symptoms
            # Render the loading page
            return render_template('loading.html')

    return render_template('results.html')

@app.route('/results')
def results():
    symptoms = session.get('symptoms')
    if not symptoms:
        return redirect(url_for('home'))

    # Simulate loading time
    time.sleep(2)

    # Split the user's input into a list of symptoms (assuming they are comma-separated)
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    # Remove any extra characters, if any
    user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
    predicted_disease = get_predicted_value(user_symptoms)
    print(predicted_disease)
    dis_des, precautions, medications, rec_diet, workout, doc_info = helper(predicted_disease)

    my_precautions = []
    for i in precautions[0]:
        my_precautions.append(i)

    return render_template('results.html', predicted_disease=predicted_disease, symptoms=symptoms, dis_des=dis_des,
                           my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                           workout=workout, doc_info=doc_info)








# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/firstaid')
def firstaid():
    return render_template("firstaid.html")
@app.route('/walkthrough')
def walkthrough():
    return render_template("walkthrough.html")
# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("chatbot.html")

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Use your email provider's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'nandeeshh07@gmail.com'  # Your email address
app.config['MAIL_PASSWORD'] = 'eilw pvmz jmjn fyby'  # Your email password (use app password if 2FA is enabled)
app.config['MAIL_DEFAULT_SENDER'] = 'nandeeshh07@gmail.com'  # Default sender

mail = Mail(app)

@app.route('/sendemail', methods=['POST'])
def send_email():
    # Get all form data
    hospital_name = request.form.get('hospitalName')
    doctor_name = request.form.get('doctorName')
    meeting_timing = request.form.get('meetingTiming')
    assistant_phone = request.form.get('assistantPhone')
    patient_name = request.form.get('patientName')
    patient_contact = request.form.get('patientContact')
    patient_email = request.form.get('patientEmail')  # New field

    # Create the email content for the hospital
    hospital_email_body = f"""
    You have received a new appointment booking.

    Hospital Name: {hospital_name}
    Doctor Name: {doctor_name}
    Meeting Timing: {meeting_timing}
    Assistant Phone: {assistant_phone}
    Patient Name: {patient_name}
    Patient Contact: {patient_contact}
    Patient Email: {patient_email}
    """

    # Create the email content for the patient
    patient_email_body = f"""
    Dear {patient_name},

    Your appointment has been successfully booked.

    Hospital Name: {hospital_name}
    Doctor Name: {doctor_name}
    Meeting Timing: {meeting_timing}
    Assistant Phone: {assistant_phone}

    If you have any questions, feel free to contact us at {assistant_phone}.

    Best regards,
    {hospital_name} Team
    """

    try:
        # Send email to the hospital
        hospital_msg = Message(
            subject="New Appointment Booking",  # Email subject
            recipients=["hiremathnandeesh06@gmail.com"],  # Replace with the hospital's email
            body=hospital_email_body
        )
        mail.send(hospital_msg)

        # Send confirmation email to the patient
        patient_msg = Message(
            subject="Appointment Confirmation",  # Email subject
            recipients=[patient_email],  # Patient's email address
            body=patient_email_body
        )
        mail.send(patient_msg)

        return jsonify({"message": "Appointment booked successfully! Emails sent."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


















