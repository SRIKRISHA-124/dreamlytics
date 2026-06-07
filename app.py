import streamlit as st
import joblib
import numpy as np
from google import genai


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Dreamlytics AI", page_icon="🌙", layout="wide")

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= GEMINI SETUP =================
from google import genai

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

def gemini_response(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents="You are a sleep health AI assistant. Give short clear advice.\nUser: " + prompt
        )
        return response.text

    except Exception as e:
        return f"AI temporarily unavailable: {e}"

def gemini_response(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents="You are a sleep health AI assistant. Give short clear advice.\nUser: " + prompt
        )
        return response.text

    except Exception as e:
        return f"AI temporarily unavailable: {e}"


# ================= LOAD MODELS =================
model = joblib.load("best_model_decision_tree.pkl")
gender_encoder = joblib.load("Gender_label_encoder.pkl")
occupation_encoder = joblib.load("Occupation_label_encoder.pkl")
bmi_encoder = joblib.load("BMI Category_label_encoder.pkl")
scaler = joblib.load("minmax_scaler_split.pkl")

# ================= FUNCTIONS =================
def recommend_doctor(pred):
    if pred == 0:
        return "🟢 No specialist required. Maintain healthy sleep habits."
    elif pred == 1:
        return "🟡 Sleep Specialist / Psychiatrist recommended."
    elif pred == 2:
        return "🔴 Pulmonologist or Sleep Clinic recommended."
    return "General Physician"

def emergency_check(pred, stress, heart):
    if pred == 2 and stress > 8 and heart > 100:
        return "🚨 EMERGENCY: High risk Sleep Apnea detected!"
    return None

# ================= LOGIN PAGE =================
if not st.session_state.logged_in:

    st.title("🔐 Dreamlytics Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.user = username
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("🌙 Dreamlytics AI")
st.sidebar.success(f"User: {st.session_state.user}")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"

if st.sidebar.button("🔮 Sleep Analysis"):
    st.session_state.page = "analysis"

if st.sidebar.button("🤖 AI Chat"):
    st.session_state.page = "chat"

if st.sidebar.button("🏥 Hospital"):
    st.session_state.page = "hospital"

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ================= HOME =================
if st.session_state.page == "home":

    st.title("🌙 Dreamlytics AI")
    st.subheader("Sleep Disorder Prediction + AI Health System")

    st.success("AI + ML + Gemini Integrated 🤖")

    if st.button("Start Analysis"):
        st.session_state.page = "analysis"

# ================= ANALYSIS =================
elif st.session_state.page == "analysis":

    st.title("🔮 Sleep Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.selectbox("Occupation", ["Doctor","Teacher","Engineer","Nurse","Others"])
        bmi = st.selectbox("BMI", ["Normal","Overweight","Obese"])

    with col2:
        sleep = st.slider("Sleep Duration", 0.0, 12.0, 6.0)
        quality = st.slider("Sleep Quality", 0, 10, 5)
        activity = st.slider("Activity Level", 0, 100, 50)

    with col3:
        stress = st.slider("Stress Level", 0, 10, 5)
        heart = st.number_input("Heart Rate", 75)
        steps = st.number_input("Steps", 5000)
        systolic = st.number_input("Systolic", 120)
        diastolic = st.number_input("Diastolic", 80)

    # encoding
    g = gender_encoder.transform([gender])[0]
    o = occupation_encoder.transform([occupation])[0]
    b = bmi_encoder.transform([bmi])[0]

    numeric = [age, sleep, quality, activity, stress, heart, steps, systolic, diastolic]

    temp = np.zeros((1, 12))
    temp[0, :9] = numeric
    scaled = scaler.transform(temp).flatten()

    features = np.array([
        g, scaled[0], o, scaled[1], scaled[2], scaled[3],
        scaled[4], b, scaled[5], scaled[6], scaled[7], scaled[8]
    ])

    if st.button("Predict"):
        pred = model.predict(features.reshape(1, -1))[0]

        st.subheader("Result")

        if pred == 0:
            st.success("No Disorder")
        elif pred == 1:
            st.error("Insomnia Detected")
        else:
            st.warning("Sleep Apnea Detected")

        # doctor
        st.markdown("## 🩺 Doctor Recommendation")
        st.info(recommend_doctor(pred))

        # emergency
        alert = emergency_check(pred, stress, heart)
        if alert:
            st.error(alert)

# ================= CHATBOT =================
elif st.session_state.page == "chat":

    st.title("🤖 Dreamlytics AI Chatbot")

    user = st.text_input("Ask something about sleep")

    if st.button("Send"):
        reply = gemini_response(user)
        st.session_state.chat.append(("You", user))
        st.session_state.chat.append(("AI", reply))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")

# ================= HOSPITAL =================
elif st.session_state.page == "hospital":

    st.title("🏥 Hospital Finder")

    st.link_button(
        "📍 Open Nearby Sleep Clinics",
        "https://www.google.com/maps/search/sleep+clinic+near+me"
    )

    st.subheader("📅 Appointment")

    name = st.text_input("Name")
    phone = st.text_input("Phone")
    date = st.date_input("Date")

    if st.button("Book"):
        st.success("Appointment Requested (Demo)")