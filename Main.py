import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from streamlit_option_menu import option_menu

# 1) Page config
st.set_page_config(
    page_title="Real-Time Face Emotion Detector", 
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="auto"
)

# 2) Hide default Streamlit header/footer
st.markdown("""
    <style>
      #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3) Your custom CSS (cards, title bar, footer)
st.markdown("""
    <style>
      .title {
        font-size: 2.8rem; font-weight: bold;
        color: white; text-align: center;
        padding: 1rem; background: linear-gradient(90deg,#2C3E50,#4CA1AF);
        border-radius: .5rem; margin-bottom: 1rem;
      }
      .card {
        background: white; border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 1.5rem; margin-bottom: 1rem;
      }
      .footer { text-align: center; font-size: .9rem; color: #888; margin-top: 2rem; }
    </style>
""", unsafe_allow_html=True)




# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result

def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15 # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame

def facesentiment(stop_button):
    # st.title("Real-Time Facial Analysis with Streamlit")
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    stframe = st.image([])  # Placeholder for the webcam feed

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Analyze the frame using DeepFace
        result = analyze_frame(frame)

        # Extract the face coordinates
        face_coordinates = result[0]["region"]
        x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{result[0]['dominant_emotion']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Convert the BGR frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Overlay white rectangle with text on the frame
        # texts = [
        #     f"Age: {result[0]['age']}",
        #     f"Face Confidence: {round(result[0]['face_confidence'],3)}",
        #     # f"Gender: {result[0]['dominant_gender']} {result[0]['gender'][result[0]['dominant_gender']]}",
        #     f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
        #     f"Race: {result[0]['dominant_race']}",
        #     f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
        # ]
        texts = [
            f"Face Confidence: {round(result[0]['face_confidence'],3)}",
            f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
        ]


        frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)

        # Display the frame in Streamlit
        stframe.image(frame_with_overlay, channels="RGB")
        if stop_button:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Face Analysis Application #
    # st.title("Real Time Face Emotion Detection Application")
    activities = ["Webcam Face Detection", "About"]
    with st.sidebar:
        st.markdown("## 🤖 Face Emotion App")
        choice = option_menu(
            menu_title=None,
            options=["Home", "Real-Time Detection", "About"],
            icons=["house","camera-video","info-circle"],
            default_index=0,
            styles={
                "nav-link": {"font-size":"1.1rem","margin":"0.5rem 0"},
                "nav-link-selected": {"background-color":"#4CA1AF","color":"#FFF"}
            }
        )
        st.markdown("---")
        st.markdown("<div class='card'><h5>🔧 Developed by Rohan Garad</h5><p>Thanks for visiting!</p></div>", unsafe_allow_html=True)
    
    if choice == "Home":
        st.markdown("<div class='title'>Welcome to Real‑Time Face Emotion Detector</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><p>Use your webcam to detect emotions live.</p></div>", unsafe_allow_html=True)

        # Project overview
        st.markdown("""
            <div class='card'>
                <h4>📝 Project Overview</h4>
                <p>This app captures your webcam stream, detects faces using OpenCV, and analyzes emotions with DeepFace—all in real time!</p>
            </div>
        """, unsafe_allow_html=True)

        # How it works
        st.markdown("""
            <div class='card'>
                <h4>⚙️ How It Works</h4>
                <ol>
                    <li>🖥️ OpenCV grabs frames from your webcam.</li>
                    <li>🤖 DeepFace processes each frame for emotion predictions.</li>
                    <li>📊 Results are displayed on-screen and can be plotted live.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        # Quick start
        st.markdown("""
            <div class='card'>
                <h4>🚀 Quick Start</h4>
                <ul>
                    <li>1. Allow camera access when prompted.</li>
                    <li>2. Click “Start” to begin detection.</li>
                    <li>3. Watch your emotions update live on screen.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Tech stack
        st.markdown("""
            <div class='card'>
                <h4>🛠️ Tech Stack</h4>
                <p>Streamlit • OpenCV • DeepFace • Python 3.8+</p>
            </div>
        """, unsafe_allow_html=True)


    elif choice == "Real-Time Detection":
        st.markdown("<div class='title'>Webcam Face Detection</div>", unsafe_allow_html=True)
        run = st.button("▶️ Start")
        stop = st.button("⏹️ Stop")
        if run:
            facesentiment(stop_button=stop)   # call your existing function

    elif choice == "About":
        st.markdown("<div class='title'>About This App</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><p>This app uses OpenCV, DeepFace & Streamlit.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Built with ❤️ by Rohan Garad</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()