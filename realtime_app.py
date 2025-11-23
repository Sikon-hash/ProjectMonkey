import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import os
import pygame
pygame.mixer.init()

#model
model = joblib.load("pose_classifier.pkl")

#mediapipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

#suara
def play_sound(sound_file):
    try:
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    except:
        print(f"Tidak bisa mainkan: {sound_file}")

#sedang berfikir
img_path = "monkey_thinking.png"
if not os.path.exists(img_path):
    img_path = "monkey_thinking.jpg"

if os.path.exists(img_path):
    monkey_img = Image.open(img_path)
    monkey_img = monkey_img.resize((600, 600))
else:
    print("ERROR: Tidak ada file 'monkey_thinking.png' atau 'monkey_thinking.jpg'!")
    exit()

#apa lo liat liat
idle_img_path = "monkey_idle.png"
if not os.path.exists(idle_img_path):
    idle_img_path = "monkey_idle.jpg"

if os.path.exists(idle_img_path):
    monkey_idle_img = Image.open(idle_img_path)
    monkey_idle_img = monkey_idle_img.resize((600, 600))
else:
    print("ERROR: Tidak ada file 'monkey_idle.png' atau 'monkey_idle.jpg'!")
    exit()

#aku punya ide
idea_img_path = "monkey_idea.png"
if not os.path.exists(idea_img_path):
    idea_img_path = "monkey_idea.jpg"

if os.path.exists(idea_img_path):
    monkey_idea_img = Image.open(idea_img_path)
    monkey_idea_img = monkey_idea_img.resize((600, 600))
else:
    print("ERROR: Tidak ada file 'monkey_idea.png' atau 'monkey_idea.jpg'!")
    exit()

#ba kaget
shocked_img_path = "monkey_shocked.png"
if not os.path.exists(shocked_img_path):
    shocked_img_path = "monkey_shocked.jpg"

if os.path.exists(shocked_img_path):
    monkey_shocked_img = Image.open(shocked_img_path)
    monkey_shocked_img = monkey_shocked_img.resize((600, 600))
else:
    print("ERROR: Tidak ada file 'monkey_shocked.png' atau 'monkey_shocked.jpg'!")
    exit()

#jokowi
jokowi_img_path = "hidup_jokowi.png"
if not os.path.exists(jokowi_img_path):
    jokowi_img_path = "hidup_jokowi.jpg"

if os.path.exists(jokowi_img_path):
    hidup_jokowi_img = Image.open(jokowi_img_path)
    hidup_jokowi_img = hidup_jokowi_img.resize((600,600))
else:
    print("No Jokowi")

#monekey F
# monkey_fuck_img_path = "monkey_fuck.png"
# if not os.path.exists(monkey_fuck_img_path):
#     monkey_fuck_img_path = "monkey_fuck.jpg"

# if os.path.exists(monkey_fuck_img_path):
#     monkey_fuck_img = Image.open(monkey_fuck_img_path)
#     monkey_fuck_img = monkey_fuck_img.resize((400,400))
# else:
#     print("no Monkey")

#tkinter
root = tk.Tk()
root.title("Monkey Pose Detector") 
root.geometry("1920x1080")

#gradien background
canvas = tk.Canvas(root, width=1280, height=1080)
canvas.pack(fill="both", expand=True)

# gradien verti
def create_gradient(canvas, color1, color2, width, height):
    r1, g1, b1 = root.winfo_rgb(color1)
    r2, g2, b2 = root.winfo_rgb(color2)

    def rgb(r, g, b):
        return f'#{int(r):04x}{int(g):04x}{int(b):04x}'

    for i in range(height):
        r = r1 + (r2 - r1) * i // height
        g = g1 + (g2 - g1) * i // height
        b = b1 + (b2 - b1) * i // height
        canvas.create_line(0, i, width, i, fill=rgb(r >> 8, g >> 8, b >> 8))

#biru ke ungu
# create_gradient(canvas, "#000033", "#330066", 800, 600)

# Frame utama di atas canvas
frame_main = tk.Frame(canvas, bg="#000033") 
canvas.create_window((0, 0), window=frame_main, anchor="nw")

# Label untuk kamera (dengan background transparan)
label_camera = tk.Label(frame_main, bg="#000033")
label_camera.pack(side="left", padx=10, pady=10)

# Label untuk gambar monyet (dengan background transparan)
label_monkey = tk.Label(frame_main, bg="#000033")
label_monkey.pack(side="right", padx=10, pady=10)

# Status label (dengan background dan teks yang terlihat)
status_label = tk.Label(root, text="Tunggu... Deteksi pose", font=("Arial", 14), bg="#000033", fg="white")
status_label.place(x=400, y=550, anchor="center")

cap = cv2.VideoCapture(0)

def extract_features_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if not results.pose_landmarks:
        return None
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.visibility])
    return np.array(landmarks).flatten()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Deteksi pose
    features = extract_features_from_frame(frame)
    if features is not None:
        # Prediksi
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0].max()

        # Tampilkan landmark pose dan tangan di frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)

        # Gambar landmark pose
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  
            )

        # Gambar landmark tangan (jari)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),  
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2) 
                )

        # Tampilkan hasil prediksi
        if prediction == 1:
            status_label.config(text=f"💡 POSE BERPIKIR! (Conf: {confidence:.2f})", fg="green")
            photo = ImageTk.PhotoImage(monkey_img)
            label_monkey.config(image=photo)
            label_monkey.image = photo
        elif prediction == 2:
            status_label.config(text=f"😴 IDLE - Diam saja (Conf: {confidence:.2f})", fg="orange")
            photo = ImageTk.PhotoImage(monkey_idle_img)
            label_monkey.config(image=photo)
            label_monkey.image = photo
        elif prediction == 3:
            status_label.config(text=f"💡 IDEAAA! (Conf: {confidence:.2f})", fg="purple")
            photo = ImageTk.PhotoImage(monkey_idea_img)
            label_monkey.config(image=photo)
            label_monkey.image = photo
        elif prediction == 4:
            status_label.config(text=f"😱 SHOCKED! (Conf: {confidence:.2f})", fg="red")
            photo = ImageTk.PhotoImage(monkey_shocked_img)
            label_monkey.config(image=photo)
            label_monkey.image = photo
        elif prediction == 5:
            status_label.config(text=f"Hidup Jokowi✊(Conf: {confidence:.2f})", fg="red")
            photo = ImageTk.PhotoImage(hidup_jokowi_img)
            play_sound("audio/hidup_jokowi.mp3")
            label_monkey.config(image=photo)
            label_monkey.image = photo
        # elif prediction == 6:
        #    status_label.config(text=f"Fuck u(Conf: {confidence:.2f})", fg="red")
        #    photo = ImageTk.PhotoImage(monkey_fuck_img)
        #    label_monkey.config(image=photo)
        #    label_monkey.image = photo
        else:
            status_label.config(text=f"🔍 Bukan thinking/idle/idea/shocked (Conf: {confidence:.2f})", fg="blue")
            label_monkey.config(image="")
    else:
        status_label.config(text="❌ Pose tidak terdeteksi", fg="red")

    # Tampilkan frame kamera
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label_camera.config(image=imgtk)
    label_camera.image = imgtk

    root.after(30, update_frame)

update_frame()
root.mainloop()