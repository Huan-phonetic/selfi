import cv2
import dlib
import numpy as np
import os
import time
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# === GUI ===
def select_reference_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        reference_path.set(path)

def select_input_folder():
    path = filedialog.askdirectory()
    if path:
        input_folder.set(path)

def select_output_folder():
    path = filedialog.askdirectory()
    if path:
        output_folder.set(path)

def run_alignment_thread():
    thread = threading.Thread(target=run_alignment)
    thread.start()

def run_alignment():
    ref_path = reference_path.get()
    in_path = input_folder.get()
    out_path = output_folder.get()

    if not (ref_path and in_path and out_path):
        messagebox.showerror("错误", "请确保所有路径都已选择。")
        return

    try:
        progress_label.config(text="处理中，请稍候...")
        process_faces(ref_path, in_path, out_path)
        progress_label.config(text="完成！")
        messagebox.showinfo("完成", "人脸对齐完成！")
    except Exception as e:
        progress_label.config(text="")
        messagebox.showerror("出错了", str(e))
    finally:
        progress_bar.stop()

# === 核心逻辑 ===
def process_faces(reference_img_path, input_folder, output_folder):
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    predictor_path = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
    face_rec_model_path = os.path.join(model_dir, "dlib_face_recognition_resnet_model_v1.dat")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    def get_landmarks(image, face):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return predictor(gray, face)

    def get_face_descriptor(image, face):
        landmarks = get_landmarks(image, face)
        return np.array(face_rec_model.compute_face_descriptor(image, landmarks))

    def align_image(image, landmarks_points, output_size=(900, 1600), face_ratio=0.2):
        left_eye_pts = np.array(landmarks_points[36:42])
        right_eye_pts = np.array(landmarks_points[42:48])
        left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eye_dist = np.linalg.norm(right_eye_center - left_eye_center)
        desired_eye_dist = face_ratio * output_size[1]
        scale = desired_eye_dist / eye_dist
        eyes_center = (
            int((left_eye_center[0] + right_eye_center[0]) / 2),
            int((left_eye_center[1] + right_eye_center[1]) / 2)
        )
        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
        M[0, 2] += (output_size[0] / 2 - eyes_center[0])
        M[1, 2] += (output_size[1] / 2 - eyes_center[1])
        return cv2.warpAffine(image, M, output_size, flags=cv2.INTER_CUBIC)

    ref_img = cv2.imread(reference_img_path)
    ref_faces = detector(ref_img)
    if not ref_faces:
        raise ValueError("参考图像中没有检测到人脸。")
    ref_descriptor = get_face_descriptor(ref_img, ref_faces[0])

    date_pattern = re.compile(r"(\d{8})")
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
    total = len(image_files)

    # 设置进度条
    progress_bar["maximum"] = total
    progress_bar["value"] = 0

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        faces = detector(image)
        if not faces:
            continue

        best_face = None
        min_dist = float("inf")
        for face in faces:
            try:
                descriptor = get_face_descriptor(image, face)
                dist = np.linalg.norm(descriptor - ref_descriptor)
                if dist < min_dist:
                    min_dist = dist
                    best_face = face
            except:
                continue

        if best_face:
            landmarks = get_landmarks(image, best_face)
            landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
            aligned_image = align_image(image, landmarks_points)

            date_match = date_pattern.search(img_name)
            if date_match:
                date_str = date_match.group(1)
            else:
                mod_timestamp = os.path.getmtime(img_path)
                date_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(mod_timestamp))

            base_name = f"{date_str}.jpg"
            save_path = os.path.join(output_folder, base_name)
            count = 1
            while os.path.exists(save_path):
                save_path = os.path.join(output_folder, f"{date_str}_{count}.jpg")
                count += 1

            cv2.imwrite(save_path, aligned_image)

        # 更新进度条
        progress_bar["value"] = i + 1
        root.update_idletasks()

# === Tkinter GUI ===
root = tk.Tk()
root.title("人脸对齐工具")

reference_path = tk.StringVar()
input_folder = tk.StringVar()
output_folder = tk.StringVar()

tk.Label(root, text="参考图像:").grid(row=0, column=0, sticky="e")
tk.Entry(root, textvariable=reference_path, width=50).grid(row=0, column=1)
tk.Button(root, text="选择", command=select_reference_image).grid(row=0, column=2)

tk.Label(root, text="输入文件夹:").grid(row=1, column=0, sticky="e")
tk.Entry(root, textvariable=input_folder, width=50).grid(row=1, column=1)
tk.Button(root, text="选择", command=select_input_folder).grid(row=1, column=2)

tk.Label(root, text="输出文件夹:").grid(row=2, column=0, sticky="e")
tk.Entry(root, textvariable=output_folder, width=50).grid(row=2, column=1)
tk.Button(root, text="选择", command=select_output_folder).grid(row=2, column=2)

tk.Button(root, text="开始处理", command=run_alignment_thread, bg="lightgreen").grid(row=3, column=1, pady=10)

# === 进度条和标签 ===
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=4, column=1, columnspan=2, pady=10)

progress_label = tk.Label(root, text="")
progress_label.grid(row=5, column=1, columnspan=2)

root.mainloop()
