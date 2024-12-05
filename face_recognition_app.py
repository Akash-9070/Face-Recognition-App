
# ======================

import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import os
import json
from datetime import datetime
import torch
from PIL import Image, ImageTk
from facenet_pytorch import InceptionResnetV1, MTCNN
from tkinter import PhotoImage



class FaceNetRecognition:
    def __init__(self, image_folder: str, similarity_threshold=0.8):
        # Initialize FaceNet model and MTCNN for face detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.image_folder = image_folder
        self.similarity_threshold = similarity_threshold
        self.known_faces = self.load_known_faces()

    def load_known_faces(self):
        """Load all images from folder and compute embeddings."""
        known_faces = []

        for filename in os.listdir(self.image_folder):
            img_path = os.path.join(self.image_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Unable to load {img_path}")
                continue

            # Convert to RGB for MTCNN
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect face and compute embedding
            boxes, _ = self.mtcnn.detect(img_rgb)
            if boxes is None:
                print(f"No face detected in {filename}")
                continue

            # Crop face and compute embedding
            face = self.mtcnn.extract(img_rgb, boxes, None)[0].to(self.device)
            embedding = self.model(face.unsqueeze(0)).detach().cpu().numpy()
            label = os.path.splitext(filename)[0]  # Use filename as label

            known_faces.append({"label": label, "embedding": embedding})

        print(f"Loaded {len(known_faces)} known faces.")
        return known_faces

    def recognize_face(self, embedding):
        """Compare given embedding with known faces."""
        max_similarity = 0
        recognized_label = "Unknown"

        for face in self.known_faces:
            # Compute cosine similarity
            similarity = np.dot(embedding, face["embedding"].T) / (
                    np.linalg.norm(embedding) * np.linalg.norm(face["embedding"]))

            # Ensure similarity is a scalar
            similarity = similarity.item() if isinstance(similarity, np.ndarray) else similarity

            if similarity > self.similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                recognized_label = face["label"]

        return recognized_label, max_similarity

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in the frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(frame_rgb)

        results = []
        if boxes is not None:
            faces = self.mtcnn.extract(frame_rgb, boxes, None)
            for i, face in enumerate(faces):
                face = face.to(self.device)
                embedding = self.model(face.unsqueeze(0)).detach().cpu().numpy()
                label, similarity = self.recognize_face(embedding)
                bbox = boxes[i].astype(int)
                results.append({"label": label, "bbox": bbox, "similarity": similarity})

        return results

    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        for result in results:
            x1, y1, x2, y2 = result['bbox']

            # Safely extract similarity value
            similarity_value = result['similarity']
            if isinstance(similarity_value, np.ndarray):
                similarity_value = similarity_value.item()

            label = f"{result['label']} ({similarity_value:.2f})"
            color = (0, 255, 0) if result["label"] != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


class FaceRecognitionGUI:
    def __init__(self, master, image_folder):
        self.master = master
        master.title("Face Recognition System")

        # Configure root window style
        master.configure(bg='white')  # Light gray background

        # Create a custom navbar
        self.navbar = tk.Frame(master, bg='#3498db', height=50)  # Blue navbar
        self.navbar.pack(side=tk.TOP, fill=tk.X)

        # Add a title to the navbar
        self.navbar_title = tk.Label(self.navbar,
                                     text="Face Recognition System",
                                     font=("Helvetica", 16, "bold"),
                                     bg='#3498db',
                                     fg='white')
        self.navbar_title.pack(side=tk.LEFT, padx=10, pady=10)

        # Persistent detection tracking
        self.detected_faces = {}
        self.log_file = "detected_faces_log.json"
        self.load_detected_faces()

        # Set up the main frame
        self.main_frame = tk.Frame(master, bg='#f0f0f0')
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create left frame for camera feed
        self.left_frame = tk.Frame(self.main_frame, width=640, height=480, bg='#f0f0f0')
        self.left_frame.pack(side=tk.LEFT, padx=10)


        try:
            # Create a label with black background for camera feed
            self.camera_label = tk.Label(self.left_frame, bg='black', width=640, height=480)
            self.camera_label.pack()
        except Exception as e:
            print(f"Error creating camera label: {e}")
            # Fallback to a standard black label if something goes wrong
            self.camera_label = tk.Label(self.left_frame, bg='black')
            self.camera_label.pack()


        # Create right frame for detected faces list
        self.right_frame = tk.Frame(self.main_frame, width=300, bg='#f0f0f0')
        self.right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        # Detected faces label with styled font
        tk.Label(self.right_frame,
                 text="Detected Faces",
                 font=("Helvetica", 16, "bold"),
                 bg='#f0f0f0').pack()

        # Treeview to show detected faces with custom style
        style = ttk.Style()
        style.theme_use('clam')  # You can try different themes like 'alt', 'default', 'classic'
        style.configure('Treeview',
                        background='#ecf0f1',
                        foreground='black',
                        rowheight=25,
                        fieldbackground='#ecf0f1')
        style.map('Treeview',
                  background=[('selected', '#3498db')],
                  foreground=[('selected', 'white')])

        self.faces_tree = ttk.Treeview(self.right_frame,
                                       columns=('Name', 'First Detected', 'Last Detected', 'Count'),
                                       show='headings',
                                       style='Treeview')
        self.faces_tree.heading('Name', text='Name')
        self.faces_tree.heading('First Detected', text='First Detected')
        self.faces_tree.heading('Last Detected', text='Last Detected')
        self.faces_tree.heading('Count', text='Count')
        self.faces_tree.pack(fill=tk.BOTH, expand=True)

        # Populate treeview with existing detected faces
        self.update_faces_treeview()

        # Initialize face recognition
        self.recognizer = FaceNetRecognition(image_folder)

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Stop flag for threading
        self.stop_event = threading.Event()

        # Start video feed
        self.update_frame()

        # Add styled quit button
        self.quit_button = tk.Button(master,
                                     text="Quit",
                                     command=self.on_quit,
                                     bg='#ec7063',  # Red background
                                     fg='white',  # White text
                                     font=("Helvetica", 10, "bold"))
        self.quit_button.pack(side='right', anchor='e', padx=20, pady=10)


    def load_detected_faces(self):
        """Load previously detected faces from JSON file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    self.detected_faces = json.load(f)
        except Exception as e:
            print(f"Error loading detected faces: {e}")
            self.detected_faces = {}

    def save_detected_faces(self):
        """Save detected faces to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.detected_faces, f, indent=4)
        except Exception as e:
            print(f"Error saving detected faces: {e}")

    def update_detected_faces(self, name):
        """Update detected faces log"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if name == "Unknown":
            return

        if name not in self.detected_faces:
            # First time detection
            self.detected_faces[name] = {
                'first_detected': current_time,
                'last_detected': current_time,
                'detection_count': 1
            }
        else:
            # Update existing detection
            self.detected_faces[name]['last_detected'] = current_time
            self.detected_faces[name]['detection_count'] += 1

        # Save to file
        self.save_detected_faces()

        # Update treeview
        self.update_faces_treeview()

    def update_faces_treeview(self):
        """Update the treeview with detected faces"""
        # Clear existing items
        for i in self.faces_tree.get_children():
            self.faces_tree.delete(i)

        # Populate with detected faces
        for name, details in self.detected_faces.items():
            self.faces_tree.insert('', 'end', values=(
                name,
                details['first_detected'],
                details['last_detected'],
                details['detection_count']
            ))

    def update_frame(self):
        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                self.master.after(30, self.update_frame)
                return

            # Detect and recognize faces
            results = self.recognizer.detect_and_recognize(frame)

            # Draw results on frame
            output_frame = self.recognizer.draw_results(frame, results)

            # Update detected faces list and log
            detected_names = set()
            for result in results:
                name = result['label']
                detected_names.add(name)
                self.update_detected_faces(name)

            # Convert frame to PhotoImage
            cv2image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        except Exception as e:
            print(f"Error in update_frame: {e}")

        # Schedule next update
        self.master.after(30, self.update_frame)

    def on_quit(self):
        self.stop_event.set()
        self.cap.release()
        self.master.quit()
        self.master.destroy()


def main():
    # Path to the folder containing reference images
    image_folder = "images"

    # Create main window
    root = tk.Tk()

    # Set window size
    root.geometry("1200x600")

    # Create GUI
    app = FaceRecognitionGUI(root, image_folder)

    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()