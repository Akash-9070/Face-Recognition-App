import cv2
import numpy as np
import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

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
            similarity_value = float(result['similarity']) if isinstance(result['similarity'], (np.ndarray, list)) else result['similarity']
            label = f"{result['label']} ({similarity_value:.2f})"

#           label = f"{result['label']} ({result['similarity']:.2f})" if hasattr(result['similarity'], 'item') else f"{result['label']} ({result['similarity']:.2f})"
            color = (0, 255, 0) if result["label"] != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


def main():
    # Path to the folder containing reference images
    image_folder = "images"
    recognizer = FaceNetRecognition(image_folder)

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and recognize faces
        results = recognizer.detect_and_recognize(frame)
        
        # Draw results on the frame
        output_frame = recognizer.draw_results(frame, results)
        
        # Display the frame
        cv2.imshow("Face Recognition", output_frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
