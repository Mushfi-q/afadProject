import os
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

# Initialize MTCNN for face detection
# keep_all=False ensures we only grab the most prominent face per frame
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

# Define Xception preprocessing pipeline
# Xception typically expects 299x299, normalized with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def detect_and_preprocess_face(frame_path, output_dir=None):
    """
    Reads an image (frame), detects the face using MTCNN,
    and preprocesses it for an Xception model (returns a tensor).
    Optionally saves the cropped face for debugging.
    """
    if not os.path.exists(frame_path):
        return None

    try:
        img = Image.open(frame_path).convert('RGB')
        
        # Detect face & get bounding boxes
        boxes, probs = mtcnn.detect(img)
        
        if boxes is None or len(boxes) == 0:
            return None # No face found
            
        # Get the first (highest probability) box
        box = boxes[0].astype(int)
        
        # Crop face using PIL
        face_pil = img.crop((box[0], box[1], box[2], box[3]))
        
        # Save for visualization/logs if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            face_filename = f"face_{os.path.basename(frame_path)}"
            face_pil.save(os.path.join(output_dir, face_filename))
        
        # Preprocess for Xception (Resize to 299x299, ToTensor, Normalize)
        face_tensor = preprocess(face_pil)
        
        # Add batch dimension (1, C, H, W)
        face_tensor = face_tensor.unsqueeze(0)
        
        return face_tensor
        
    except Exception as e:
        # print(f"Error processing {frame_path}: {e}")
        return None

if __name__ == "__main__":
    test_frame = "temp/frames/000_frame_0000.jpg" 
    if os.path.exists(test_frame):
        tensor = detect_and_preprocess_face(test_frame, "temp/faces")
        if tensor is not None:
            print(f"Face processed successfully! Tensor shape: {tensor.shape}")
        else:
            print("No face detected.")
    else:
        print(f"Test frame not found: {test_frame}")
