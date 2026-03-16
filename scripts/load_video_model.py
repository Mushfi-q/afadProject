import torch
import torch.nn as nn
import timm

def load_video_deepfake_model(device='cpu'):
    """
    Loads a pretrained Xception model.
    To simulate a full deepfake detector without downloading huge custom weights,
    we load an Xception backbone from timm and modify the head for binary classification.
    """
    print("Loading pretrained Xception model...")
    try:
        # Load Xception base trained on ImageNet
        model = timm.create_model('xception41', pretrained=True, num_classes=2)
        
        # NOTE FOR TRAINING: Normally you'd load deepfake-specific weights here using:
        # model.load_state_dict(torch.load('path/to/xception_deepfake.pth'))
        
        model = model.to(device)
        model.eval() # Set to evaluation mode
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading video model: {e}")
        return None

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_video_deepfake_model(device)
    if model:
        # Create a dummy tensor of the correct shape (Batch, Channels, Height, Width)
        dummy_input = torch.randn(1, 3, 299, 299).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.nn.functional.softmax(output, dim=1)
            print(f"Test inference complete. Output probabilities (fake/real shape): {probs.shape}")
