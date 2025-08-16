import os
from src.data_loader import load_data
from src.modelbuild import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_brain_tumor
from tensorflow.keras.models import load_model

base_dir = 'data'
model_path = "model/brain_tumor_cnn_model.h5"

def main():
    if not os.path.exists(model_path):
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_data(base_dir)

        # Build model
        model = build_model()

        # Train model
        train_model(model, X_train, y_train, X_test, y_test)

        # Save model
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")

    # Evaluate model
    X_train, X_test, y_train, y_test = load_data(base_dir)
    evaluate_model(model, X_test, y_test)

    # Predict on a new image
    test_img = os.path.join(base_dir, r"C:\Users\Ruchita M Nayak\OneDrive\Attachments\Desktop\Braintumorwebapp\data\yes\Y6.jpg")
    predict_brain_tumor(model, test_img)
    

if __name__ == "__main__":
    main()
