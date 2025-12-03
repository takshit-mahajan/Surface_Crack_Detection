
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model():
    """
    Evaluate the trained model on test data
    """
    print("📊 Evaluating model performance...")
    
    # CORRECT PATHS BASED ON YOUR STRUCTURE
    BASE_PATH = "D:/DeepLearningModels/Model1(SurfaceCrack)"
    PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "data/processed")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    RESULTS_PATH = os.path.join(BASE_PATH, "results")
    
    print(f"🔍 Looking for models in: {MODELS_PATH}")
    
    # Check which model file exists
    best_model_path = os.path.join(MODELS_PATH, "best_model.h5")
    final_model_path = os.path.join(MODELS_PATH, "final_model.h5")
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print("📁 Loading BEST model...")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print("📁 Loading FINAL model...")
    else:
        print("❌ No model found! Available files:")
        for file in os.listdir(MODELS_PATH):
            print(f"   - {file}")
        return
    
    print(f"📁 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Setup test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_data_path = os.path.join(PROCESSED_DATA_PATH, 'test')
    print(f"🔍 Loading test data from: {test_data_path}")
    
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(227, 227),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    print(f"📊 Class names: {class_names}")
    print(f"📊 Test samples: {test_generator.samples}")
    
    # Evaluate the model
    print("🧪 Running evaluation on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Predictions
    print("🔮 Making predictions...")
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification Report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(true_classes, predicted_classes, target_names=class_names)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Surface Crack Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    os.makedirs(RESULTS_PATH, exist_ok=True)
    confusion_matrix_path = os.path.join(RESULTS_PATH, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved: {confusion_matrix_path}")
    plt.show()
    
    # Save performance metrics
    metrics_path = os.path.join(RESULTS_PATH, 'performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("SURFACE CRACK DETECTION - MODEL EVALUATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
        # Add class distribution
        f.write("\n\nCLASS DISTRIBUTION IN TEST SET:\n")
        for i, class_name in enumerate(class_names):
            count = np.sum(true_classes == i)
            f.write(f"{class_name}: {count} samples\n")
    
    print(f"💾 Performance metrics saved: {metrics_path}")
    
    # Display final summary
    print("\n" + "🎯 EVALUATION SUMMARY " + "="*30)
    print(f"📍 Model: {os.path.basename(model_path)}")
    print(f"📍 Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"📍 Test Samples: {test_generator.samples}")
    print(f"📍 Results saved to: {RESULTS_PATH}")
    
    return test_accuracy, test_loss

if __name__ == "__main__":
    evaluate_model()