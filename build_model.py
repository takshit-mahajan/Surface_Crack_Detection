import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_cnn_model(input_shape=(227, 227, 3), num_classes=2):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Classifier
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def print_model_summary(model):
    """
    Print model architecture summary
    """
    print("🤖 CNN MODEL ARCHITECTURE")
    print("=" * 50)
    model.summary()
    
    # Count total parameters
    total_params = model.count_params()
    print(f"\n📊 Total Parameters: {total_params:,}")
    
    # Count trainable parameters (FIXED CODE)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"📊 Trainable Parameters: {trainable_params:,}")
    print(f"📊 Non-trainable Parameters: {non_trainable_params:,}")

if __name__ == "__main__":
    print("🛠️ Building Surface Crack Detection CNN Model...")
    print("=" * 50)
    
    # Create the model
    model = create_cnn_model(input_shape=(227, 227, 3))
    
    # Compile the model
    model = compile_model(model, learning_rate=0.001)
    
    # Display model architecture
    print_model_summary(model)
    
    print("\n✅ Model built successfully!")
    print("📁 Next step: Run '03_train_model.py' to train the model")