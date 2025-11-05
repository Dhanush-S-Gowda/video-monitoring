import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import albumentations as A

# ========================
# CONFIGURATION
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Update this path to your dataset
IMAGES_DIR = 'images'  # Your folder containing class folders (bhuvan, dhanush, etc.)

# These will be created automatically
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# ========================
# 1. DATASET SPLITTING
# ========================
def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split images from source directory into train/val/test sets"""
    
    print("\n" + "="*50)
    print("SPLITTING DATASET")
    print("="*50)
    
    # Create destination directories
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all class folders
    class_folders = [f for f in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, f))]
    
    if len(class_folders) == 0:
        print(f"ERROR: No class folders found in '{source_dir}'")
        return None
    
    print(f"\nFound {len(class_folders)} classes: {class_folders}")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name in class_folders:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all images for this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if len(image_files) == 0:
            print(f"⚠ Warning: No images found for class '{class_name}'")
            continue
        
        print(f"\n📁 Class '{class_name}': {len(image_files)} images")
        
        # Shuffle images
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        print(f"  ├─ Train: {len(train_files)} images")
        print(f"  ├─ Val:   {len(val_files)} images")
        print(f"  └─ Test:  {len(test_files)} images")
        
        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)
        
        # Create class directories in train/val/test
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Copy files to respective directories
        for img in train_files:
            src = os.path.join(class_path, img)
            dst = os.path.join(TRAIN_DIR, class_name, img)
            shutil.copy2(src, dst)
        
        for img in val_files:
            src = os.path.join(class_path, img)
            dst = os.path.join(VAL_DIR, class_name, img)
            shutil.copy2(src, dst)
        
        for img in test_files:
            src = os.path.join(class_path, img)
            dst = os.path.join(TEST_DIR, class_name, img)
            shutil.copy2(src, dst)
    
    print("\n" + "="*50)
    print(f"✓ Dataset split complete!")
    print(f"  Total Train: {total_train} images")
    print(f"  Total Val:   {total_val} images")
    print(f"  Total Test:  {total_test} images")
    print("="*50)
    
    return class_folders

# ========================
# 2. DATA GENERATORS WITH HEAVY AUGMENTATION
# ========================
def create_data_generators():
    """Create data generators with comprehensive augmentation for training"""
    
    # Training data with HEAVY augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        
        # Geometric transformations
        rotation_range=30,              # Rotate images up to 30 degrees
        width_shift_range=0.25,         # Shift horizontally by 25%
        height_shift_range=0.25,        # Shift vertically by 25%
        shear_range=0.25,               # Shear transformation
        zoom_range=0.3,                 # Zoom in/out by 30%
        
        # Flipping
        horizontal_flip=True,           # Randomly flip horizontally
        
        # Color/Lighting augmentations
        brightness_range=[0.6, 1.4],    # Vary brightness (60% to 140%)
        channel_shift_range=30.0,       # Shift color channels
        
        # Fill mode for areas after transformations
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\n📊 Data Augmentation Applied:")
    print("  ✓ Rotation: ±30°")
    print("  ✓ Width/Height Shift: ±25%")
    print("  ✓ Shear Transformation: 25%")
    print("  ✓ Zoom: ±30%")
    print("  ✓ Horizontal Flip")
    print("  ✓ Brightness: 60%-140%")
    print("  ✓ Channel Shift: ±30")
    
    return train_generator, val_generator

# Alternative: Advanced augmentation using Albumentations (Optional)
def create_advanced_augmentation():
    """
    Advanced augmentation pipeline using Albumentations
    Uncomment and use this if you want even more aggressive augmentation
    """
    transform = A.Compose([
        # Geometric transformations
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        
        # Color/Lighting augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.8),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=7, p=1),
        ], p=0.3),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        ], p=0.2),
        
        # Quality degradation
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Downscale(scale_min=0.7, scale_max=0.9, p=0.2),
        
        # Lighting conditions
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.1),
        
        # Cutout/Coarse Dropout
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    ])
    
    return transform

# ========================
# 3. MODEL BUILDING
# ========================
def build_face_recognition_model(num_classes):
    """Build MobileNetV2-based face recognition model"""
    
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    print(f"Base Model: MobileNetV2")
    print(f"Total layers in base model: {len(base_model.layers)}")
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.3, name='dropout2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"Output classes: {num_classes}")
    print(f"Total parameters: {model.count_params():,}")
    print("="*50)
    
    return model, base_model

# ========================
# 4. TRAINING FUNCTION
# ========================
def train_model(train_gen, val_gen, num_classes):
    """Train the face recognition model with two-phase approach"""
    
    # Build model
    model, base_model = build_face_recognition_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_face_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # ========== PHASE 1: Transfer Learning ==========
    print("\n" + "="*50)
    print("PHASE 1: TRANSFER LEARNING")
    print("Training with frozen MobileNet base")
    print("Epochs: 20")
    print("="*50)
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========== PHASE 2: Fine-tuning ==========
    print("\n" + "="*50)
    print("PHASE 2: FINE-TUNING")
    print("Unfreezing and fine-tuning top layers")
    print("Epochs: 30 (20-50)")
    print("="*50)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze first 100 layers, fine-tune the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    trainable_layers = sum([1 for layer in model.layers if layer.trainable])
    print(f"Trainable layers: {trainable_layers}")
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        initial_epoch=20,
        verbose=1
    )
    
    # Save final model
    model.save('final_face_model.h5')
    print("\n✓ Model saved as 'final_face_model.h5'")
    
    # Save class names
    class_names = list(train_gen.class_indices.keys())
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("✓ Class names saved as 'class_names.txt'")
    
    return model, history1, history2

# ========================
# 5. VISUALIZATION
# ========================
def plot_training_history(history1, history2):
    """Plot training and validation metrics"""
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.axvline(x=20, color='green', linestyle='--', label='Fine-tuning starts', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.axvline(x=20, color='green', linestyle='--', label='Fine-tuning starts', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'training_history.png'")
    plt.show()

# ========================
# 6. MAIN EXECUTION
# ========================
def main():
    """Main function to train the face recognition model"""
    
    print("\n" + "="*60)
    print(" "*10 + "FACE RECOGNITION MODEL TRAINING")
    print(" "*15 + "Using MobileNetV2")
    print("="*60)
    
    # Step 1: Check and split dataset
    if not os.path.exists(IMAGES_DIR):
        print(f"\n❌ ERROR: Images directory '{IMAGES_DIR}' not found!")
        print("Please update IMAGES_DIR variable to point to your images folder")
        print("\nExpected structure:")
        print(f"{IMAGES_DIR}/")
        print("  ├── bhuvan/")
        print("  ├── dhanush/")
        print("  └── [other names]/")
        return
    
    # Check if dataset needs to be split
    need_split = True
    if os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0:
        print(f"\n✓ Found existing dataset splits")
        response = input("Do you want to re-split the dataset? (y/n): ")
        need_split = response.lower() == 'y'
    
    if need_split:
        class_folders = split_dataset(IMAGES_DIR)
        if class_folders is None:
            return
    
    # Step 2: Create data generators
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    train_gen, val_gen = create_data_generators()
    
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMG_SIZE}")
    
    # Step 3: Train model
    model, history1, history2 = train_model(train_gen, val_gen, num_classes)
    
    # Step 4: Plot training history
    plot_training_history(history1, history2)
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE! 🎉")
    print("="*60)
    
    final_train_acc = history2.history['accuracy'][-1] * 100
    final_val_acc = history2.history['val_accuracy'][-1] * 100
    
    print(f"\n📈 Final Results:")
    print(f"  Training Accuracy:   {final_train_acc:.2f}%")
    print(f"  Validation Accuracy: {final_val_acc:.2f}%")
    
    print(f"\n💾 Saved Files:")
    print(f"  ├─ best_face_model.h5      (best model during training)")
    print(f"  ├─ final_face_model.h5     (final trained model)")
    print(f"  ├─ class_names.txt         (list of person names)")
    print(f"  └─ training_history.png    (training plots)")
    
    print(f"\n✅ Model is ready for testing and deployment!")
    print(f"   Use 'final_face_model.h5' for predictions")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()