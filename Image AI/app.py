import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from PIL import Image, ImageFilter, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from datetime import datetime
import json

class AIImageDetector:
    def __init__(self):
        self.model = None
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 20
        
    def load_and_preprocess_data(self, real_images_path, ai_images_path):
        """Load and preprocess training data"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Load real images (label = 0)
        if os.path.exists(real_images_path):
            for filename in os.listdir(real_images_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(real_images_path, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, self.img_size)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(0)  # Real image
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        # Load AI images (label = 1)
        if os.path.exists(ai_images_path):
            for filename in os.listdir(ai_images_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(ai_images_path, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, self.img_size)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(1)  # AI image
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        if not images:
            raise ValueError("No images found in the specified directories")
        
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images")
        print(f"Real images: {np.sum(labels == 0)}")
        print(f"AI images: {np.sum(labels == 1)}")
        
        return images, labels
    
    def create_model(self):
        """Create CNN model for AI detection"""
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def train_model(self, real_images_path, ai_images_path):
        """Train the AI detection model"""
        # Load data
        images, labels = self.load_and_preprocess_data(real_images_path, ai_images_path)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create model
        self.model = self.create_model()
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            keras.callbacks.ModelCheckpoint('ai_detector_model.h5', save_best_only=True)
        ]
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        # Generate predictions for detailed analysis
        y_pred = (self.model.predict(X_test) > 0.5).astype(int).flatten()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'AI']))
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def extract_image_features(self, image_path):
        """Extract technical features that might indicate AI generation"""
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            features = {}
            
            # Color distribution analysis
            if len(img_array.shape) == 3:
                for i, color in enumerate(['Red', 'Green', 'Blue']):
                    channel = img_array[:, :, i]
                    features[f'{color}_mean'] = np.mean(channel)
                    features[f'{color}_std'] = np.std(channel)
                    features[f'{color}_variance'] = np.var(channel)
            
            # Edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            features['freq_energy'] = np.sum(magnitude_spectrum)
            
            # Noise analysis
            noise = cv2.GaussianBlur(gray, (5, 5), 0) - gray
            features['noise_level'] = np.std(noise)
            
            # Compression artifacts (simplified)
            features['file_size'] = os.path.getsize(image_path)
            features['resolution'] = img.size[0] * img.size[1]
            features['compression_ratio'] = features['file_size'] / features['resolution']
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {}
    
    def predict_image(self, image_path):
        """Predict if an image is AI-generated"""
        if self.model is None:
            try:
                self.model = keras.models.load_model('ai_detector_model.h5')
            except:
                raise ValueError("No trained model found. Please train the model first.")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        confidence = abs(prediction - 0.5) * 2  # Convert to confidence score
        
        # Extract technical features
        features = self.extract_image_features(image_path)
        
        # Generate analysis
        analysis = self.generate_analysis(prediction, confidence, features)
        
        return {
            'prediction': 'AI-Generated' if prediction > 0.5 else 'Real',
            'confidence': f"{confidence:.2%}",
            'probability': f"{prediction:.4f}",
            'analysis': analysis,
            'technical_features': features
        }
    
    def generate_analysis(self, prediction, confidence, features):
        """Generate descriptive analysis"""
        analysis = []
        
        # Basic prediction analysis
        if prediction > 0.7:
            analysis.append("🤖 Strong indicators of AI generation detected")
        elif prediction > 0.5:
            analysis.append("⚠️ Moderate indicators of AI generation")
        elif prediction < 0.3:
            analysis.append("📷 Strong indicators of authentic photography")
        else:
            analysis.append("🔍 Moderate indicators of authentic photography")
        
        # Technical analysis
        if 'edge_density' in features:
            if features['edge_density'] > 0.1:
                analysis.append("• High edge density suggests detailed content")
            else:
                analysis.append("• Low edge density indicates smooth/blurred areas")
        
        if 'noise_level' in features:
            if features['noise_level'] < 5:
                analysis.append("• Very low noise levels (common in AI images)")
            elif features['noise_level'] > 15:
                analysis.append("• Natural noise levels detected")
        
        if 'compression_ratio' in features:
            if features['compression_ratio'] < 0.01:
                analysis.append("• High compression detected")
            else:
                analysis.append("• Low compression, good image quality")
        
        # Color analysis
        color_variance = 0
        if all(f'{color}_variance' in features for color in ['Red', 'Green', 'Blue']):
            color_variance = np.mean([features[f'{color}_variance'] for color in ['Red', 'Green', 'Blue']])
            if color_variance < 1000:
                analysis.append("• Limited color variation (possible AI characteristic)")
            else:
                analysis.append("• Natural color variation detected")
        
        return analysis


class AIDetectorGUI:
    def __init__(self):
        self.detector = AIImageDetector()
        self.root = tk.Tk()
        self.root.title("AI Image Detector")
        self.root.geometry("800x600")
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Image Detector", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Training section
        train_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="10")
        train_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(train_frame, text="Real Images Folder:").grid(row=0, column=0, sticky=tk.W)
        self.real_path_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.real_path_var, width=50).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(train_frame, text="Browse", command=self.browse_real_folder).grid(row=0, column=2, padx=(5, 0))
        
        ttk.Label(train_frame, text="AI Images Folder:").grid(row=1, column=0, sticky=tk.W)
        self.ai_path_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.ai_path_var, width=50).grid(row=1, column=1, padx=(5, 0))
        ttk.Button(train_frame, text="Browse", command=self.browse_ai_folder).grid(row=1, column=2, padx=(5, 0))
        
        ttk.Button(train_frame, text="Train Model", command=self.train_model_thread).grid(row=2, column=1, pady=(10, 0))
        
        # Detection section
        detect_frame = ttk.LabelFrame(main_frame, text="Image Detection", padding="10")
        detect_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(detect_frame, text="Upload Image for Analysis", command=self.analyze_image).grid(row=0, column=0, pady=(0, 10))
        
        # Results area
        self.result_text = tk.Text(detect_frame, height=15, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(detect_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def browse_real_folder(self):
        folder = filedialog.askdirectory(title="Select folder with real images")
        if folder:
            self.real_path_var.set(folder)
    
    def browse_ai_folder(self):
        folder = filedialog.askdirectory(title="Select folder with AI-generated images")
        if folder:
            self.ai_path_var.set(folder)
    
    def train_model_thread(self):
        if not self.real_path_var.get() or not self.ai_path_var.get():
            messagebox.showerror("Error", "Please select both real and AI image folders")
            return
        
        def train():
            try:
                self.progress.start()
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Training model... This may take a while.\n\n")
                self.root.update()
                
                history = self.detector.train_model(self.real_path_var.get(), self.ai_path_var.get())
                
                self.progress.stop()
                self.result_text.insert(tk.END, "Model training completed successfully!\n")
                self.result_text.insert(tk.END, "Model saved as 'ai_detector_model.h5'\n")
                messagebox.showinfo("Success", "Model trained successfully!")
                
            except Exception as e:
                self.progress.stop()
                error_msg = f"Training failed: {str(e)}"
                self.result_text.insert(tk.END, error_msg)
                messagebox.showerror("Error", error_msg)
        
        threading.Thread(target=train, daemon=True).start()
    
    def analyze_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image to analyze",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        try:
            self.progress.start()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Analyzing image...\n\n")
            self.root.update()
            
            result = self.detector.predict_image(file_path)
            
            self.progress.stop()
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"📊 ANALYSIS RESULTS\n")
            self.result_text.insert(tk.END, f"{'='*50}\n\n")
            self.result_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
            self.result_text.insert(tk.END, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            self.result_text.insert(tk.END, f"🎯 PREDICTION: {result['prediction']}\n")
            self.result_text.insert(tk.END, f"📈 Confidence: {result['confidence']}\n")
            self.result_text.insert(tk.END, f"🔢 Raw Probability: {result['probability']}\n\n")
            
            self.result_text.insert(tk.END, f"🔍 DETAILED ANALYSIS:\n")
            for point in result['analysis']:
                self.result_text.insert(tk.END, f"{point}\n")
            
            self.result_text.insert(tk.END, f"\n🛠️ TECHNICAL FEATURES:\n")
            for key, value in result['technical_features'].items():
                if isinstance(value, float):
                    self.result_text.insert(tk.END, f"• {key}: {value:.4f}\n")
                else:
                    self.result_text.insert(tk.END, f"• {key}: {value}\n")
            
        except Exception as e:
            self.progress.stop()
            error_msg = f"Analysis failed: {str(e)}"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, error_msg)
            messagebox.showerror("Error", error_msg)
    
    def run(self):
        self.root.mainloop()


# Example usage and main execution
if __name__ == "__main__":
    print("AI Image Detection System")
    print("=" * 40)
    
    # You can run the GUI
    app = AIDetectorGUI()
    app.run()
    
    # Or use the detector programmatically:
    # detector = AIImageDetector()
    # 
    # # Train model (requires folders with real and AI images)
    # detector.train_model("path/to/real/images", "path/to/ai/images")
    # 
    # # Analyze a single image
    # result = detector.predict_image("path/to/test/image.jpg")
    # print(result)