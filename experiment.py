import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import CICDDoS2019Preprocessor
from ddos_detection import IoTDDoSDetector
import time
import os

class ExperimentRunner:
    def __init__(self, data_dir):
        """Initialize the experiment runner with optimized parameters"""
        self.data_dir = data_dir
        print("Initializing experiment with optimized parameters")
        print("Using 2% sampling rate and 50 chunks per file")
        self.detector = IoTDDoSDetector()
        self.preprocessor = CICDDoS2019Preprocessor(data_dir)
        
    def run_experiment(self):
        """Run the complete experiment pipeline with progress tracking"""
        print("Starting experiment with reduced dataset...")
        
        # Preprocessing Phase
        print("\nPreprocessing data...")
        start_time = time.time()
        X_train, X_test, y_train, y_test = self.detector.preprocess_dataset(self.data_dir)
        preprocessing_time = time.time() - start_time
        
        # Training Phase
        print("\nTraining model...")
        start_time = time.time()
        history, evaluation_results = self.detector.train_and_evaluate(X_train, X_test, y_train, y_test)
        training_time = time.time() - start_time
        
        # Save Results
        self.save_results(history, evaluation_results, training_time, preprocessing_time)
        
        print(f"\nExperiment completed in {(training_time + preprocessing_time):.2f} seconds")
        return history, evaluation_results
    
    def save_results(self, history, results, training_time, preprocessing_time):
        """Save experiment results with detailed metrics"""
        print("\nResults Summary:")
        print(f"Preprocessing Time: {preprocessing_time:.2f} seconds")
        print(f"Training Time: {training_time:.2f} seconds")
        if isinstance(results, dict):
            print("\nModel Performance Metrics:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric.capitalize()}: {value:.4f}")
        
        if history and hasattr(history, 'history'):
            print("\nTraining History:")
            for metric, values in history.history.items():
                if values:
                    print(f"Final {metric}: {values[-1]:.4f}")
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Save metrics
        metrics = {
            "accuracy": results.get('accuracy', 0),
            "loss": history.history.get('loss', [0])[-1],
            "training_time": training_time,
            "preprocessing_time": preprocessing_time
        }
        
        pd.DataFrame([metrics]).to_csv("results/metrics.csv", index=False)
        
        # Plot and save training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("results/training_history.png")
        plt.close()
        
        print("\nResults saved in 'results' directory")

if __name__ == "__main__":
    # Set data directory using absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "CICDDoS2019")
    
    # Create necessary directories
    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "model"), exist_ok=True)
    
    # Run experiment
    print(f"Using data directory: {data_dir}")
    runner = ExperimentRunner(data_dir)
    history, results = runner.run_experiment()
    
    # Save results
    print("\nSaving results and model...")
    model_path = os.path.join(current_dir, "model", "ddos_detector.h5")
    results_path = os.path.join(current_dir, "results", "experiment_results.txt")
    
    # Save model
    runner.detector.global_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save results to file
    with open(results_path, 'w') as f:
        f.write("DDoS Detection Experiment Results\n")
        f.write("================================\n\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
    
    print(f"Results saved to: {results_path}")
    print("\nExperiment completed successfully!")
