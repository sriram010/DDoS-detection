import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, clone_model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, concatenate, Flatten, Reshape, MaxPooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
import json

class IoTDDoSDetector:
    def __init__(self):
        """Initialize the DDoS detector with default parameters"""
        self.attack_types = [
            'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
            'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP',
            'Syn', 'TFTP', 'UDPLag', 'Benign'  # Keep all attack types
        ]
        
        self.file_mapping = {
            'DrDoS_DNS': 'DrDoS_DNS.csv',
            'DrDoS_LDAP': 'DrDoS_LDAP.csv',
            'DrDoS_MSSQL': 'DrDoS_MSSQL.csv',
            'DrDoS_NetBIOS': 'DrDoS_NetBIOS.csv',
            'DrDoS_NTP': 'DrDoS_NTP.csv',
            'DrDoS_SNMP': 'DrDoS_SNMP.csv',
            'DrDoS_SSDP': 'DrDoS_SSDP.csv',
            'DrDoS_UDP': 'DrDoS_UDP.csv',
            'Syn': 'Syn.csv',
            'TFTP': 'TFTP.csv',
            'UDPLag': 'UDPLag.csv'
        }
        
        self.num_classes = len(self.attack_types)
        self.input_shape = None  # Will be set during preprocessing
        self.scaler = StandardScaler()
        
        # Initialize models as None
        self.cnn_model = None
        self.lstm_model = None
        self.autoencoder_model = None
        self.global_model = None
        
        # Initialize hyperparameters for sub-models with optimized parameters
        self.sub_model_params = {
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'batch_size': 64,  # Increased batch size for faster training
            'epochs': 5  # Reduced epochs
        }
        
        # Initialize hyperparameters for global model with optimized parameters
        self.global_model_params = {
            'learning_rate': 0.001,
            'batch_size': 64,  # Increased batch size for faster training
            'epochs': 10,  # Reduced epochs
            'loss': 'categorical_crossentropy',
            'optimizer': Adam(learning_rate=0.001)
        }

    def preprocess_dataset(self, data_dir):
        """Preprocess the dataset with reduced data loading and value clipping"""
        print("\nPreprocessing dataset...")
        try:
            dfs = []
            
            # Define columns to process
            ip_cols = ['Source IP', 'Destination IP']
            categorical_cols = ['Protocol']
            
            for attack_type, file in self.file_mapping.items():
                file_path = os.path.join(data_dir, file)
                print(f"\nProcessing {file}...")
                
                if os.path.exists(file_path):
                    try:
                        # Read data in chunks with reduced chunk size
                        chunk_size = 1000  # Smaller chunk size
                        chunks = []
                        chunk_count = 0
                        
                        for chunk in pd.read_csv(file_path, chunksize=chunk_size, on_bad_lines='skip'):
                            try:
                                # Replace infinite values with NaN
                                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                                
                                # Fill NaN values with 0
                                chunk = chunk.fillna(0)
                                
                                # Handle IP columns
                                for col in ip_cols:
                                    if col in chunk.columns:
                                        try:
                                            chunk[col] = chunk[col].apply(self.ip_to_int).astype(np.float32)
                                            # Clip values to prevent overflow
                                            chunk[col] = np.clip(chunk[col], -1e9, 1e9)
                                        except Exception as e:
                                            print(f"Warning: Error processing column {col}: {str(e)}")
                                            chunk[col] = 0
                                
                                # Handle other categorical columns
                                for col in categorical_cols:
                                    if col in chunk.columns:
                                        try:
                                            chunk[col] = pd.Categorical(chunk[col]).codes.astype(np.float32)
                                        except Exception as e:
                                            print(f"Warning: Error processing column {col}: {str(e)}")
                                            chunk[col] = 0
                                
                                # Handle numeric columns
                                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                                for col in numeric_cols:
                                    if col not in ip_cols and col not in categorical_cols:
                                        # Clip values to prevent overflow
                                        chunk[col] = np.clip(chunk[col], -1e9, 1e9)
                                
                                # Sample only 2% from each chunk for much faster processing
                                sampled_chunk = chunk.sample(frac=0.02, random_state=42)
                                chunks.append(sampled_chunk)
                                
                                chunk_count += 1
                                if chunk_count >= 50:  # Limit to 50 chunks per file
                                    break
                                
                                if chunk_count % 10 == 0:
                                    print(f"Processed {chunk_count} chunks from {file}")
                                    
                            except Exception as e:
                                print(f"Warning: Error processing chunk: {str(e)}")
                                continue
                        
                        if chunks:
                            # Combine chunks for this attack type
                            df = pd.concat(chunks, ignore_index=True)
                            df['Label'] = attack_type  # Add label
                            print(f"Shape after sampling: {df.shape}")
                            dfs.append(df)
                            print(f"Successfully processed {file}")
                        else:
                            print(f"Warning: No valid data found in {file}")
                            
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        continue
                else:
                    print(f"Warning: File not found - {file}")
            
            if not dfs:
                raise ValueError("No data files were successfully processed")
            
            # Combine all dataframes
            print("\nCombining datasets...")
            final_df = pd.concat(dfs, ignore_index=True)
            print(f"Final combined shape: {final_df.shape}")
            
            # Ensure all numeric columns and clip values
            for col in final_df.columns:
                if col != 'Label':
                    try:
                        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                        final_df[col] = np.clip(final_df[col], -1e9, 1e9)
                    except Exception as e:
                        print(f"Warning: Could not convert column {col} to numeric: {str(e)}")
                        final_df[col] = 0
            
            # Split features and labels
            X = final_df.drop('Label', axis=1)
            y = pd.Categorical(final_df['Label']).codes
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features using robust scaling
            X_train = np.clip(self.scaler.fit_transform(X_train), -1e9, 1e9)
            X_test = np.clip(self.scaler.transform(X_test), -1e9, 1e9)
            
            # Convert to float32 to reduce memory usage
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            
            print(f"\nTraining set shape: {X_train.shape}")
            print(f"Testing set shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Complete training and evaluation pipeline"""
        print("Starting training and evaluation pipeline...")
        
        # Set input shape for models
        self.input_shape = (X_train.shape[1],)
        print(f"Input shape: {self.input_shape}")
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        # Training Phase
        print("\n2. Training Phase")
        histories = self.train_sub_models(X_train, y_train, X_test, y_test)
        
        # Training Global Model
        print("\n3. Training Global Model")
        history, results = self.train_global_model(X_train, y_train, X_test, y_test)
        
        # Testing Phase
        print("\n4. Testing Phase")
        
        # Plot and save results
        print("\n5. Generating Visualizations")
        self.plot_results(history, results)
        
        print("\nTraining and evaluation completed successfully!")
        return history, results

    def create_cnn_2d(self):
        """Create a 2D CNN model"""
        input_layer = Input(shape=self.input_shape)
        x = Reshape((-1, 1))(input_layer)  # Reshape to (batch_size, timesteps, features)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_lstm(self):
        """Create an LSTM model"""
        input_layer = Input(shape=self.input_shape)
        x = Reshape((-1, 1))(input_layer)  # Reshape to (batch_size, timesteps, features)
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_autoencoder(self):
        """Create an autoencoder model for feature extraction"""
        input_layer = Input(shape=self.input_shape)
        x = Dense(256, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_global_model(self, sub_models):
        """Create a global model that combines outputs from sub-models"""
        input_layer = Input(shape=self.input_shape)
        
        # Process input through each sub-model (excluding their output layers)
        sub_outputs = []
        for model in sub_models:
            # Create a new model excluding the last layer
            intermediate_model = Model(
                inputs=model.input,
                outputs=model.layers[-2].output
            )
            # Get features
            x = intermediate_model(input_layer)
            if len(x.shape) > 2:
                x = Flatten()(x)
            sub_outputs.append(x)
        
        # Concatenate features if there are multiple models
        if len(sub_outputs) > 1:
            combined = concatenate(sub_outputs)
        else:
            combined = sub_outputs[0]
        
        # Add final dense layers
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_sub_models(self, X_train, y_train, X_test, y_test):
        """Train individual models"""
        histories = []
        
        print("\nTraining CNN model...")
        self.cnn_model = self.create_cnn_2d()
        cnn_history = self.cnn_model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=8,  # Increased epochs
            validation_data=(X_test, y_test),
            verbose=1
        )
        histories.append(cnn_history)
        
        print("\nTraining LSTM model...")
        self.lstm_model = self.create_lstm()
        lstm_history = self.lstm_model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=8,  # Increased epochs
            validation_data=(X_test, y_test),
            verbose=1
        )
        histories.append(lstm_history)
        
        print("\nTraining Autoencoder model...")
        self.ae_model = self.create_autoencoder()
        ae_history = self.ae_model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=8,  # Increased epochs
            validation_data=(X_test, y_test),
            verbose=1
        )
        histories.append(ae_history)
        
        return histories

    def train_global_model(self, X_train, y_train, X_test, y_test):
        """Train the global model"""
        print("\nTraining Global Model...")
        
        # Create and compile global model
        self.global_model = self.create_global_model([self.cnn_model, self.lstm_model, self.ae_model])
        
        # Train the model
        history = self.global_model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=15,  # Increased epochs for global model
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Get predictions
        y_pred = self.global_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        classification_rep = classification_report(y_test_classes, y_pred_classes)
        test_loss, test_accuracy = self.global_model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_rep,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return history, results

    def plot_results(self, history, results):
        """Plot training history, confusion matrix and ROC curves"""
        # Create subplots
        fig = plt.figure(figsize=(20, 8))
        
        # Plot training history
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        
        # Plot confusion matrix
        plt.subplot(1, 3, 2)
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(self.num_classes),
            yticklabels=range(self.num_classes)
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Plot ROC curves
        plt.subplot(1, 3, 3)
        
        # Calculate ROC curve and ROC area for each class
        n_classes = self.num_classes
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Get predictions
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # First calculate per-class ROC curves
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            class_name = self.attack_types[i] if i < len(self.attack_types) else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of {class_name} (area = {roc_auc[i]:0.5f})')
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot micro and macro average curves
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.5f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.5f})',
                color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multiclass')
        plt.legend(loc="lower right", fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(results['classification_report'])
        
        # Print final metrics
        print("\nFinal Results:")
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
        # Save ROC curve data
        roc_data = {
            'micro_fpr': fpr["micro"].tolist(),
            'micro_tpr': tpr["micro"].tolist(),
            'micro_auc': roc_auc["micro"],
            'macro_fpr': fpr["macro"].tolist(),
            'macro_tpr': tpr["macro"].tolist(),
            'macro_auc': roc_auc["macro"],
            'class_fpr': {str(i): fpr[i].tolist() for i in range(n_classes)},
            'class_tpr': {str(i): tpr[i].tolist() for i in range(n_classes)},
            'class_auc': {str(i): float(roc_auc[i]) for i in range(n_classes)}
        }
        
        with open('results/roc_data.json', 'w') as f:
            json.dump(roc_data, f, indent=2)
        
    def ip_to_int(self, ip):
        try:
            parts = str(ip).split('.')
            return sum(int(part) * (256 ** (3-i)) for i, part in enumerate(parts))
        except:
            return 0

if __name__ == "__main__":
    try:
        # Initialize the detector
        detector = IoTDDoSDetector()
        
        # Set the data directory
        data_dir = "data"  # Update to correct directory
        
        # Train and evaluate the model
        X_train, X_test, y_train, y_test = detector.preprocess_dataset(data_dir)
        history, results = detector.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Print final results
        print("\nFinal Results:")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Precision: {results['classification_report']}")
        print(f"Recall: {results['classification_report']}")
        print(f"F1-score: {results['classification_report']}")
        
        # Save the model
        detector.global_model.save('model/ddos_detector.h5')
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
