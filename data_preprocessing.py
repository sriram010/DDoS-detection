import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

class CICDDoS2019Preprocessor:
    def __init__(self, data_dir, max_rows=None):
        """Initialize preprocessor with data directory path"""
        self.data_dir = data_dir
        self.max_rows = max_rows  # New parameter to limit rows
        self.attack_files = [
            'UDPLag.csv', 'TFTP.csv', 'Syn.csv', 'DrDoS_UDP.csv',
            'DrDoS_SSDP.csv', 'DrDoS_SNMP.csv', 'DrDoS_NTP.csv',
            'DrDoS_NetBIOS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_LDAP.csv',
            'DrDoS_DNS.csv'
        ]
        self.unimportant_features = [
            'Unnamed 0', 'Flow ID', 'Source IP', 'Source Port',
            'Destination IP', 'Timestamp', 'SimilarHTTP', 'Inbound'
        ]

    def load_and_merge_data(self):
        """Load and merge all CSV files into a single dataset"""
        print("Loading and merging CSV files...")
        dfs = []
        
        for file in self.attack_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                print(f"Loading {file}...")
                df = pd.read_csv(file_path, nrows=self.max_rows)  # Limit rows read
                attack_type = file.replace('.csv', '')
                df['Attack_Type'] = attack_type
                dfs.append(df)
            else:
                print(f"Warning: {file} not found in {self.data_dir}")
        
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"Total samples after merging: {len(merged_df)}")
        return merged_df

    def remove_unimportant_features(self, df):
        """Remove unimportant features from the dataset"""
        print("Removing unimportant features...")
        columns_to_drop = [col for col in self.unimportant_features if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        return df

    def normalize_features(self, df):
        """Normalize features using min-max normalization"""
        print("Normalizing features...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'Attack_Type':  # Don't normalize the target variable
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val != 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df

    def prepare_data(self):
        """Complete data preparation pipeline"""
        # Load and merge data
        df = self.load_and_merge_data()
        
        # Remove unimportant features
        df = self.remove_unimportant_features(df)
        
        # Remove redundant features (you might want to add correlation analysis here)
        
        # Normalize features
        df = self.normalize_features(df)
        
        # Encode attack types
        attack_types = pd.get_dummies(df['Attack_Type'])
        df = pd.concat([df.drop('Attack_Type', axis=1), attack_types], axis=1)
        
        print(f"Final dataset shape: {df.shape}")
        print("\nFeature names:")
        print(df.columns.tolist())
        
        return df

    def split_data(self, df, test_size=0.2):
        """Split data into training and test sets"""
        # Separate features and labels
        X = df.iloc[:, :88]  # First 88 columns are features
        y = df.iloc[:, 88:]  # Remaining columns are attack types
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("\nData split summary:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_dir = "./data/CICDDoS2019"  # Update with your data directory
    preprocessor = CICDDoS2019Preprocessor(data_dir)
    
    # Prepare the data
    processed_data = preprocessor.prepare_data()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = preprocessor.split_data(processed_data)
    
    # Save processed datasets
    print("\nSaving processed datasets...")
    processed_data.to_csv("processed_data.csv", index=False)
    pd.concat([X_train, y_train], axis=1).to_csv("train_data.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)
