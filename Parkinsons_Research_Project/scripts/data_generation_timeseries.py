
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json 
# --- FIX 1: Added tqdm import clearly at the top of the script ---
from tqdm.notebook import tqdm

# --- Helper functions (generate_signal, etc.) ---
def generate_signal(base_freq, length, noise_level, drift, jitter):
    t = np.linspace(0, 1, length)
    signal = np.sin(t * base_freq * (1 + np.random.randn() * jitter))
    noise = np.random.randn(length) * noise_level
    drift_line = np.linspace(0, np.random.randn() * drift, length)
    return signal + noise + drift_line

def generate_pd_voice_signal(length=128):
    return generate_signal(base_freq=10, length=length, noise_level=0.4, drift=0.5, jitter=0.5)

def generate_control_voice_signal(length=128):
    return generate_signal(base_freq=10, length=length, noise_level=0.1, drift=0.1, jitter=0.1)

def generate_pd_gait_signal(length=256):
    t = np.linspace(0, 8, length)
    freq = 1.5 + np.random.normal(0, 0.3)
    signal = np.sin(t * freq * np.pi * 2)**2 
    noise = np.random.randn(length) * 0.15
    try:
        hesitation_start = np.random.randint(50, 150)
        hesitation_len = np.random.randint(10, 30)
        if hesitation_start + hesitation_len < length:
             signal[hesitation_start : hesitation_start + hesitation_len] *= 0.1
    except ValueError as e:
        print(f"Warning: Hesitation generation failed - {e}")
    return signal + noise

def generate_control_gait_signal(length=256):
    t = np.linspace(0, 8, length)
    freq = 2.0 + np.random.normal(0, 0.05)
    signal = np.sin(t * freq * np.pi * 2)**2 
    noise = np.random.randn(length) * 0.05
    return signal + noise

class UltraRealisticDataGenerator_TS:
    '''Generates a challenging time-series dataset.'''
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.voice_length = 128
        self.gait_length = 256
        self.voice_features_ts = [f'v_{i}' for i in range(self.voice_length)]
        self.gait_features_ts = [f'g_{i}' for i in range(self.gait_length)]

    def generate_ts_voice_data(self, n_controls=500, n_parkinsons=500):
        print(f"🎤 Generating {n_controls+n_parkinsons} TS voice subjects...")
        n_subjects = n_controls + n_parkinsons
        voice_data = []
        for subject_id in tqdm(range(1, n_subjects + 1), desc="Generating Voice Subjects"):
            is_parkinson = 1 if subject_id > n_controls else 0
            if is_parkinson:
                signal = generate_pd_voice_signal(self.voice_length)
            else:
                signal = generate_control_voice_signal(self.voice_length)
            
            row = {'subject#': subject_id, 'true_label': is_parkinson}
            row.update({f'v_{i}': val for i, val in enumerate(signal)})
            voice_data.append(row)
            
        voice_df = pd.DataFrame(voice_data)
        print(f"✅ Generated TS voice data: {voice_df.shape}")
        return voice_df

    def generate_ts_gait_data(self, n_controls=500, n_parkinsons=500):
        print(f"🚶 Generating {n_controls+n_parkinsons} TS gait subjects...")
        n_subjects = n_controls + n_parkinsons
        gait_data = []
        for i in tqdm(range(n_subjects), desc="Generating Gait Subjects"):
            is_parkinson = 1 if i >= n_controls else 0
            if is_parkinson:
                signal = generate_pd_gait_signal(self.gait_length)
            else:
                signal = generate_control_gait_signal(self.gait_length)

            row = {'subject_id': f"Sub_{i+1:04d}", 'true_label': is_parkinson}
            row.update({f'g_{i}': val for i, val in enumerate(signal)})
            gait_data.append(row)
            
        gait_df = pd.DataFrame(gait_data)
        print(f"✅ Generated TS gait data: {gait_df.shape}")
        return gait_df

class RealisticFederatedDataset_TS:
    '''Combines TS voice and gait data with noise.'''
    
    def __init__(self, label_noise=0.20, feature_noise=0.10, random_state=42):
        self.label_noise = label_noise
        self.feature_noise = feature_noise
        self.voice_scaler = StandardScaler()
        self.gait_scaler = StandardScaler()
        np.random.seed(random_state)
        
    def create_realistic_pairs(self, voice_df, gait_df, voice_features, gait_features):
        print("Creating realistic multimodal TS pairs...")
        
        subject_voice = voice_df.copy()
        gait_processed = gait_df.copy() 
        
        print(f"Available subjects: Voice={len(subject_voice)}, Gait={len(gait_processed)}")

        voice_scaled_data = self.voice_scaler.fit_transform(subject_voice[voice_features])
        gait_scaled_data = self.gait_scaler.fit_transform(gait_processed[gait_features])
        
        voice_scaled = pd.DataFrame(voice_scaled_data, columns=voice_features)
        voice_scaled['subject#'] = subject_voice['subject#']
        voice_scaled['true_label'] = subject_voice['true_label']
        
        gait_scaled = pd.DataFrame(gait_scaled_data, columns=gait_features)
        gait_scaled['subject_id'] = gait_processed['subject_id']
        gait_scaled['true_label'] = gait_processed['true_label']
        
        multimodal_data = []
        multimodal_labels = []
        pair_info = []
        
        n_pairs = min(len(voice_scaled), len(gait_scaled))
        print(f"Creating {n_pairs} pairs with {self.label_noise*100:.0f}% label noise...")
        
        gait_controls = gait_scaled[gait_scaled['true_label'] == 0]
        gait_parkinsons = gait_scaled[gait_scaled['true_label'] == 1]

        for i in tqdm(range(n_pairs), desc="Generating TS Pairs"):
            voice_row = voice_scaled.iloc[i]
            voice_label = voice_row['true_label']
            voice_data_scaled = voice_row[voice_features].values
            
            if np.random.random() < self.label_noise:
                pair_type = "MISMATCHED"
                if voice_label == 0:
                    gait_row = gait_parkinsons.sample(1).iloc[0]
                else:
                    gait_row = gait_controls.sample(1).iloc[0]
            else:
                pair_type = "MATCHED"
                if voice_label == 0:
                    gait_row = gait_controls.sample(1).iloc[0]
                else:
                    gait_row = gait_parkinsons.sample(1).iloc[0]
            
            gait_data_scaled = gait_row[gait_features].values

            voice_noisy = voice_data_scaled + np.random.normal(0, self.feature_noise, voice_data_scaled.shape)
            gait_noisy = gait_data_scaled + np.random.normal(0, self.feature_noise, gait_data_scaled.shape)
            
            combined_features = np.concatenate([voice_noisy, gait_noisy])
            multimodal_data.append(combined_features)
            multimodal_labels.append(voice_label) 
            
            pair_info.append({
                'voice_subject': voice_row['subject#'],
                'voice_label': int(voice_label),
                'gait_subject_id': gait_row['subject_id'],
                'gait_label': int(gait_row['true_label']),
                'pair_type': pair_type,
                'label_match': int(voice_label == gait_row['true_label'])
            })
            
        multimodal_data = np.array(multimodal_data)
        multimodal_labels = np.array(multimodal_labels)
        pair_info_df = pd.DataFrame(pair_info)
        
        match_rate = pair_info_df['label_match'].mean()
        print(f"Pairing Analysis: Matched={match_rate*100:.1f}%, Mismatched={(1-match_rate)*100:.1f}%")
        
        return multimodal_data, multimodal_labels, pair_info_df, self.voice_scaler, self.gait_scaler
