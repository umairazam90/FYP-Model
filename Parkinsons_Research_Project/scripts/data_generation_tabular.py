
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

class UltraRealisticDataGenerator:
    '''Generates the 28-feature tabular dataset'''

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        # 11 voice features
        self.voice_features = ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Shimmer',
                               'Shimmer(dB)', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
        # 17 gait features
        self.gait_features = ['stride_time_mean', 'stride_time_std', 'step_time_mean', 'step_time_std',
                              'cadence', 'velocity', 'step_length', 'stride_length', 'force_asymmetry',
                              'step_time_cv', 'stride_time_cv', 'left_force_mean', 'right_force_mean',
                              'left_force_std', 'right_force_std', 'swing_time_ratio', 'stance_time_ratio']

    def generate_voice_data(self, n_controls=2500, n_parkinsons=2500, n_recordings=5):
        print(f"🎤 Generating {n_controls+n_parkinsons} voice subjects ({n_recordings} recordings each)...")
        n_subjects = n_controls + n_parkinsons
        # Parameters with high overlap
        voice_params = {
            'Jitter(%)': (1.0, 0.6), 'Jitter(Abs)': (0.0001, 0.00005), 'Jitter:RAP': (0.7, 0.4),
            'Jitter:PPQ5': (0.75, 0.45), 'Shimmer': (4.5, 2.0), 'Shimmer(dB)': (0.35, 0.15),
            'NHR': (0.13, 0.08), 'HNR': (19, 6), 'RPDE': (0.55, 0.2),
            'DFA': (0.63, 0.18), 'PPE': (0.16, 0.08)
        }
        voice_data = []
        for subject_id in tqdm(range(1, n_subjects + 1), desc="Generating Voice Subjects"):
            is_parkinson = 1 if subject_id > n_controls else 0
            for _ in range(n_recordings):
                row = {'subject#': subject_id, 'true_label': is_parkinson}
                for feature in self.voice_features:
                    shared_mean, shared_std = voice_params[feature]
                    base_value = np.random.normal(shared_mean, shared_std * 1.8)
                    class_bias = np.random.normal(0.1, 0.05) if is_parkinson else np.random.normal(-0.1, 0.05)
                    row[feature] = max(0, base_value * (1 + class_bias * 0.1))
                voice_data.append(row)
        return pd.DataFrame(voice_data)

    def generate_gait_data(self, n_controls=2500, n_parkinsons=2500):
        print(f"🚶 Generating {n_controls+n_parkinsons} gait subjects...")
        n_subjects = n_controls + n_parkinsons
        gait_params = {
            'stride_time_mean': (1.2, 0.15), 'stride_time_std': (0.10, 0.04), 'step_time_mean': (0.6, 0.08),
            'step_time_std': (0.06, 0.03), 'cadence': (105, 12), 'velocity': (1.05, 0.3),
            'step_length': (0.58, 0.12), 'stride_length': (1.15, 0.22), 'force_asymmetry': (30, 20),
            'step_time_cv': (0.10, 0.04), 'stride_time_cv': (0.08, 0.04), 'left_force_mean': (425, 80),
            'right_force_mean': (420, 85), 'left_force_std': (110, 35), 'right_force_std': (115, 38),
            'swing_time_ratio': (0.39, 0.05), 'stance_time_ratio': (0.61, 0.05)
        }
        gait_data = []
        for i in tqdm(range(n_subjects), desc="Generating Gait Subjects"):
            is_parkinson = 1 if i >= n_controls else 0
            row = {'subject_id': f"Sub_{i+1:04d}", 'true_label': is_parkinson}
            for feature in self.gait_features:
                shared_mean, shared_std = gait_params[feature]
                base_value = np.random.normal(shared_mean, shared_std * 1.5)
                class_bias = np.random.normal(0.08, 0.03) if is_parkinson else np.random.normal(-0.08, 0.03)
                row[feature] = max(0, base_value * (1 + class_bias * 0.05))
            gait_data.append(row)
        return pd.DataFrame(gait_data)

class RealisticFederatedDataset:
    def __init__(self, label_noise=0.10, feature_noise=0.10, random_state=42):
        self.label_noise = label_noise; self.feature_noise = feature_noise
        self.voice_scaler = StandardScaler(); self.gait_scaler = StandardScaler()
        np.random.seed(random_state)

    def create_realistic_pairs(self, voice_df, gait_df, voice_features, gait_features):
        print("Creating realistic multimodal pairs...")
        subject_voice = voice_df.groupby('subject#')[voice_features].mean().reset_index()
        subject_voice['true_label'] = voice_df.groupby('subject#')['true_label'].first().values
        gait_processed = gait_df.copy()
        print(f"Available subjects: Voice={len(subject_voice)}, Gait={len(gait_processed)}")
        voice_scaled_data = self.voice_scaler.fit_transform(subject_voice[voice_features])
        gait_scaled_data = self.gait_scaler.fit_transform(gait_processed[gait_features])
        voice_scaled = pd.DataFrame(voice_scaled_data, columns=voice_features); voice_scaled['subject#'] = subject_voice['subject#']; voice_scaled['true_label'] = subject_voice['true_label']
        gait_scaled = pd.DataFrame(gait_scaled_data, columns=gait_features); gait_scaled['subject_id'] = gait_processed['subject_id']; gait_scaled['true_label'] = gait_processed['true_label']
        multimodal_data, multimodal_labels, pair_info = [], [], []
        n_pairs = min(len(voice_scaled), len(gait_scaled))
        print(f"Creating {n_pairs} pairs with {self.label_noise*100:.0f}% label noise...")
        gait_controls = gait_scaled[gait_scaled['true_label'] == 0]
        gait_parkinsons = gait_scaled[gait_scaled['true_label'] == 1]

        for i in tqdm(range(n_pairs), desc=f"Generating {n_pairs} Pairs"):
            voice_row = voice_scaled.iloc[i]; voice_label = voice_row['true_label']; voice_data_scaled = voice_row[voice_features].values
            if np.random.random() < self.label_noise:
                pair_type = "MISMATCHED"; gait_row = gait_parkinsons.sample(1).iloc[0] if voice_label == 0 else gait_controls.sample(1).iloc[0]
            else:
                pair_type = "MATCHED"; gait_row = gait_controls.sample(1).iloc[0] if voice_label == 0 else gait_parkinsons.sample(1).iloc[0]
            gait_data_scaled = gait_row[gait_features].values
            voice_noisy = voice_data_scaled + np.random.normal(0, self.feature_noise, voice_data_scaled.shape)
            gait_noisy = gait_data_scaled + np.random.normal(0, self.feature_noise, gait_data_scaled.shape)
            multimodal_data.append(np.concatenate([voice_noisy, gait_noisy])); multimodal_labels.append(voice_label)
            pair_info.append({'voice_subject': voice_row['subject#'], 'voice_label': int(voice_label), 'gait_subject_id': gait_row['subject_id'], 'gait_label': int(gait_row['true_label']), 'label_match': int(voice_label == gait_row['true_label'])})

        multimodal_data = np.array(multimodal_data); multimodal_labels = np.array(multimodal_labels); pair_info_df = pd.DataFrame(pair_info)
        match_rate = pair_info_df['label_match'].mean()
        print(f"Pairing Analysis: Matched={match_rate*100:.1f}%, Mismatched={(1-match_rate)*100:.1f}%")
        # Return the scalers so they can be saved
        return multimodal_data, multimodal_labels, pair_info_df, self.voice_scaler, self.gait_scaler
