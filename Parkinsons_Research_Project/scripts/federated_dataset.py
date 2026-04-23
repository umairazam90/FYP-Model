
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib # Added

class RealisticFederatedDataset:
    '''Combines voice and gait data with noise and misalignment.'''

    def __init__(self, label_noise=0.35, feature_noise=0.12, random_state=42):
        self.label_noise = label_noise
        self.feature_noise = feature_noise
        self.voice_scaler = StandardScaler()
        self.gait_scaler = StandardScaler()
        np.random.seed(random_state)

    def create_realistic_pairs(self, voice_df, gait_df, voice_features, gait_features):
        print("Creating realistic multimodal pairs...")

        # Aggregate voice data
        subject_voice = voice_df.groupby('subject#')[voice_features].mean().reset_index()
        subject_voice['true_label'] = voice_df.groupby('subject#')['true_label'].first().values

        gait_processed = gait_df.copy() # Gait data is already one row per subject

        print(f"Available subjects: Voice={len(subject_voice)}, Gait={len(gait_processed)}")

        # Scale features - Fit scalers here
        voice_scaled = self.voice_scaler.fit_transform(subject_voice[voice_features])
        gait_scaled = self.gait_scaler.fit_transform(gait_processed[gait_features])

        multimodal_data = []
        multimodal_labels = []
        pair_info = []

        n_pairs = min(len(voice_scaled), len(gait_scaled))
        print(f"Creating {n_pairs} pairs with {self.label_noise*100:.0f}% label noise...")

        for i in range(n_pairs):
            voice_label = subject_voice.iloc[i]['true_label']
            pair_type = "UNKNOWN"

            if np.random.random() < self.label_noise:
                # Mismatched pair
                different_class_gait_indices = gait_processed[gait_processed['true_label'] != voice_label].index
                if len(different_class_gait_indices) > 0:
                    gait_idx = np.random.choice(different_class_gait_indices)
                    gait_data = gait_scaled[gait_idx]
                    pair_type = "MISMATCHED"
                else: # Fallback
                    same_class_gait_indices = gait_processed[gait_processed['true_label'] == voice_label].index
                    if len(same_class_gait_indices) > 0:
                       gait_idx = np.random.choice(same_class_gait_indices)
                       gait_data = gait_scaled[gait_idx]
                       pair_type = "MATCHED (Fallback)"
                    else:
                        gait_idx = i % len(gait_scaled)
                        gait_data = gait_scaled[gait_idx]
                        pair_type = "RANDOM (Fallback)"
            else:
                # Matched pair
                same_class_gait_indices = gait_processed[gait_processed['true_label'] == voice_label].index
                if len(same_class_gait_indices) > 0:
                   gait_idx = np.random.choice(same_class_gait_indices)
                   gait_data = gait_scaled[gait_idx]
                   pair_type = "MATCHED"
                else: # Fallback
                    different_class_gait_indices = gait_processed[gait_processed['true_label'] != voice_label].index
                    if len(different_class_gait_indices) > 0:
                        gait_idx = np.random.choice(different_class_gait_indices)
                        gait_data = gait_scaled[gait_idx]
                        pair_type = "MISMATCHED (Fallback)"
                    else:
                        gait_idx = i % len(gait_scaled)
                        gait_data = gait_scaled[gait_idx]
                        pair_type = "RANDOM (Fallback)"

            # Add feature noise
            voice_noisy = voice_scaled[i] + np.random.normal(0, self.feature_noise, voice_scaled[i].shape)
            gait_noisy = gait_data + np.random.normal(0, self.feature_noise, gait_data.shape)

            combined_features = np.concatenate([voice_noisy, gait_noisy])
            multimodal_data.append(combined_features)
            multimodal_labels.append(voice_label) # Use voice label as ground truth

            pair_info.append({
                'voice_subject': subject_voice.iloc[i]['subject#'],
                'voice_label': int(voice_label),
                'gait_subject_id': gait_processed.iloc[gait_idx]['subject_id'],
                'gait_label': int(gait_processed.iloc[gait_idx]['true_label']),
                'pair_type': pair_type,
                'label_match': int(voice_label == gait_processed.iloc[gait_idx]['true_label'])
            })

        multimodal_data = np.array(multimodal_data)
        multimodal_labels = np.array(multimodal_labels)
        pair_info_df = pd.DataFrame(pair_info)

        match_rate = pair_info_df['label_match'].mean()
        print(f"Pairing Analysis: Matched={match_rate*100:.1f}%, Mismatched={(1-match_rate)*100:.1f}%")

        # Return the fitted scalers along with the data
        return multimodal_data, multimodal_labels, pair_info_df, subject_voice, gait_processed, self.voice_scaler, self.gait_scaler
