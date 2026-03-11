import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("c:/code/Dis/src")))

from dis_alignment.data.swd import load_swd_audio, SWDDataset
from dis_alignment.model.inference import load_trained_encoder, extract_deep_features

def main():
    print("Loading model...")
    encoder, _ = load_trained_encoder("c:/code/Dis/checkpoints/best_model.pt")
    
    print("Loading audio...")
    swd = SWDDataset("c:/code/Dis/data/swd")
    pair = next(swd.iter_pairs())
    audio_a, _ = load_swd_audio(pair.piece_a)
    
    print("Extracting features...")
    # shape: (embed_dim, time)
    feat_a = extract_deep_features(audio_a, encoder, sr=22050, hop_length=220)
    
    print(f"Feature shape: {feat_a.shape}")
    
    # Check variance across time for each embedding dimension
    std_across_time = np.std(feat_a, axis=1)
    mean_std = np.mean(std_across_time)
    
    # Check distance between frame 0 and frame 1000
    if feat_a.shape[1] > 1000:
        f0 = feat_a[:, 0]
        f1000 = feat_a[:, 1000]
        dist = np.linalg.norm(f0 - f1000)
    else:
        dist = 0
        
    print(f"Mean std deviation across time: {mean_std:.6f}")
    print(f"Distance between frame 0 and 1000: {dist:.6f}")
    
    if mean_std < 1e-3:
        print("!!! MODEL COLLAPSED !!! All frames are nearly identical.")
    else:
        print("Model has variance, no complete collapse.")

if __name__ == "__main__":
    main()
