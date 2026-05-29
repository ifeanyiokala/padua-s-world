import numpy as np
import matplotlib.pyplot as plt
import os

def verify_data(filepath, expected_strength):
    print(f"\n--- Verifying {filepath} ---")
    data = np.load(filepath)
    goals, colors, images = data['goals'], data['colors'], data['obs']
    
    causal_mapping = {'north': 'red', 'south': 'blue', 'east': 'yellow', 'west': 'purple'}
    matches = sum(1 for g, c in zip(goals, colors) if causal_mapping.get(g) == c)
    total = len(goals)
    
    empirical_correlation = matches / total
    print(f"Expected Correlation: ~{expected_strength:.2f}")
    print(f"Empirical Correlation: {empirical_correlation:.2f} ({matches}/{total} steps)")
    assert abs(empirical_correlation - expected_strength) < 0.05, "Correlation mismatch!"
    
    return goals, colors, images

if __name__ == "__main__":
    goals_tr, colors_tr, images_tr = verify_data('train_data.npz', expected_strength=0.80)
    verify_data('test_data.npz', expected_strength=0.25)
    
    # Generate and save the visual proof
    indices = np.random.choice(len(goals_tr), 4, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, idx in enumerate(indices):
        axes[i].imshow(images_tr[idx])
        axes[i].set_title(f"Goal: {goals_tr[idx]}\nColor: {colors_tr[idx]}")
        axes[i].axis('off')
        
    plt.tight_layout()
    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/causal_env_samples.png')
    print("\nSaved visual proof to docs/causal_env_samples.png")
