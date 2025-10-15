"""
Utility script for testing and extracting sample images from the dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

def extract_sample_images(num_samples=10, output_dir='sample_images'):
    """
    Extract sample images from the dataset for testing
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    images = np.load('extracted_data/images.npy')
    labels = np.load('extracted_data/labels.npy')
    
    # Get indices for flaws and no flaws
    flaw_indices = np.where(labels == 1)[0]
    no_flaw_indices = np.where(labels == 0)[0]
    
    # Sample equal amounts from each class
    num_per_class = num_samples // 2
    
    selected_flaw_indices = np.random.choice(flaw_indices, num_per_class, replace=False)
    selected_no_flaw_indices = np.random.choice(no_flaw_indices, num_per_class, replace=False)
    
    # Extract and save images
    print(f"\nExtracting {num_samples} sample images...")
    
    for i, idx in enumerate(selected_flaw_indices):
        img = images[idx]
        label = labels[idx]
        
        # Save as PNG
        filename = f"{output_dir}/flaw_{i+1}.png"
        plt.imsave(filename, img, cmap='gray')
        print(f"Saved: {filename} (FLAW)")
    
    for i, idx in enumerate(selected_no_flaw_indices):
        img = images[idx]
        label = labels[idx]
        
        # Save as PNG
        filename = f"{output_dir}/no_flaw_{i+1}.png"
        plt.imsave(filename, img, cmap='gray')
        print(f"Saved: {filename} (NO FLAW)")
    
    print(f"\nExtracted {num_samples} images to '{output_dir}/' directory")
    print("You can now upload these images to the web application for testing!")

def visualize_samples(num_samples=4):
    """
    Visualize sample images with their labels
    """
    print("Loading data...")
    images = np.load('extracted_data/images.npy')
    labels = np.load('extracted_data/labels.npy')
    
    # Get samples from each class
    flaw_indices = np.where(labels == 1)[0][:num_samples//2]
    no_flaw_indices = np.where(labels == 0)[0][:num_samples//2]
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(20, 8))
    fig.suptitle('NDT Image Samples', fontsize=16, fontweight='bold')
    
    # Plot flaws
    for i, idx in enumerate(flaw_indices):
        axes[0, i].imshow(images[idx], cmap='gray')
        axes[0, i].set_title(f'FLAW #{idx}', color='red', fontweight='bold')
        axes[0, i].axis('off')
    
    # Plot no flaws
    for i, idx in enumerate(no_flaw_indices):
        axes[1, i].imshow(images[idx], cmap='gray')
        axes[1, i].set_title(f'NO FLAW #{idx}', color='green', fontweight='bold')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'sample_visualization.png'")
    plt.show()

def test_model_on_samples(model_path='model/ndt_flaw_model.keras', num_tests=10):
    """
    Test the trained model on random samples
    """
    import tensorflow as tf
    from tensorflow import keras
    
    print("Loading model...")
    model = keras.models.load_model(model_path)
    
    print("Loading data...")
    images = np.load('extracted_data/images.npy')
    labels = np.load('extracted_data/labels.npy')
    
    # Random samples
    indices = np.random.choice(len(images), num_tests, replace=False)
    
    print(f"\nTesting model on {num_tests} random samples:\n")
    print("=" * 70)
    
    correct = 0
    for idx in indices:
        img = images[idx].reshape(1, 480, 7168, 1)
        true_label = labels[idx]
        
        # Predict
        prediction = model.predict(img, verbose=0)[0][0]
        predicted_label = 1 if prediction >= 0.5 else 0
        
        # Display result
        status = "✓ CORRECT" if predicted_label == true_label else "✗ WRONG"
        true_class = "FLAW" if true_label == 1 else "NO FLAW"
        pred_class = "FLAW" if predicted_label == 1 else "NO FLAW"
        
        print(f"Sample #{idx}")
        print(f"  True: {true_class}")
        print(f"  Predicted: {pred_class} (Confidence: {prediction*100:.2f}%)")
        print(f"  {status}")
        print("-" * 70)
        
        if predicted_label == true_label:
            correct += 1
    
    accuracy = (correct / num_tests) * 100
    print(f"\nAccuracy on samples: {accuracy:.2f}% ({correct}/{num_tests})")

def check_data_distribution():
    """
    Check and display the distribution of data
    """
    print("Loading data...")
    images = np.load('extracted_data/images.npy')
    labels = np.load('extracted_data/labels.npy')
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal images: {len(images):,}")
    print(f"Image shape: {images.shape[1:]}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")
    
    num_flaws = np.sum(labels == 1)
    num_no_flaws = np.sum(labels == 0)
    
    print(f"\nClass Distribution:")
    print(f"  Flaws: {num_flaws:,} ({num_flaws/len(labels)*100:.1f}%)")
    print(f"  No Flaws: {num_no_flaws:,} ({num_no_flaws/len(labels)*100:.1f}%)")
    
    # Memory usage
    memory_mb = images.nbytes / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    
    print("="*50 + "\n")

def main():
    """
    Main menu for utility functions
    """
    print("\n" + "="*50)
    print("NDT FLAW DETECTION - TEST UTILITIES")
    print("="*50)
    print("\nChoose an option:")
    print("1. Extract sample images for testing")
    print("2. Visualize sample images")
    print("3. Test model on random samples")
    print("4. Check data distribution")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        num = int(input("How many sample images to extract? (default 10): ") or "10")
        extract_sample_images(num_samples=num)
    
    elif choice == '2':
        num = int(input("How many samples to visualize? (default 4): ") or "4")
        visualize_samples(num_samples=num)
    
    elif choice == '3':
        num = int(input("How many samples to test? (default 10): ") or "10")
        test_model_on_samples(num_tests=num)
    
    elif choice == '4':
        check_data_distribution()
    
    elif choice == '5':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()