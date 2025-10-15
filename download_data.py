"""
Helper script to download NDT data files from GitHub
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_from_github():
    """
    Download data files from GitHub repository
    """
    print("="*70)
    print("NDT ML Flaw - Data Download Helper")
    print("="*70)
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("\nIMPORTANT: Automatic download from GitHub may not work for large files.")
    print("If download fails, please manually download files from:")
    print("https://github.com/koomas/NDT_ML_Flaw\n")
    
    # List of files to download (you'll need to update these based on actual repo)
    base_url = "https://github.com/koomas/NDT_ML_Flaw/raw/main/"
    
    # Example files - update based on your actual repository structure
    files = [
        "100_batch1.xz",
        "100_batch1.txt",
        "200_batch1.xz", 
        "200_batch1.txt",
        # Add more files as needed
    ]
    
    print(f"Attempting to download {len(files)} files to '{data_dir}/'...\n")
    
    success_count = 0
    failed_files = []
    
    for filename in files:
        destination = data_dir / filename
        
        if destination.exists():
            print(f"✓ {filename} already exists, skipping...")
            success_count += 1
            continue
        
        try:
            url = base_url + filename
            print(f"\nDownloading {filename}...")
            download_file(url, destination)
            print(f"✓ Successfully downloaded {filename}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            failed_files.append(filename)
    
    print("\n" + "="*70)
    print(f"Download Summary: {success_count}/{len(files)} files successful")
    
    if failed_files:
        print(f"\nFailed downloads: {', '.join(failed_files)}")
        print("\nPlease manually download these files from:")
        print("https://github.com/koomas/NDT_ML_Flaw")
        print(f"And place them in the '{data_dir}' folder")
    else:
        print("\n✓ All files downloaded successfully!")
    
    print("="*70)

def manual_download_instructions():
    """
    Print manual download instructions
    """
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nStep 1: Go to the GitHub repository")
    print("  → https://github.com/koomas/NDT_ML_Flaw")
    
    print("\nStep 2: Download the data files")
    print("  → Look for files with .xz extension (e.g., 100_batch1.xz)")
    print("  → Download corresponding .txt files (e.g., 100_batch1.txt)")
    print("  → Click on each file, then click 'Download raw file' button")
    
    print("\nStep 3: Place files in the 'data' folder")
    print(f"  → Create a folder named 'data' in your project directory")
    print(f"  → Move all downloaded .xz and .txt files to this folder")
    
    print("\nStep 4: Verify the files")
    print("  Your 'data' folder should look like:")
    print("  data/")
    print("    ├── 100_batch1.xz")
    print("    ├── 100_batch1.txt")
    print("    ├── 200_batch1.xz")
    print("    ├── 200_batch1.txt")
    print("    └── ... (other files)")
    
    print("\nStep 5: Run the extraction script")
    print("  → python 1_data_extraction.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Try automatic download (may not work for large files)")
    print("2. Show manual download instructions")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            download_from_github()
        except Exception as e:
            print(f"\nAutomatic download failed: {e}")
            print("\nFalling back to manual instructions...\n")
            manual_download_instructions()
    else:
        manual_download_instructions()