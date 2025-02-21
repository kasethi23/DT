import os
import numpy as np
import pickle

# Directory containing the .npz files
data_dir = "gym/data/iterative_data"  # Ensure this is the correct path
output_file = "gym/data/dataset.pkl"  # Output pickle file

# Initialize a list to store all trajectories
all_data = []

# Loop through all .npz files in the directory
for file in sorted(os.listdir(data_dir)):  # Sorting ensures order consistency
    if file.endswith(".npz"):
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path, allow_pickle=True)  # Load .npz file
        
        # Extract all arrays inside the .npz file
        extracted_data = {key: data[key] for key in data.files}  
        
        all_data.append(extracted_data)  # Append structured dictionary to the dataset list

# Convert list to dictionary for structured saving
dataset = {"trajectories": all_data}  

# Save to .pkl file
with open(output_file, "wb") as f:
    pickle.dump(dataset, f)

print(f"Saved {len(all_data)} trajectories to {output_file}")
