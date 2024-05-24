import numpy as np
import matplotlib.pyplot as plt
import sys

def load_and_plot_bin(file_path):
    # Read the binary file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Convert binary data to numpy array of unsigned 8-bit integers
    array = np.frombuffer(data, dtype=np.uint8)
    
    # Calculate the size for a square array (for visualization purposes)
    size = int(np.ceil(np.sqrt(array.size)))
    
    # Pad the array with zeros if necessary to make it square
    padded_array = np.pad(array, (0, size*size - array.size), 'constant')
    
    # Reshape the array to a 2D array (bitmap)
    bitmap = padded_array.reshape((size, size))
    
    # Plot the bitmap
    plt.figure(figsize=(10,10))
    plt.imshow(bitmap, cmap='magma', interpolation='nearest')
    plt.colorbar()
    plt.title('Bitmap Visualization of Binary Data')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_bin_file>")
    else:
        file_path = sys.argv[1]
        load_and_plot_bin(file_path)
