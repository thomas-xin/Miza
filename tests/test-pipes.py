import os
import sys
import time

def generate_random_binary_data(size=16):
    """Generate random binary data of a given size."""
    return os.urandom(size)

def main():
    while True:
        # Generate 16 bytes of random binary data
        data = generate_random_binary_data(size=128)
        
        # Write the binary data to stdout
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()  # Ensure data is written immediately
        
        # Sleep for 2 seconds
        time.sleep(0.1)

if __name__ == "__main__":
    main()