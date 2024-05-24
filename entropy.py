import os
import math
import struct
from collections import defaultdict
from Crypto.Cipher import AES
from Crypto.Util import Counter
from scipy.special import erfc
import numpy as np
from tqdm import tqdm
import hashlib
import time
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from tests import generate_seed
import random

class FortunaRNG:
    def __init__(self):
        self.POOL_SIZE = 64
        self.MAX_GENERATE_BYTES = 64
        self.reseed_interval = 10000  # Number of bytes generated before reseeding
        self.last_reseed_time = time.time()
        self.pools = [bytearray() for _ in range(self.POOL_SIZE)]
        self.seed = os.urandom(32)
        self.key = b''
        self.counter = 0

    def reseed(self):
        seed_material = bytearray()
        seed_material.extend(os.urandom(self.MAX_GENERATE_BYTES))
        seed_material.extend(self.seed)
        seed_material.extend(self.key)
        seed_material.extend(int(time.time()).to_bytes(8, 'big'))
        self.seed = hashlib.sha256(seed_material).digest()
        self.key = hashlib.sha256(self.key + self.seed).digest()
        self.counter += 1
        self.last_reseed_time = time.time()

    def generate(self, num_bits):
        if time.time() - self.last_reseed_time > self.reseed_interval:
            self.reseed()

        result_bytes = bytearray()
        while len(result_bytes) * 8 < num_bits:
            output = hashlib.sha256(self.key + self.seed + self.counter.to_bytes(4, 'big')).digest()
            result_bytes.extend(output)
            self.key = hashlib.sha256(self.key + output).digest()
            self.counter += 1

        return bytes(result_bytes[:num_bits//8])


def frequency_test(sequence):
    n = len(sequence)
    S = 0
    for bit in sequence:
        if bit == '1':
            S += 1
        else:
            S -= 1
    S_obs = abs(S) / math.sqrt(n)
    p_value = erfc(S_obs / math.sqrt(2))
    return p_value

def runs_test(sequence):
    n = len(sequence)
    pi = sequence.count('1') / n
    if abs(pi - 0.5) > (2 / math.sqrt(n)):
        return 0.0  # Sequence is not random

    V_n = 1
    for i in range(1, n):
        if sequence[i] != sequence[i - 1]:
            V_n += 1

    p_value = erfc(abs(V_n - (2 * n * pi * (1 - pi))) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    return p_value

def approximate_entropy_test(sequence, m=2):
    def phi(m):
        counts = defaultdict(int)
        for i in range(n):
            pattern = sequence[i:i + m]
            if len(pattern) == m:
                counts[pattern] += 1
        total = sum(counts.values())
        return sum(count / total * math.log(count / total) for count in counts.values())

    n = len(sequence)
    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    ApEn = phi_m - phi_m1
    chi_squared = 2 * n * (math.log(2) - ApEn)
    p_value = math.exp(-chi_squared / 2)
    return p_value

def monobit_test(data):
    n = len(data)
    data_int = [int(bit) for bit in data]  # Convert bits to integers
    S = np.sum(data_int) - 0.5 * n
    s_obs = abs(S) / np.sqrt(n / 4)
    p_value = chi2.sf(s_obs**2, 1)  # Using chi-square distribution
    return p_value


def chi_test(data):
    seq = np.array([int(bit) for bit in data])
    # Count occurrences of 0s and 1s
    counts = np.bincount(seq)

    # Expected frequencies (assuming a fair coin, so equal probability for 0 and 1)
    expected_frequencies = np.array([len(data)//2, len(data)//2])

    chi2_stat, p_val, _, _ = chi2_contingency([counts, expected_frequencies])
    return p_val


def combined_randomness_test(sequence):
    p1 = frequency_test(sequence)
    p2 = runs_test(sequence)
    p3 = approximate_entropy_test(sequence)
    p4 = monobit_test(sequence)
    p5 = chi_test(sequence)
    result = np.array([p1, p2, p3, p4, p5])
    result_pass = result > 0.1
    return np.sum(result_pass)


def save_sequence_to_txt(sequence, filename):
    with open(filename, 'w') as f:
        f.write(sequence)

def xor_sequence(seq1, seq2):
    return ''.join(str(int(a) ^ int(b)) for a, b in zip(seq1, seq2))


def generate_numpy_sequence(length):
    random_bits = np.random.randint(0, 2, size=length)
    sequence = ''.join(map(str, random_bits))
    return sequence

def generate_fortuna_sequence(length):
    rng = FortunaRNG()
    rng.reseed()  # Ensure initial reseed
    return rng.generate(length)

def generate_pso_sequence(length):
    seed = generate_seed()
    random.seed(seed)
    random_bits = [random.randint(0, 1) for _ in range(length)]
    sequence = ''.join(map(str, random_bits))
    return sequence

def generate_os_sequence(length):
    binary_numbers = [int.from_bytes(os.urandom(1), byteorder='big') % 2 for _ in range(sequence_length)]
    os_sequence = ''.join(map(str, binary_numbers))
    return os_sequence




# # Save sequences to .txt files
# save_sequence_to_txt(sequence, 'fortuna_sequence.txt')
# save_sequence_to_txt(numpy_sequence, 'numpy_sequence.txt')

# print("Sequences saved to .txt files successfully.")

# def load_sequence_from_txt(filename):
#     with open(filename, 'r') as f:
#         sequence = f.read().strip()
#     return sequence

# fortuna_sequence = load_sequence_from_txt('fortuna_sequence.txt')
# numpy_sequence = load_sequence_from_txt('numpy_sequence.txt')

# Run the combined randomness test on each sequence




# print('OS RNG Combined Randomness Score:', p_value_os)
# print('Fortuna RNG Combined Randomness Score:', p_value_fortuna)
# print('NumPy RNG Combined Randomness Score:', p_value_numpy)




# Evaluate randomness using the combined randomness test
# avg_fortuna = 0
# avg_np = 0
# # Run the combined randomness test
# n = 1000
# for i in range(1000):

#     avg_fortuna += combined_randomness_test(sequence)
#     avg_np  += combined_randomness_test(numpy_sequence)


# print(f'Combined Randomness Score: {avg_fortuna/n}')
# print("NumPy RNG Randomness Score:", avg_np/n)

if __name__ == "__main__":
    #do nothing
    print("do nothing")
    sequence_length = 1_000_000
    # Generate sequences for each RNG
    numpy_sequence = generate_numpy_sequence(sequence_length)
    fortuna_sequence = generate_fortuna_sequence(sequence_length)
    pso_sequence = generate_pso_sequence(sequence_length)
    os_sequence = generate_os_sequence(sequence_length)
    xor_sequence = xor_sequence(fortuna_sequence, os_sequence)


    print(f"numpy_sequence: {combined_randomness_test(numpy_sequence)}")
    print(f"fortuna_sequence: {combined_randomness_test(fortuna_sequence)}")
    print(f"pso_sequence: {combined_randomness_test(pso_sequence)}")
    print(f"os_sequence: {combined_randomness_test(os_sequence)}")
    print(f"xor_sequence: {combined_randomness_test(xor_sequence)}")

    # Compute p-values for each test and each RNG
    p_values = {
        "Frequency Test": {
            "numpy_sequence": frequency_test(numpy_sequence),
            "fortuna_sequence": frequency_test(fortuna_sequence),
            "os_sequence": frequency_test(os_sequence),
            "xor_sequence": frequency_test(xor_sequence),
            "pso_sequence": frequency_test(pso_sequence)
        },
        "Runs Test": {
            "numpy_sequence": runs_test(numpy_sequence),
            "fortuna_sequence": runs_test(fortuna_sequence),
            "os_sequence": runs_test(os_sequence),
            "xor_sequence": runs_test(xor_sequence),
            "pso_sequence": runs_test(pso_sequence)
        },
        "Approximate Entropy Test": {
            "numpy_sequence": approximate_entropy_test(numpy_sequence),
            "fortuna_sequence": approximate_entropy_test(fortuna_sequence),
            "os_sequence": approximate_entropy_test(os_sequence),
            "xor_sequence": approximate_entropy_test(xor_sequence),
            "pso_sequence": approximate_entropy_test(pso_sequence)
        },
        "Monobit Test":{
            "numpy_sequence": monobit_test(numpy_sequence),
            "fortuna_sequence": monobit_test(fortuna_sequence),
            "os_sequence": monobit_test(os_sequence),
            "xor_sequence": monobit_test(xor_sequence),
            "pso_sequence": monobit_test(pso_sequence)

        },
        "chi_test":{
            "numpy_sequence": chi_test(numpy_sequence),
            "fortuna_sequence": chi_test(fortuna_sequence),
            "os_sequence": chi_test(os_sequence),
            "xor_sequence": chi_test(xor_sequence),
            "pso_sequence": chi_test(pso_sequence)

        }
    }

    # Transpose p_values dictionary
    tests = list(p_values.keys())
    rng_names = list(p_values[tests[0]].keys())
    p_values_transposed = {rng_name: {test_name: p_values[test_name][rng_name] for test_name in tests} for rng_name in rng_names}

    # Plot p-values for each set in a single plot
    plt.figure(figsize=(10, 6))
    for rng_name, rng_results in p_values_transposed.items():
        plt.plot(tests, list(rng_results.values()), marker='o', label=rng_name)

    plt.xlabel('Test')
    plt.ylabel('p-value')
    plt.title('p-values for Randomness Tests')
    plt.legend()
    plt.ylim(0, 1)  # Set y-axis limits
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent overlapping labels
    plt.show()
