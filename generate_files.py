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
from entropy import FortunaRNG

def xor_sequence(seq1, seq2):
    xor_result = [int(a) ^ int(b) for a, b in zip(seq1, seq2)]
    xor_bytes = bytes(xor_result)
    return xor_bytes

def generate_numpy_sequence(length):
    random_bytes = np.random.bytes(length)
    return random_bytes

def generate_fortuna_sequence(length):
    rng = FortunaRNG()
    rng.reseed()  # Ensure initial reseed
    return rng.generate(length)


def generate_pso_sequence(length):
    seed = generate_seed()
    random.seed(seed)
    random_bytes = bytes([random.randint(0, 255) for _ in range(length)])
    return random_bytes

def generate_os_sequence(length):
    os_sequence = os.urandom(length)
    return os_sequence

sequence_length = 1_000_000

def save_sequence_to_bin(sequence, filename):
    with open(filename, 'wb') as f:
        f.write(sequence)
    print(f'{filename} saved...')

output_dir = 'binary_files'

numpy_sequence = generate_numpy_sequence(sequence_length)
fortuna_sequence = generate_fortuna_sequence(sequence_length)
pso_sequence = generate_pso_sequence(sequence_length)
os_sequence = generate_os_sequence(sequence_length)
xor_sequence = xor_sequence(fortuna_sequence, os_sequence)


save_sequence_to_bin(numpy_sequence, f'./{output_dir}/numpy_sequence.bin')
save_sequence_to_bin(fortuna_sequence, f'./{output_dir}/fortuna_sequence.bin')
save_sequence_to_bin(pso_sequence, f'./{output_dir}/pso_sequence.bin')
save_sequence_to_bin(os_sequence, f'./{output_dir}/os_sequence.bin')
save_sequence_to_bin(xor_sequence, f'./{output_dir}/xor_sequence.bin')
