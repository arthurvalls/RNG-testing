import matplotlib.pyplot as plt
import numpy as np
# Data
tests = [
    "monobit_test",
    "frequency_within_block_test",
    "runs_test",
    "longest_run_ones_in_a_block_test",
    "binary_matrix_rank_test",
    "dft_test",
    "non_overlapping_template_matching_test",
    "overlapping_template_matching_test",
    "maurers_universal_test",
    "linear_complexity_test",
    "serial_test",
    "approximate_entropy_test",
    "cumulative_sums_test",
    "random_excursion_test",
    "random_excursion_variant_test"
]

fortuna = [0.2946386068817432, 0.7032266413107591, 0.5061129037877993, 0.07126662664092913, 0.41220447555505146, 0.8688028353193417, 0.9999434389052023, 0.7167066625145646, 0.7066719799760678, 0.5895080800179835, 0.5153331046600836, 0.740794957838768, 0.3675621117641168, 0.21613575490713685, 0.4039064250130682]
numpy = [0.8624618706440484, 0.5723777924345216, 0.1203030147645023, 0.7769898845072255, 0.890316096188681, 0.7018365073962303, 1.0000085441935112, 0.7567066625145646, 0.34618162564337757, 0.870125188732929, 0.8110438884372981, 0.810850412752456, 0.7724780263023368, 0.247782610978812, 0.11841279988976117]
os = [0.7041559884049895, 0.5637519463874059, 0.3918001733365396, 0.27316795003241595, 0.03867895968944739, 0.4342690840297489, 0.999992672025933, 0.4176030352181234, 0.5870176485658896, 0.004292086270314747, 0.1620744121694078, 0.16214648372791982, 0.7421582321444102, 0.18170949076408466, 0.2296780767992129]
pso = [0.7772974107895215, 0.998434296101267, 0.46125797887895814, 0.6222235254107447, 0.3042714560235986, 0.8355073996699607, 1.001443931502838, 0.9121439052639134, 0.9284760463902099, 0.12421461667833308, 0.7227206073853373, 0.7599687076908709, 0.823778331133888, 0.08587214126356223, 0.012995884780436767]
xor = [0.9521556346917863, 0.5453187107108446, 0.11641427244235385, 0.3200666876436551, 0.9105947999987611, 0.02455871132665136, 1.000001109549806, 0.0, 0.35168585838560096, 0.2110030253874, 0.018984059786093924, 0.051166414742374794, 0.7820847647878382, 0.12491135527096912, 0.16950902538212384]

methods = ["FORTUNA", "NUMPY", "OS", "PSO", "XOR"]

print(f"fortuna [mean, std.dv]: [{np.mean(fortuna)}, {np.std(fortuna)}]")
print(f"numpy [mean, std.dv]: [{np.mean(numpy)}, {np.std(numpy)}]")
print(f"os [mean, std.dv]: [{np.mean(os)}, {np.std(os)}]")
print(f"pso [mean, std.dv]: [{np.mean(pso)}, {np.std(pso)}]")
print(f"xor [mean, std.dv]: [{np.mean(xor)}, {np.std(xor)}]")

# Plotting
plt.figure(figsize=(12, 8))

for method, pvalues in zip(methods, [fortuna, numpy, os, pso, xor]):
    plt.plot(tests, pvalues, marker='o', linestyle='-', label=method)

plt.xlabel('Tests')
plt.ylabel('P-Value')
plt.title('P-Values for Different Tests and Methods')
plt.legend()
plt.grid(True)
# plt.gca().invert_yaxis()  # Invert y-axis to have the same order as in the data

plt.gca().set_xticklabels(tests, rotation=15, ha='right')
plt.show()
