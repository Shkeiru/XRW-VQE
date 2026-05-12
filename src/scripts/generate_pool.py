import json
import argparse
import os

try:
    from openfermion.ops import QubitOperator
    from openfermion.transforms import get_fermion_operator, jordan_wigner
except ImportError:
    print("Error: OpenFermion is required to generate the pool.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ADAPT operator pool")
    parser.add_argument("--n_qubits", type=int, required=True, help="Number of qubits")
    parser.add_argument("--n_electrons", type=int, required=True, help="Number of electrons")
    return parser.parse_args()

def generate_hardware_efficient_pool(n_qubits):
    """
    Generates a hardware-efficient pool (e.g. single qubit Y and Z, two-qubit CNOT equivalent)
    For VQE, we need anti-Hermitian operators. A Pauli string P with purely imaginary coeff is anti-Hermitian.
    """
    pool = []
    
    # 1. Single qubit rotations (Y)
    for i in range(n_qubits):
        gadgets = [{"pauli_chars": "Y", "targets": [i], "multiplier": -2.0}]
        pool.append({"gadgets": gadgets})
        
    # 2. Entangling gates (e.g. Z_i Y_j which is anti-Hermitian)
    for i in range(n_qubits - 1):
        gadgets = [{"pauli_chars": "ZY", "targets": [i, i+1], "multiplier": -2.0}]
        pool.append({"gadgets": gadgets})
        
    return pool

def main():
    args = parse_args()
    
    pool_raw = generate_hardware_efficient_pool(args.n_qubits)
    
    with open("pool.json", "w") as f:
        json.dump({"pool": pool_raw}, f, indent=4)
        
    print(f"Generated pool.json with {len(pool_raw)} operators.")

if __name__ == "__main__":
    main()
