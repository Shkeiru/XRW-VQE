import argparse
import json
import os
import openfermion


def generate_1rdm_mapping(n_qubits, mapping):
    results = []

    for p in range(n_qubits):
        for q in range(n_qubits):
            # Create the fermionic operator a^\dagger_p a_q
            op = openfermion.FermionOperator(f"{p}^ {q}", 1.0)

            # Apply the requested mapping
            if mapping == "jordan_wigner":
                qubit_op = openfermion.jordan_wigner(op)
            elif mapping == "bravyi_kitaev":
                qubit_op = openfermion.bravyi_kitaev(op, n_qubits=n_qubits)
            else:
                raise ValueError(f"Unsupported mapping: {mapping}")

            # Iterate over the resulting QubitOperator terms
            for term, coeff in qubit_op.terms.items():
                # term is a tuple like ((0, 'X'), (2, 'Z')), potentially empty for identity
                chars = ["I"] * n_qubits
                for idx, pauli in term:
                    chars[idx] = pauli

                pauli_string = "".join(chars)

                results.append(
                    {
                        "p": p,
                        "q": q,
                        "coeff_real": coeff.real,
                        "coeff_imag": coeff.imag,
                        "string": pauli_string,
                    }
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate 1-RDM Pauli string mapping.")
    parser.add_argument(
        "--n_qubits", type=int, required=True, help="Number of qubits (spin-orbitals)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        choices=["jordan_wigner", "bravyi_kitaev"],
        help="Fermion to qubit mapping",
    )

    args = parser.parse_args()

    mapping_data = generate_1rdm_mapping(args.n_qubits, args.mapping)

    out_name = f"1rdm_mapping_N{args.n_qubits}_{args.mapping}.json"

    # Write to the current working directory from which the script is called
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)

    # Only print the filename to stdout for the C++ backend.
    # The script runs in WSL, so os.path.abspath outputs a Linux-style path
    # (/mnt/c/...), which the Windows C++ application cannot read.
    print(out_name)


if __name__ == "__main__":
    main()
