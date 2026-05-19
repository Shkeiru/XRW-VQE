//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file ansatz.hpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Definition of Ansatz abstract base class and derived classes (HEA,
 * UCCSD).
 */

#pragma once

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "compat.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <quest.h>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
//     BASE CLASS
//------------------------------------------------------------------------------

/**
 * @class Ansatz
 * @brief Abstract base class for variational ansatz circuits.
 *
 * Defines the interface for generating quantum circuits based on a specific
 * strategy (e.g., Hardware Efficient, UCCSD).
 */
class Ansatz {

public:
  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   */
  virtual ~Ansatz() = default;

  /**
   * @brief Constructs the quantum circuit for the ansatz.
   *
   * @param qubits The quantum register to apply operations to.
   * @param params The variational parameters.
   * @param pauli_strings The list of Pauli strings (needed for some ansatzes).
   */
  virtual void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) = 0;

  /**
   * @brief Gets the number of variational parameters required.
   * @return int Number of parameters.
   */
  virtual int get_num_params() const = 0;

  /**
   * @brief Gets the name and description of the ansatz.
   * @return std::string Name/Description.
   */
  virtual std::string get_name() const = 0;

  /**
   * @brief Indicates if the ansatz mathematically preserves the particle
   * number.
   * @return bool True if particle number is preserved, false otherwise.
   */
  virtual bool preserves_particle_number() const = 0;

  /**
   * @brief Indicates if the ansatz mathematically preserves the spin projection (S_z).
   * @return bool True if spin projection is preserved, false otherwise.
   */
  virtual bool preserves_spin() const = 0;

  /**
   * @brief Returns the gate multiplicity for each parameter.
   *
   * result[k] = number of distinct gates in the circuit controlled by params[k].
   * Default returns all 1s (each parameter controls exactly one gate).
   *
   * @return std::vector<int> Multiplicities vector of size get_num_params().
   */
  virtual std::vector<int> get_gate_multiplicities() const {
    return std::vector<int>(get_num_params(), 1);
  }

  /**
   * @brief Constructs the circuit with a shift applied to only one specific gate
   *        of a shared parameter (Generalized Parameter Shift Rule).
   *
   * For parameters with multiplicity M > 1, this method applies the shift_value
   * only on the shifted_gate_idx-th gate of the shifted_param_idx parameter,
   * keeping all other gates at nominal values.
   *
   * Default implementation ignores shifted_gate_idx and shifts the entire
   * parameter (correct for M = 1).
   *
   * @param qubits Quantum register.
   * @param params Current parameter values.
   * @param pauli_strings Pauli strings (for ansatzes that need them).
   * @param shifted_param_idx Index of the parameter being shifted.
   * @param shifted_gate_idx Which gate (0..M-1) of that parameter to shift.
   * @param shift_value The shift amount (typically ±π/2).
   */
  virtual void construct_circuit_with_shift(
      Qureg qubits, const std::vector<double> &params,
      const std::vector<std::string> &pauli_strings,
      int shifted_param_idx, int shifted_gate_idx, double shift_value) {
    std::vector<double> shifted_params = params;
    shifted_params[shifted_param_idx] += shift_value;
    construct_circuit(qubits, shifted_params, pauli_strings);
  }
};

//------------------------------------------------------------------------------
//     HARDWARE EFFICIENT ANSATZ (HEA)
//------------------------------------------------------------------------------

/**
 * @class HEA
 * @brief Hardware Efficient Ansatz implementation.
 *
 * Uses a layered structure of single-qubit rotations (Rx, Ry, Rz) followed by
 * entangling gates (CNOT chain). Designed to be suitable for NISQ devices.
 */
class HEA : public Ansatz {

private:
  int num_qubits; ///< Number of qubits.
  int depth;      ///< Number of layers.

public:
  /**
   * @brief Constructs a new HEA object.
   *
   * @param num_qubits Number of qubits.
   * @param depth Depth of the ansatz (number of layers).
   */
  HEA(int num_qubits, int depth) : num_qubits(num_qubits), depth(depth) {}

  void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) override;

  int get_num_qubits() const;

  int get_depth() const;

  int get_num_params() const override;

  std::string get_name() const override;

  bool preserves_particle_number() const override;

  bool preserves_spin() const override;
};

//------------------------------------------------------------------------------
//     UCCSD ANSATZ
//------------------------------------------------------------------------------

/**
 * @struct UCCSDExcitation
 * @brief Represents a single UCCSD excitation term.
 */
struct UCCSDExcitation {
  struct Term {
    std::string pauli;          ///< Pauli string (e.g., "X0 Y1").
    std::complex<double> coeff; ///< Complex coefficient.
  };
  std::vector<Term> terms;
};

// --- FINI L'USINE A GAZ DE GATE TYPE ---
// Voici la nouvelle structure pour tes gadgets, propre et digne d'un pro.
struct GadgetInst {
  std::string pauli_chars;    // Ex: "XYZ"
  std::vector<int> targets;   // Ex: {0, 1, 5}
  int param_idx;
  double multiplier;
};

/**
 * @class UCCSD
 * @brief Unitary Coupled Cluster Singles and Doubles Ansatz.
 *
 * Implements the UCCSD ansatz, chemically inspired, by generating excitations
 * based on electron number and mapping (Jordan-Wigner, Bravyi-Kitaev).
 * Relies on an external Python script to generate the excitation list.
 */
class UCCSD : public Ansatz {

private:
  int num_qubits;
  int num_electrons;
  std::vector<UCCSDExcitation> excitations;
  std::vector<GadgetInst> optimized_tape; // Adieu 'compiled_tape' !
  std::vector<int> gate_counts;           // Number of gates per parameter (for generalized PSR)

public:
  ~UCCSD() override;
  /**
   * @brief Constructs a new UCCSD object.
   *
   * Triggers the generation of the `uccsd.json` file via python script.
   *
   * @param num_qubits Number of qubits.
   * @param num_electrons Number of electrons.
   * @param mapping Mapping type (default: "jordan_wigner").
   */
  UCCSD(int num_qubits, int num_electrons,
        std::string mapping = "jordan_wigner");

  void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) override;

  int get_num_qubits() const;
  int get_num_params() const override;
  std::string get_name() const override;
  bool preserves_particle_number() const override;
  bool preserves_spin() const override;

  std::vector<int> get_gate_multiplicities() const override;
  void construct_circuit_with_shift(
      Qureg qubits, const std::vector<double> &params,
      const std::vector<std::string> &pauli_strings,
      int shifted_param_idx, int shifted_gate_idx, double shift_value) override;
};

//------------------------------------------------------------------------------
//     ADAPT ANSATZ
//------------------------------------------------------------------------------

/**
 * @class ADAPTAnsatz
 * @brief Dynamically growing ansatz for ADAPT-VQE.
 *
 * Implements an ansatz where operators (lists of Pauli gadgets) can be dynamically 
 * added or removed during the ADAPT-VQE algorithm.
 */
class ADAPTAnsatz : public Ansatz {

private:
  int num_qubits;
  int num_electrons;
  std::vector<std::vector<GadgetInst>> operators_tape;

public:
  ~ADAPTAnsatz() override;

  /**
   * @brief Constructs a new ADAPTAnsatz object.
   * @param num_qubits Number of qubits.
   * @param num_electrons Number of electrons.
   */
  ADAPTAnsatz(int num_qubits, int num_electrons);

  /**
   * @brief Adds a new operator to the ansatz and links its parameter index.
   * @param op The list of GadgetInst representing the operator.
   */
  void add_operator(const std::vector<GadgetInst> &op);

  /**
   * @brief Removes the last operator added to the ansatz.
   */
  void remove_last_operator();

  void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) override;

  int get_num_qubits() const;
  int get_num_params() const override;
  std::string get_name() const override;
  bool preserves_particle_number() const override;
  bool preserves_spin() const override;

  std::vector<int> get_gate_multiplicities() const override;
  void construct_circuit_with_shift(
      Qureg qubits, const std::vector<double> &params,
      const std::vector<std::string> &pauli_strings,
      int shifted_param_idx, int shifted_gate_idx, double shift_value) override;
};