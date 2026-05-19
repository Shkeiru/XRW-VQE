//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file ansatz.cpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Implementation of HEA and UCCSD ansatz classes.
 */

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "ansatz.hpp"
#include <fstream>
#include <iostream>
#include <regex>
#include <spdlog/spdlog.h>
#include <vector>

#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

//------------------------------------------------------------------------------
//     HELPERS
//------------------------------------------------------------------------------

/**
 * @brief Parses a complex number from a string.
 *
 * Handles formats like "0.5j", "(1+0j)", or simple doubles.
 *
 * @param s String representation of the complex number.
 * @return std::complex<double> Parsed complex number.
 */
std::complex<double> parse_complex(std::string s) {
  // Case "0.5j" or "-0.25j"
  if (s.back() == 'j') {
    try {
      double val = std::stod(s.substr(0, s.size() - 1));
      return std::complex<double>(0.0, val);
    } catch (...) {
    }
  }
  // Case "(real+imagj)" - Currently assuming purely real or imaginary if simple
  // parsing fails
  try {
    return std::complex<double>(std::stod(s), 0.0);
  } catch (...) {
    return 0.0;
  }
}

//------------------------------------------------------------------------------
//     HEA IMPLEMENTATION
//------------------------------------------------------------------------------

/**
 * @brief Gets the name of the HEA ansatz.
 * @return std::string Name.
 */
std::string HEA::get_name() const {
  return "HEA Ansatz, depth " + std::to_string(depth) + ", qubits " +
         std::to_string(num_qubits) + ", Type : RX-RY-RZ + CNOT en ligne";
}

/**
 * @brief Gets the number of parameters for HEA.
 *
 * 3 parameters per qubit per layer (Rx, Ry, Rz).
 *
 * @return int Number of parameters.
 */
int HEA::get_num_params() const { return 2 * num_qubits * depth; }

/**
 * @brief Constructs the HEA circuit.
 *
 * Applies rotation layers followed by entangling CNOT layers.
 *
 * @param qubits Quantum register.
 * @param params Entanglement parameters.
 * @param pauli_strings Unused for HEA.
 */
void HEA::construct_circuit(Qureg qubits, const std::vector<double> &params,
                            const std::vector<std::string> &pauli_strings) {
  // 1. Safety Check
  if (params.size() != get_num_params()) {
    spdlog::error("[HEA] ERREUR: Recu {} params, attendu {}", params.size(),
                  get_num_params());
    return;
  }

  // 2. Construction
  for (int i = 0; i < depth; ++i) {
    for (int j = 0; j < num_qubits; ++j) {
      int param_index = 2 * (i * num_qubits + j);
      // Apply rotations RY, RZ
      applyRotateX(qubits, j, params[param_index]);
      applyRotateY(qubits, j, params[param_index + 1]);
    }
    // Apply linear CNOT entanglement
    for (int j = 0; j < num_qubits - 1; ++j) {
      applyControlledPauliZ(qubits, j, j + 1);
    }
  }
}

int HEA::get_num_qubits() const { return num_qubits; }

int HEA::get_depth() const { return depth; }

bool HEA::preserves_particle_number() const { return false; }

bool HEA::preserves_spin() const { return false; }

//------------------------------------------------------------------------------
//     UCCSD IMPLEMENTATION
//------------------------------------------------------------------------------

UCCSD::UCCSD(int num_qubits, int num_electrons, std::string mapping)
    : num_qubits(num_qubits), num_electrons(num_electrons) {
  spdlog::trace(
      "[UCCSD] Initializing UCCSD with {} qubits, {} electrons, mapping: {}",
      num_qubits, num_electrons, mapping);

  // 1. Call Python script to generate uccsd.json
  std::string cmd;
#ifdef _WIN32
  cmd = "wsl ";
#endif
  cmd += "python3 python/generate_uccsd.py --n_qubits " +
         std::to_string(num_qubits) + " --n_electrons " +
         std::to_string(num_electrons) + " --mapping " + mapping +
         " --output uccsd.json";

  std::cout << "[UCCSD] Executing: " << cmd << std::endl;
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "[UCCSD] Error: Python script failed with code " << ret
              << std::endl;
  }

  // 2. Load uccsd.json
  std::ifstream f("uccsd.json");
  if (!f.good()) {
    spdlog::warn("[UCCSD] uccsd.json not found. Ansatz will be empty.");
    return;
  }

  try {
    json j;
    f >> j;

    for (const auto &exc_json : j["excitations"]) {
      UCCSDExcitation exc;
      for (const auto &term_json : exc_json["pauli_terms"]) {
        UCCSDExcitation::Term t;
        t.pauli = term_json["pauli"].get<std::string>();
        t.coeff = parse_complex(term_json["coeff"].get<std::string>());
        exc.terms.push_back(t);
      }
      excitations.push_back(exc);
    }
    spdlog::info("[UCCSD] Loaded {} excitations.", excitations.size());

    // Génération de la tape optimisée pour les Pauli Gadgets
    for (size_t k = 0; k < excitations.size(); k++) {
      const auto &exc = excitations[k];

      for (const auto &term : exc.terms) {
        std::stringstream ss(term.pauli);
        std::string segment;
        
        std::string p_chars = "";
        std::vector<int> p_targets;

        while (ss >> segment) {
          if (segment.length() < 2) continue;
          char op = segment[0];
          int idx = std::stoi(segment.substr(1));
          
          // On ignore les identités, le gadget s'en fiche
          if (op == 'X' || op == 'Y' || op == 'Z') {
            p_chars += op;
            p_targets.push_back(idx);
          }
        }

        if (p_targets.empty()) continue;

        double v = term.coeff.imag();
        double precomputed_multiplier = -2.0 * v; // Le bon vieux facteur

        optimized_tape.push_back({p_chars, p_targets, (int)k, precomputed_multiplier});
      }
    }

    spdlog::info("[UCCSD] Tape optimization complete. Final size: {} gadgets.", optimized_tape.size());

    // Pre-compute gate counts per parameter for generalized PSR
    gate_counts.assign(excitations.size(), 0);
    for (const auto &inst : optimized_tape) {
      gate_counts[inst.param_idx]++;
    }

    spdlog::trace("[UCCSD] Number of parameters generated: {}", get_num_params());

  } catch (const std::exception &e) {
    spdlog::error("[UCCSD] Error parsing JSON: {}", e.what());
  }
}

UCCSD::~UCCSD() {}

std::string UCCSD::get_name() const {
  return "UCCSD (" + std::to_string(excitations.size()) + " excitations)";
}

int UCCSD::get_num_params() const { return excitations.size(); }
int UCCSD::get_num_qubits() const { return num_qubits; }
bool UCCSD::preserves_particle_number() const { return true; }
bool UCCSD::preserves_spin() const { return true; }

void UCCSD::construct_circuit(Qureg qubits, const std::vector<double> &params,
                              const std::vector<std::string> &pauli_strings) {
  if (params.size() != get_num_params()) {
    spdlog::error("[UCCSD] Error: param size mismatch. Attendu: {}, Reçu: {}",
                  get_num_params(), params.size());
    return;
  }

  // Déroulage avec les Pauli Gadgets de QuEST (Laisse le GPU respirer un peu)
  for (const auto &inst : optimized_tape) {
    // Création de la chaîne PauliStr
    PauliStr str = getPauliStr(inst.pauli_chars.c_str(), const_cast<int*>(inst.targets.data()), inst.targets.size());
    
    // Calcul de l'angle
    double angle = params[inst.param_idx] * inst.multiplier;
    
    // BAM ! Une seule instruction C++.
    applyPauliGadget(qubits, str, angle);
  }
}

std::vector<int> UCCSD::get_gate_multiplicities() const {
  return gate_counts;
}

void UCCSD::construct_circuit_with_shift(
    Qureg qubits, const std::vector<double> &params,
    const std::vector<std::string> &pauli_strings,
    int shifted_param_idx, int shifted_gate_idx, double shift_value) {

  if (params.size() != get_num_params()) {
    spdlog::error("[UCCSD] Error: param size mismatch in shifted circuit. Attendu: {}, Reçu: {}",
                  get_num_params(), params.size());
    return;
  }

  int gate_counter = 0; // counts gates belonging to shifted_param_idx
  for (const auto &inst : optimized_tape) {
    PauliStr str = getPauliStr(inst.pauli_chars.c_str(), const_cast<int*>(inst.targets.data()), inst.targets.size());

    double angle;
    if (inst.param_idx == shifted_param_idx) {
      if (gate_counter == shifted_gate_idx) {
        angle = (params[inst.param_idx] + shift_value) * inst.multiplier;
      } else {
        angle = params[inst.param_idx] * inst.multiplier;
      }
      gate_counter++;
    } else {
      angle = params[inst.param_idx] * inst.multiplier;
    }

    applyPauliGadget(qubits, str, angle);
  }
}

//------------------------------------------------------------------------------
//     ADAPT ANSATZ
//------------------------------------------------------------------------------

ADAPTAnsatz::~ADAPTAnsatz() {}

ADAPTAnsatz::ADAPTAnsatz(int num_qubits, int num_electrons) : num_qubits(num_qubits), num_electrons(num_electrons) {}

void ADAPTAnsatz::add_operator(const std::vector<GadgetInst> &op) {
  int current_param_idx = operators_tape.size(); // The new parameter index is the current size
  std::vector<GadgetInst> op_copy = op;
  
  // Re-link the parameter index for all gadgets in this operator
  for (auto &gadget : op_copy) {
    gadget.param_idx = current_param_idx;
  }
  
  operators_tape.push_back(op_copy);
}

void ADAPTAnsatz::remove_last_operator() {
  if (!operators_tape.empty()) {
    operators_tape.pop_back();
  }
}

int ADAPTAnsatz::get_num_params() const { 
  return operators_tape.size(); 
}

int ADAPTAnsatz::get_num_qubits() const { 
  return num_qubits; 
}

std::string ADAPTAnsatz::get_name() const {
  return "ADAPTAnsatz (" + std::to_string(operators_tape.size()) + " operators)";
}

bool ADAPTAnsatz::preserves_particle_number() const { 
  // ADAPT usually constructs an ansatz out of preserving operators, 
  // but just to be safe with any pool, we return false to enforce particle number penalties.
  return false; 
}

bool ADAPTAnsatz::preserves_spin() const { 
  return false; 
}

void ADAPTAnsatz::construct_circuit(Qureg qubits, const std::vector<double> &params,
                                    const std::vector<std::string> &pauli_strings) {
  if (params.size() != get_num_params()) {
    spdlog::error("[ADAPTAnsatz] Error: param size mismatch. Attendu: {}, Reçu: {}",
                  get_num_params(), params.size());
    return;
  }

  // Déroulage avec les Pauli Gadgets de QuEST
  for (const auto &op : operators_tape) {
    for (const auto &inst : op) {
      // Création de la chaîne PauliStr
      PauliStr str = getPauliStr(inst.pauli_chars.c_str(), const_cast<int*>(inst.targets.data()), inst.targets.size());
      
      // Calcul de l'angle
      double angle = params[inst.param_idx] * inst.multiplier;
      
      applyPauliGadget(qubits, str, angle);
    }
  }
}

std::vector<int> ADAPTAnsatz::get_gate_multiplicities() const {
  std::vector<int> mults;
  mults.reserve(operators_tape.size());
  for (const auto &op : operators_tape) {
    mults.push_back(static_cast<int>(op.size()));
  }
  return mults;
}

void ADAPTAnsatz::construct_circuit_with_shift(
    Qureg qubits, const std::vector<double> &params,
    const std::vector<std::string> &pauli_strings,
    int shifted_param_idx, int shifted_gate_idx, double shift_value) {

  if (params.size() != get_num_params()) {
    spdlog::error("[ADAPTAnsatz] Error: param size mismatch in shifted circuit. Attendu: {}, Reçu: {}",
                  get_num_params(), params.size());
    return;
  }

  for (size_t op_idx = 0; op_idx < operators_tape.size(); ++op_idx) {
    const auto &op = operators_tape[op_idx];
    for (size_t g = 0; g < op.size(); ++g) {
      const auto &inst = op[g];
      PauliStr str = getPauliStr(inst.pauli_chars.c_str(), const_cast<int*>(inst.targets.data()), inst.targets.size());

      double angle;
      if ((int)op_idx == shifted_param_idx && (int)g == shifted_gate_idx) {
        angle = (params[inst.param_idx] + shift_value) * inst.multiplier;
      } else {
        angle = params[inst.param_idx] * inst.multiplier;
      }

      applyPauliGadget(qubits, str, angle);
    }
  }
}
