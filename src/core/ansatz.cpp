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
int HEA::get_num_params() const { return 3 * num_qubits * depth; }

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
      int param_index = 3 * (i * num_qubits + j);
      // Apply rotations RX, RY, RZ
      applyRotateX(qubits, j, params[param_index]);
      applyRotateY(qubits, j, params[param_index + 1]);
      applyRotateZ(qubits, j, params[param_index + 2]);
    }
    // Apply linear CNOT entanglement
    for (int j = 0; j < num_qubits - 1; ++j) {
      applyControlledPauliX(qubits, j, j + 1);
    }
  }
}

int HEA::get_num_qubits() const { return num_qubits; }

int HEA::get_depth() const { return depth; }

bool HEA::preserves_particle_number() const { return false; }

//------------------------------------------------------------------------------
//     UCCSD IMPLEMENTATION
//------------------------------------------------------------------------------

/**
 * @brief Constructs the UCCSD ansatz object.
 *
 * Generates and loads the `uccsd.json` file containing excitation terms.
 *
 * @param num_qubits Number of qubits.
 * @param num_electrons Number of electrons.
 * @param mapping Mapping string.
 */
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
  cmd += "python3 ./python/generate_uccsd.py --n_qubits " +
         std::to_string(num_qubits) + " --n_electrons " +
         std::to_string(num_electrons) + " --mapping " + mapping +
         " --output uccsd.json";

  std::cout << "[UCCSD] Executing: " << cmd << std::endl;
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "[UCCSD] Error: Python script failed with code " << ret
              << std::endl;
    // Continuing just in case file already exists or user wants to handle it.
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
    // Now, read the excitation to build the compiled_tape
    //  La fameuse usine à Tape (Bande Magnétique)
    for (size_t k = 0; k < excitations.size(); k++) {
      const auto &exc = excitations[k];

      for (const auto &term : exc.terms) {
        // Parse Pauli strings into dense qubit operations (0:I, 1:X, 2:Y, 3:Z)
        std::vector<int> codes(num_qubits, 0);
        std::stringstream ss(term.pauli);
        std::string segment;

        while (ss >> segment) {
          if (segment.length() < 2)
            continue;
          char op = segment[0];
          int idx = std::stoi(segment.substr(1));
          if (idx < num_qubits) {
            if (op == 'X')
              codes[idx] = 1;
            else if (op == 'Y')
              codes[idx] = 2;
            else if (op == 'Z')
              codes[idx] = 3;
          }
        }

        // Pré-calcul mathématique absolu
        // On extrait la partie imaginaire et on intègre le facteur -2.0 du Rz
        // de QuEST
        double v = term.coeff.imag();
        double precomputed_multiplier = -2.0 * v;

        std::vector<int> active_qubits;
        for (int q = 0; q < num_qubits; ++q) {
          if (codes[q] != 0)
            active_qubits.push_back(q);
        }

        if (active_qubits.empty())
          continue;

        // --- GENERATION DE LA TAPE ---

        // 1. Changement de base (Descente)
        for (int q : active_qubits) {
          if (codes[q] == 1) { // X -> Z via Hadamard
            compiled_tape.push_back({GateType::Hadamard, q, -1, -1, 0.0});
          } else if (codes[q] == 2) { // Y -> Z via Rx(pi/2)
            compiled_tape.push_back({GateType::RX_PI_2, q, -1, -1, 0.0});
          }
        }

        // 2. Escalier CNOT (Descente)
        for (size_t i = 0; i < active_qubits.size() - 1; ++i) {
          // target = active_qubits[i+1], control = active_qubits[i]
          compiled_tape.push_back({GateType::CNOT, active_qubits[i + 1],
                                   active_qubits[i], -1, 0.0});
        }

        // 3. Rotation Rz paramétrée (Le cœur de l'optimisation)
        int last_q = active_qubits.back();
        compiled_tape.push_back(
            {GateType::RZ_PARAM, last_q, -1, (int)k, precomputed_multiplier});

        // 4. Escalier CNOT (Remontée / Uncompute)
        for (int i = (int)active_qubits.size() - 2; i >= 0; --i) {
          compiled_tape.push_back({GateType::CNOT, active_qubits[i + 1],
                                   active_qubits[i], -1, 0.0});
        }

        // 5. Changement de base (Remontée / Uncompute)
        for (int q : active_qubits) {
          if (codes[q] == 1) {
            compiled_tape.push_back({GateType::Hadamard, q, -1, -1, 0.0});
          } else if (codes[q] == 2) { // Inverse de Rx(pi/2) c'est Rx(-pi/2) !
            compiled_tape.push_back({GateType::RX_MINUS_PI_2, q, -1, -1, 0.0});
          }
        }
      }
    }

    // --- PEEPHOLE OPTIMIZATION ---
    // Compress the tape by removing adjacent cancelling gates
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i + 1 < compiled_tape.size();) {
        auto &g1 = compiled_tape[i];
        auto &g2 = compiled_tape[i + 1];
        bool cancel = false;

        if (g1.target == g2.target) {
          if (g1.type == GateType::Hadamard && g2.type == GateType::Hadamard) {
            cancel = true;
          } else if (g1.type == GateType::CNOT && g2.type == GateType::CNOT &&
                     g1.control == g2.control) {
            cancel = true;
          } else if ((g1.type == GateType::RX_PI_2 &&
                      g2.type == GateType::RX_MINUS_PI_2) ||
                     (g1.type == GateType::RX_MINUS_PI_2 &&
                      g2.type == GateType::RX_PI_2)) {
            cancel = true;
          }
        }

        if (cancel) {
          compiled_tape.erase(compiled_tape.begin() + i,
                              compiled_tape.begin() + i + 2);
          changed = true;
        } else {
          i++;
        }
      }
    }
    spdlog::info(
        "[UCCSD] Tape optimization complete. Final size: {} instructions.",
        compiled_tape.size());
    spdlog::trace("[UCCSD] Number of parameters generated: {}",
                  get_num_params());

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

/**
 * @brief Constructs the UCCSD circuit.
 *
 * Applies Hartree-Fock initialization followed by Trotterized excitation
 * evolutions.
 *
 * @param qubits Quantum register.
 * @param params Amplitudes for each excitation.
 * @param pauli_strings Unused here.
 */
void UCCSD::construct_circuit(Qureg qubits, const std::vector<double> &params,
                              const std::vector<std::string> &pauli_strings) {
  if (params.size() != get_num_params()) {
    spdlog::error("[UCCSD] Error: param size mismatch. Attendu: {}, Reçu: {}",
                  get_num_params(), params.size());
    return;
  }

  // 1. Initialisation Hartree-Fock brute est deja deleguee au layer superieur
  // (Simulation::run)

  // 2. Déroulage de la Bande Magnétique (Vitesse maximale !)
  const double PI = 3.14159265358979323846;

  for (const auto &inst : compiled_tape) {
    switch (inst.type) {
    case GateType::Hadamard:
      applyHadamard(qubits, inst.target);
      break;

    case GateType::RX_PI_2:
      applyRotateX(qubits, inst.target, PI / 2.0);
      break;

    case GateType::RX_MINUS_PI_2:
      applyRotateX(qubits, inst.target, -PI / 2.0);
      break;

    case GateType::CNOT:
      applyControlledPauliX(qubits, inst.control, inst.target);
      break;

    case GateType::RZ_PARAM:
      // Le seul endroit où on fait des maths pendant la boucle : un vulgaire
      // produit !
      applyRotateZ(qubits, inst.target,
                   params[inst.param_idx] * inst.angle_multiplier);
      break;
    }
  }
}