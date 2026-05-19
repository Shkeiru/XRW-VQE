//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file vqe_context.cpp
 * @author Rayan MALEK
 * @date 2026-05-05
 * @brief Implementation of the VQEContext class.
 */

#include "vqe_context.hpp"
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <map>

//------------------------------------------------------------------------------
//     CONSTRUCTOR / DESTRUCTOR
//------------------------------------------------------------------------------

VQEContext::VQEContext(Physics &physics_ref, Ansatz &ansatz_ref)
    : physics(physics_ref), ansatz(ansatz_ref) {
  num_qubits = physics.get_num_qubits();
  n_electrons = physics.get_n_electrons();
  
  // Initialize QuEST register
  qubits = createQureg(num_qubits);
}

VQEContext::~VQEContext() {
  free_quest_resources();
}

void VQEContext::free_quest_resources() {
  destroyQureg(qubits);
  if (has_number_penalty) {
    destroyPauliStrSum(number_penalty_op);
    has_number_penalty = false;
  }
  if (has_spin_penalty) {
    destroyPauliStrSum(spin_penalty_op);
    has_spin_penalty = false;
  }
  for (auto &sum : single_term_sums) {
    destroyPauliStrSum(sum);
  }
  single_term_sums.clear();
}

//------------------------------------------------------------------------------
//     SETUP
//------------------------------------------------------------------------------

void VQEContext::setup(const std::string &fcalc_path, const std::string &ft_int_path) {
  if (is_setup) {
    spdlog::warn("[VQEContext] Setup already called. Skipping to avoid memory leaks.");
    return;
  }

  // Retrieve the Hamiltonian in QuEST format
  hamiltonian = physics.get_quest_hamiltonian();

  // Parse Pauli strings
  const auto &pauli_strings = physics.get_pauli_strings();
  paulis = pauli_strings;
  parsed_paulis.reserve(paulis.size());

  for (const auto &s : paulis) {
    PauliStr pStr;
    if (s == "I") {
      std::vector<int> idx = {0};
      pStr = getPauliStr("I", idx);
    } else {
      std::string codes;
      std::vector<int> idx;
      std::stringstream ss(s);
      std::string token;
      while (ss >> token) {
        if (token.length() < 2) continue;
        codes += token[0];
        idx.push_back(std::stoi(token.substr(1)));
      }
      if (codes.empty()) {
        std::vector<int> z = {0};
        pStr = getPauliStr("I", z);
      } else {
        pStr = getPauliStr(codes, idx);
      }
    }
    parsed_paulis.push_back(pStr);
  }

  // Pre-calculate individual PauliStr sums for noisy simulations
  single_term_sums.reserve(parsed_paulis.size());
  for (const auto &pStr : parsed_paulis) {
    qcomp one = 1.0;
    PauliStr terms_arr[] = {pStr};
    single_term_sums.push_back(createPauliStrSum(terms_arr, &one, 1));
  }

  //----------------------------------------------------------------------------
  // 1-RDM Generation & Extraction
  //----------------------------------------------------------------------------
  std::map<std::pair<int, int>, int> rdm1_index_map;
  try {
    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command += "python3 ./python/generate_1rdm.py --n_qubits " +
               std::to_string(physics.get_num_qubits()) + " --mapping jordan_wigner";

    spdlog::info("[VQEContext] Generating 1-RDM mapping: {}", command);

    FILE *pipe = _popen(command.c_str(), "r");
    if (!pipe) {
      throw std::runtime_error("_popen() failed!");
    }

    char buffer[256];
    std::string json_filepath = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      json_filepath += buffer;
    }
    _pclose(pipe);

    json_filepath.erase(std::remove(json_filepath.begin(), json_filepath.end(), '\n'), json_filepath.end());
    json_filepath.erase(std::remove(json_filepath.begin(), json_filepath.end(), '\r'), json_filepath.end());

    std::ifstream f(json_filepath);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open 1-RDM JSON file: " + json_filepath);
    }

    nlohmann::json rdm_json;
    f >> rdm_json;

    std::map<std::pair<int, int>, RDM1Term> rdm_map;

    for (const auto &item : rdm_json) {
      int p = item["p"];
      int q = item["q"];
      double c_real = item["coeff_real"];
      double c_imag = item["coeff_imag"];
      std::string pauli_string = item["string"];

      std::pair<int, int> pq_pair = {p, q};
      if (rdm_map.find(pq_pair) == rdm_map.end()) {
        rdm_map[pq_pair] = {p, q, {}, {}};
      }

      rdm_map[pq_pair].strings.push_back(getPauliStr(pauli_string));
      rdm_map[pq_pair].coeffs.push_back({c_real, c_imag});
    }

    int current_idx = 0;
    for (auto &pair : rdm_map) {
      rdm1_operators.push_back(pair.second);
      rdm1_index_map[{pair.second.p, pair.second.q}] = current_idx++;
    }

    spdlog::info("[VQEContext] Successfully parsed {} operator groups for 1-RDM", rdm1_operators.size());

  } catch (const std::exception &e) {
    spdlog::error("[VQEContext] Failed to generate/parse 1-RDM: {}", e.what());
  }

  //----------------------------------------------------------------------------
  // Penalties (Number & Spin)
  //----------------------------------------------------------------------------
  if (!ansatz.preserves_particle_number()) {
    double C = num_qubits / 2.0 - n_electrons;
    
    int total_terms = 1 + num_qubits + (num_qubits * (num_qubits - 1)) / 2;
    std::vector<PauliStr> terms(total_terms);
    std::vector<qcomp> coeffs(total_terms, 0.0);
    
    std::string identity = std::string(num_qubits, 'I');
    
    terms[0] = getPauliStr(identity);
    coeffs[0] = C * C + num_qubits / 4.0;
    
    int idx = 1;
    for (int i = 0; i < num_qubits; ++i) {
      std::string term = identity;
      term[i] = 'Z';
      terms[idx] = getPauliStr(term);
      coeffs[idx] = -C;
      idx++;
    }
    
    for (int i = 0; i < num_qubits; ++i) {
      for (int j = i + 1; j < num_qubits; ++j) {
        std::string term = identity;
        term[i] = 'Z';
        term[j] = 'Z';
        terms[idx] = getPauliStr(term);
        coeffs[idx] = 0.5;
        idx++;
      }
    }
    
    number_penalty_op = createPauliStrSum(terms.data(), coeffs.data(), total_terms);
    has_number_penalty = true;
  }

  if (!ansatz.preserves_spin()) {
    int n_orbs = num_qubits / 2;
    std::string identity = std::string(num_qubits, 'I');
    std::map<std::string, double> s2_map;

    for (int k = 0; k < n_orbs; ++k) {
        s2_map[identity] += 3.0 / 8.0;
        
        std::string term = identity;
        term[2*k] = 'Z'; 
        term[2*k+1] = 'Z';
        s2_map[term] -= 3.0 / 8.0;
    }

    for (int k = 0; k < n_orbs; ++k) {
        for (int l = k + 1; l < n_orbs; ++l) {
            std::string t;
            t = identity; t[2*k+1] = 'Z'; t[2*l+1] = 'Z'; s2_map[t] += 1.0/8.0;
            t = identity; t[2*k+1] = 'Z'; t[2*l] = 'Z';   s2_map[t] -= 1.0/8.0;
            t = identity; t[2*k] = 'Z';   t[2*l+1] = 'Z'; s2_map[t] -= 1.0/8.0;
            t = identity; t[2*k] = 'Z';   t[2*l] = 'Z';   s2_map[t] += 1.0/8.0;

            t = identity; t[2*k]='X'; t[2*k+1]='X'; t[2*l]='X'; t[2*l+1]='X'; s2_map[t] += 1.0/8.0;
            t = identity; t[2*k]='X'; t[2*k+1]='X'; t[2*l]='Y'; t[2*l+1]='Y'; s2_map[t] += 1.0/8.0;
            t = identity; t[2*k]='Y'; t[2*k+1]='Y'; t[2*l]='X'; t[2*l+1]='X'; s2_map[t] += 1.0/8.0;
            t = identity; t[2*k]='Y'; t[2*k+1]='Y'; t[2*l]='Y'; t[2*l+1]='Y'; s2_map[t] += 1.0/8.0;

            t = identity; t[2*k]='X'; t[2*k+1]='Y'; t[2*l]='X'; t[2*l+1]='Y'; s2_map[t] += 1.0/8.0;
            t = identity; t[2*k]='X'; t[2*k+1]='Y'; t[2*l]='Y'; t[2*l+1]='X'; s2_map[t] -= 1.0/8.0;
            t = identity; t[2*k]='Y'; t[2*k+1]='X'; t[2*l]='X'; t[2*l+1]='Y'; s2_map[t] -= 1.0/8.0;
            t = identity; t[2*k]='Y'; t[2*k+1]='X'; t[2*l]='Y'; t[2*l+1]='X'; s2_map[t] += 1.0/8.0;
        }
    }

    std::vector<PauliStr> s2_terms_vec;
    std::vector<qcomp> s2_coeffs_vec;
    for (const auto& kv : s2_map) {
        if (std::abs(kv.second) > 1e-9) {
            s2_terms_vec.push_back(getPauliStr(kv.first));
            s2_coeffs_vec.push_back(kv.second);
        }
    }
    
    spin_penalty_op = createPauliStrSum(s2_terms_vec.data(), s2_coeffs_vec.data(), s2_terms_vec.size());
    has_spin_penalty = true;
  }

  //----------------------------------------------------------------------------
  // Allocating Diffraction Data
  //----------------------------------------------------------------------------
  int N_rdm = rdm1_operators.size();
  int n_orbs = num_qubits / 2;

  if (N_rdm > 0 && !fcalc_path.empty() && !ft_int_path.empty()) {
    spdlog::info("[VQEContext] Loading diffraction data from {} and {}", fcalc_path, ft_int_path);

    std::vector<double> exp_F;
    std::vector<double> exp_sigma;
    std::ifstream fcalc_file(fcalc_path);
    if (!fcalc_file.is_open()) {
      spdlog::error("Could not open fcalc file!");
    } else {
      std::string line;
      while (std::getline(fcalc_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        int h, k, l;
        double f_obs, sigma;
        if (ss >> h >> k >> l >> f_obs >> sigma) {
          exp_F.push_back(f_obs);
          exp_sigma.push_back(sigma);
        }
      }
    }

    int M = exp_F.size();
    spdlog::info("[VQEContext] Parsed {} reflections from fcalc", M);

    exp_factors.resize(M);
    uncertainties.resize(M);
    for (int i = 0; i < M; ++i) {
      exp_factors(i) = std::complex<double>(exp_F[i], 0.0);
      uncertainties(i) = exp_sigma[i];
    }

    rdm1_alpha.resize(n_orbs * n_orbs);
    rdm1_beta.resize(n_orbs * n_orbs);
    rdm1_spatial.resize(n_orbs * n_orbs);

    integrals = Eigen::MatrixXcd::Zero(M, n_orbs * n_orbs);
    std::ifstream ft_int_file(ft_int_path);
    if (!ft_int_file.is_open()) {
      spdlog::error("Could not open ft_int file!");
    } else {
      std::string line;
      int line_count = 0;
      int total_orbs_expected = n_orbs * n_orbs;

      while (std::getline(ft_int_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        int p_file, q_file;
        double real_val, imag_val;
        if (ss >> p_file >> q_file >> real_val >> imag_val) {
          int ref_idx = line_count / total_orbs_expected;
          if (ref_idx < M) {
            int p_0 = p_file - 1;
            int q_0 = q_file - 1;

            int spatial_idx = p_0 * n_orbs + q_0;
            integrals(ref_idx, spatial_idx) = std::complex<double>(real_val, imag_val);
          }
          line_count++;
        }
      }
      spdlog::info("[VQEContext] Loaded ft_int file ({} lines processed).", line_count);
    }
  } else {
    spdlog::info("[VQEContext] No valid diffraction files specified or N_rdm == 0.");
  }

  is_setup = true;
}
