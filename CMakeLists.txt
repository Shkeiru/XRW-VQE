cmake_minimum_required(VERSION 3.15)

# Nom du projet et version C++ (20 pour être tranquille)
project(QuantumBeast VERSION 0.1 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. OPTIONS & CONFIGURATION
# ==========================================

add_compile_definitions(QuEST_PRECISION=2) # 2 = Double, 1 = Float

option(BUILD_GUI "Construire l'interface graphique (Laptop Mode)" ON)
option(USE_MPI   "Activer MPI pour le HPC (Cluster Mode)" OFF)
option(BUILD_EXAMPLES "Construire les exemples de base" OFF)

# Dossier de sortie des binaires (plus propre)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Module FetchContent pour télécharger le monde entier
include(FetchContent)

# ==========================================
# 2. DEPENDANCES (LE GROS MATOS)
# ==========================================

# --- A. QuEST (Le Moteur) ---
message(STATUS "Fetching QuEST...")

# ==========================================
# AUTO-DETECTION BACKEND ACCELERATION (GPU)
# ==========================================
include(CheckLanguage)
# 1. Tentative NVIDIA (CUDA)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "🔥 Backend NVIDIA détecté ! Activation de CUDA pour QuEST.")
    set(ENABLE_CUDA 1 CACHE BOOL "" FORCE)

    
    # Récupération automatique de la target architecture (ex: sm_80)
    # CMake >= 3.18
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE)
    
    # Les variables de QuEST (selon version)
    set(COMPILER "nvcc" CACHE STRING "" FORCE)
    set(GPU_COMPUTE_CAPABILITY "native" CACHE STRING "Cible CUDA par défaut")
else()
    # 2. Tentative AMD (ROCm / HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        message(STATUS "🔴 Backend AMD détecté ! Activation de HIP/ROCm.")
        
        # QuEST n'a pas de support natif formel CMake pour AMD sans forker,
        # Mais un build HIP générique passe si GPUACCELERATED=1
        set(GPUACCELERATED 1 CACHE BOOL "" FORCE)
        # Configuration compiler spécifique HIP si nécessaire
        set(COMPILER "hipcc" CACHE STRING "" FORCE)
    else()
        # 3. Fallback sur Numérique CPU / MPI
        message(STATUS "🐢 Aucun GPU compatible détecté. Fallback CPU multicœurs.")
        set(GPUACCELERATED 0 CACHE BOOL "" FORCE)
    endif()
endif()

# Configuration AVANT le fetch (c'est comme ça que QuEST marche)
if(USE_MPI)
    set(DISTRIBUTED 1 CACHE BOOL "" FORCE)
    set(MULTITHREADED 1 CACHE BOOL "" FORCE) # Hybride MPI+OpenMP
else()
    set(DISTRIBUTED 0 CACHE BOOL "" FORCE)
    set(MULTITHREADED 1 CACHE BOOL "" FORCE)
endif()
set(PRECISION 2 CACHE BOOL "" FORCE)      # 2 = Double, 1 = Float

FetchContent_Declare(
    QuEST
    GIT_REPOSITORY https://github.com/QuEST-Kit/QuEST.git
    GIT_TAG        v4.2.0
)
FetchContent_MakeAvailable(QuEST)

# --- B. CLI11 (L'interface Ligne de Commande) ---
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.6.1
)
FetchContent_MakeAvailable(CLI11)

# --- C. Spdlog (Les Logs) ---
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(spdlog)

# --- D. NLopt (L'Optimiseur) ---
# Note: NLopt via CMake peut être capricieux sous Windows.
# Si ça plante, commente et installe via vcpkg ou apt-get.
message(STATUS "Fetching NLopt...")
set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
set(NLOPT_PYTHON OFF CACHE BOOL "Disable Python wrappers" FORCE)
set(NLOPT_OCTAVE OFF CACHE BOOL "Disable Octave wrappers" FORCE)
set(NLOPT_MATLAB OFF CACHE BOOL "Disable Matlab wrappers" FORCE)
set(NLOPT_GUILE OFF CACHE BOOL "Disable Guile wrappers" FORCE)
set(NLOPT_SWIG OFF CACHE BOOL "Disable SWIG" FORCE)
set(NLOPT_TESTS OFF CACHE BOOL "Disable Tests" FORCE)
FetchContent_Declare(
    nlopt
    GIT_REPOSITORY https://github.com/stevengj/nlopt.git
    GIT_TAG        v2.10.1
)
FetchContent_MakeAvailable(nlopt)

# --- E. Eigen & JSON (Header Only) ---
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen Docs" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable All Testing" FORCE)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.1
)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

# ==========================================
# 3. LE COEUR (CORE LIBRARY)
# ==========================================
# C'est ta lib physique. Elle ne connait PAS le GUI.
add_library(core STATIC
    src/core/simulation.cpp
    src/core/adapt_sim.cpp
    src/core/vqe_context.cpp
    src/core/ansatz.cpp
    src/core/physics.cpp
    src/core/opt.cpp
)

# On linke les maths et la physique
target_link_libraries(core 
    PUBLIC 
    QuEST 
    Eigen3::Eigen 
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    nlopt
)
target_include_directories(core PUBLIC 
    src
    src/core
    # Fix for WSL: ensure QuEST headers are found
    ${QuEST_SOURCE_DIR}/quest/include
    ${quest_SOURCE_DIR}/quest/include
)

# ==========================================
# 4. CONFIGURATION DES MODULES
# ==========================================
# Si on construit le GUI, on désactive le CLI par défaut pour gagner du temps
if(BUILD_GUI)
    option(BUILD_CLI "Construire aussi le CLI en mode GUI" OFF)
else()
    option(BUILD_CLI "Construire le CLI (Mode Headless)" ON)
endif()

# ==========================================
# 5. L'APPLICATION CLI (HPC)
# ==========================================
if(BUILD_CLI)
    message(STATUS "Building CLI Application...")
    add_executable(quantum_cli src/apps/CLI/main_cli.cpp)
    target_link_libraries(quantum_cli PRIVATE 
        core
        QuEST 
        CLI11::CLI11)

    if(USE_MPI)
        find_package(MPI REQUIRED)
        target_link_libraries(quantum_cli PRIVATE MPI::MPI_CXX)
        target_compile_definitions(quantum_cli PRIVATE USE_MPI)
    endif()

    # Copie le script Python pour la generation d'Hamiltonien
    add_custom_command(TARGET quantum_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_cli>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_hamiltonian.py
        $<TARGET_FILE_DIR:quantum_cli>/python/generate_hamiltonian.py
    )

    # Copie le script Python pour la generation d'UCCSD
    add_custom_command(TARGET quantum_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_cli>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_uccsd.py
        $<TARGET_FILE_DIR:quantum_cli>/python/generate_uccsd.py
    )

    # Copie le script Python pour la generation de la 1RDM
    add_custom_command(TARGET quantum_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_cli>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_1rdm.py
        $<TARGET_FILE_DIR:quantum_cli>/python/generate_1rdm.py
    )
    # Copie le script Python pour la generation de la 2RDM (le commentaire original disait Hamiltonien, oups)
    add_custom_command(TARGET quantum_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_cli>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_2rdm.py
        $<TARGET_FILE_DIR:quantum_cli>/python/generate_2rdm.py
    )

    # Copie le script Python pour la generation du pool
    add_custom_command(TARGET quantum_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_cli>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_pool.py
        $<TARGET_FILE_DIR:quantum_cli>/python/generate_pool.py
    )
else()
    message(STATUS "Skipping CLI build (Focus on GUI).")
endif()

# ==========================================
# 6. L'APPLICATION GUI (LAPTOP)
# ==========================================
if(BUILD_GUI)
    message(STATUS "Configuring GUI Mode...")

    # --- Téléchargement GLFW ---
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
    )
    FetchContent_MakeAvailable(glfw)

    # --- Téléchargement ImGui ---
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        docking # Branche docking pour le multi-viewport (le futur)
    )
    FetchContent_MakeAvailable(imgui)

    # --- Téléchargement ImPlot ---
    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(implot)

    # --- Téléchargement ImGuiFileDialog ---
    FetchContent_Declare(
        ImGuiFileDialog
        GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
        GIT_TAG        v0.6.8
    )
    FetchContent_GetProperties(ImGuiFileDialog)
    if(NOT imguifiledialog_POPULATED)
        FetchContent_Populate(ImGuiFileDialog)
    endif()

    # --- GLAD (Manuel) ---
    # On crée une lib statique pour GLAD
    add_library(glad STATIC external/glad/src/glad.c)
    target_include_directories(glad PUBLIC external/glad/include)

    # --- Exécutable GUI ---
    add_executable(quantum_gui src/apps/GUI/main_gui.cpp src/apps/GUI/gui.cpp)

    # On ajoute les sources ImGui/ImPlot directement (Mode Brutal)
    # C'est nécessaire car ils n'ont pas de CMakeLists natif propre
    target_sources(quantum_gui PRIVATE
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
        ${implot_SOURCE_DIR}/implot_demo.cpp
        ${imguifiledialog_SOURCE_DIR}/ImGuiFileDialog.cpp
    )

    # Includes pour que le compilo trouve les headers ImGui
    target_include_directories(quantum_gui PRIVATE 
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${implot_SOURCE_DIR}
        ${imguifiledialog_SOURCE_DIR}
        ${QuEST_SOURCE_DIR}/quest/include
        ${quest_SOURCE_DIR}/quest/include
    )

    # Linkage final
    target_link_libraries(quantum_gui PRIVATE 
        core
        QuEST 
        glad 
        glfw 
        ${OPENGL_LIBRARIES}
    )
    
    # Copie les DLLs si sous Windows (Optionnel mais pratique)
    if(WIN32)
        add_custom_command(TARGET quantum_gui POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:glfw>
            $<TARGET_FILE_DIR:quantum_gui>
        )
    endif()

    # Copie le script Python pour la generation d'Hamiltonien
    add_custom_command(TARGET quantum_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_gui>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_hamiltonian.py
        $<TARGET_FILE_DIR:quantum_gui>/python/generate_hamiltonian.py
    )

    # Copie le script Python pour la generation d'UCCSD
    add_custom_command(TARGET quantum_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_gui>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_uccsd.py
        $<TARGET_FILE_DIR:quantum_gui>/python/generate_uccsd.py
    )

    # Copie le script Python pour la generation de la 1RDM
    add_custom_command(TARGET quantum_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_gui>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_1rdm.py
        $<TARGET_FILE_DIR:quantum_gui>/python/generate_1rdm.py
    )
    # Copie le script Python pour la generation de la 2RDM
    add_custom_command(TARGET quantum_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_gui>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_2rdm.py
        $<TARGET_FILE_DIR:quantum_gui>/python/generate_2rdm.py
    )

    # Copie le script Python pour la generation du pool
    add_custom_command(TARGET quantum_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:quantum_gui>/python
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/scripts/generate_pool.py
        $<TARGET_FILE_DIR:quantum_gui>/python/generate_pool.py
    )

else()
    message(STATUS "GUI Mode DISABLED. Building Headless only.")
endif()