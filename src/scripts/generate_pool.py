import json
import argparse
import os

try:
    from openfermion.ops import FermionOperator
    from openfermion.transforms import jordan_wigner
except ImportError:
    print("Erreur : OpenFermion est requis. Et cette fois-ci, c'est pas du bluff, on l'utilise VRAIMENT.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Générateur de pool ADAPT (Vraies excitations fermioniques)")
    parser.add_argument("--n_qubits", type=int, required=True, help="Nombre total d'orbitales spin (qubits)")
    parser.add_argument("--n_electrons", type=int, required=True, help="Nombre d'électrons (orbitales occupées)")
    return parser.parse_args()

def qubit_op_to_json_gadget(qubit_op):
    """
    Traduit un QubitOperator d'OpenFermion vers ton format JSON douteux.
    Un opérateur fermionique anti-hermitien donne des chaînes de Pauli avec des
    coefficients purement imaginaires après Jordan-Wigner. On extrait la partie imaginaire.
    """
    gadgets = []
    
    # .terms est un dictionnaire de type { ((0, 'X'), (1, 'Y')): 0.5j, ... }
    for term, coeff in qubit_op.terms.items():
        if abs(coeff) < 1e-8:
            continue # On ignore les zéros numériques
            
        # Pour ton VQE, tu veux la valeur réelle du multiplicateur de la chaîne anti-hermitienne
        multiplier = coeff.imag 
        
        if not term:
            continue # On skip l'opérateur Identité
            
        pauli_chars = "".join([pauli for _, pauli in term])
        targets = [idx for idx, _ in term]
        
        gadgets.append({
            "pauli_chars": pauli_chars,
            "targets": targets,
            "multiplier": multiplier
        })
        
    return {"gadgets": gadgets}

def generate_fermionic_pool(n_qubits, n_electrons):
    """
    Génère le pool avec les excitations simples (p, q) et doubles (p, q, r, s)
    en respectant l'approximation particule-trou.
    """
    pool = []
    
    # Espace occupé (trous potentiels) et espace virtuel (particules potentielles)
    occ = range(n_electrons)
    virt = range(n_electrons, n_qubits)
    
    # 1. Excitations Simples : p^ q - q^ p
    # Un électron saute de q (occupé) vers p (virtuel)
    for p in virt:
        for q in occ:
            # Création de l'opérateur fermionique anti-hermitien
            op = FermionOperator(f'{p}^ {q}') - FermionOperator(f'{q}^ {p}')
            
            # Transformation en qubits via Jordan-Wigner
            qubit_op = jordan_wigner(op)
            
            # Ajout au pool dans ton format
            pool.append(qubit_op_to_json_gadget(qubit_op))
            
    # 2. Excitations Doubles : p^ q^ r s - s^ r^ q p
    # Deux électrons sautent de r, s (occupés) vers p, q (virtuels)
    for p in virt:
        for q in virt:
            if p <= q: 
                continue # Ordre strict pour éviter les doublons (p > q)
                
            for r in occ:
                for s in occ:
                    if r <= s: 
                        continue # Ordre strict (r > s)
                        
                    op = FermionOperator(f'{p}^ {q}^ {r} {s}') - FermionOperator(f'{s}^ {r}^ {q} {p}')
                    qubit_op = jordan_wigner(op)
                    
                    # Optionnel mais propre : si l'opérateur est vide après simplification, on l'ignore
                    if qubit_op.terms: 
                        pool.append(qubit_op_to_json_gadget(qubit_op))
                    
    return pool

def main():
    args = parse_args()
    
    if args.n_electrons >= args.n_qubits:
        print("Génie, tu as mis plus d'électrons que d'orbitales disponibles. Révise ton principe de Pauli.")
        exit(1)
        
    pool_raw = generate_fermionic_pool(args.n_qubits, args.n_electrons)
    
    with open("pool_fermionic.json", "w") as f:
        json.dump({"pool": pool_raw}, f, indent=4)
        
    print(f"Terminé. Génération de pool_fermionic.json avec {len(pool_raw)} vrais opérateurs de chimie quantique.")

if __name__ == "__main__":
    main()