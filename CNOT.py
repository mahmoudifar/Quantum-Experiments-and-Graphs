import sympy as sp
from TranslateGraphToExperiment import PostSelection, Sort

Paths = ['a', 'b', 'c', 'd', 'V_a', 'V_b', 'p', 'p1', 'p2', 'p3', 'p4']
a, b, c, d, V_a, V_b, p, p1, p2, p3, p4 = map(sp.IndexedBase, Paths)

theta, alpha, phi, beta, gamma, eta, ommega = sp.symbols(
    'theta alpha phi beta gamma eta ommega', integer=True 
)

V, H, D, A = sp.symbols('V H D A', cls=sp.Idx)

l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = map(sp.Wild, [f'l{i}' for i in range(1, 11)])

def PBS(psi, p1, p2):
    psi = psi.subs({
        p1[H]: p2[H],
        p1[V]: p1[V],
        p2[H]: p1[H],
        p2[V]: p2[V]
    }, simultaneous=True)
    return psi

def PBSDA(psi, p1, p2):
    psi = psi.subs({
        p1[D]: p2[D],
        p1[A]: p1[A],
        p2[D]: p1[D],
        p2[A]: p2[A]
    }, simultaneous=True)
    return psi

def TODA(psi, p):
    psi = psi.subs({
        p[H]: 1 / sp.sqrt(2) * (p[D] + p[A]),
        p[V]: 1 / sp.sqrt(2) * (p[D] - p[A])
    }, simultaneous=True)
    return psi

def TOHV(psi, p):
    psi = psi.subs({
        p[D]: 1 / sp.sqrt(2) * (p[H] + p[V]),
        p[A]: 1 / sp.sqrt(2) * (p[H] - p[V])
    }, simultaneous=True)
    return psi

def EPR(p1, p2):
    return {
        'phi+': 1 / sp.sqrt(2) * p1[H] * p2[H] + 1 / sp.sqrt(2) * p1[V] * p2[V],  
        'phi-': 1 / sp.sqrt(2) * p1[H] * p2[H] - 1 / sp.sqrt(2) * p1[V] * p2[V],  
        'psi+': 1 / sp.sqrt(2) * p1[H] * p2[V] + 1 / sp.sqrt(2) * p1[V] * p2[H],  
        'psi-': 1 / sp.sqrt(2) * p1[H] * p2[V] - 1 / sp.sqrt(2) * p1[V] * p2[H] 
    }

def get_EPR_state(state_name, p1, p2):
    states = EPR(p1, p2)
    return states.get(state_name, None)

def CNOT(psi, p1, p2, p3, p4):
    # Apply a series of transformations which represent:
    # - PBS: Polarizing Beam Splitter
    # - TODA: Transformation from H/V to D/A basis
    # - TOHV: Transformation from D/A to H/V basis
    # - PBSDA: Another polarizing beam splitter operation in a D/A basis
    # This sequence simulates the quantum circuit for a CNOT operation
    psi = sp.expand(
        TODA(
            TOHV(
                TOHV(
                    PBSDA(
                        TODA(
                            TODA(
                                PBS(psi, p1, p3),  
                                p2               
                            ),
                            p4            
                        ),
                        p2, p4           
                    ),
                    p2             
                ),
                p4               
            ),
            p1              
        )
    )

    # Post-selection step, to select only the terms that match the pattern (p1[l1] * p2[l2] * p3[l3] * p4[l4])
    # where li are the polarizations of the photons
    psi = PostSelection(psi, Sort(psi))
    
    # Collect terms based on the polarization states of the ancilla photons
    return psi.collect([p1[D] * p2[H], p1[D] * p2[V], p1[A] * p2[H], p1[A] * p2[V]])


def QuantumParityCheck(psi, p1, p2):
    psi = sp.expand(
        TODA(
            PBS(psi, p1, p2),
            p1
        )
    )
    return psi


def DestructiveCNOT(psi, p1, p2):
    psi = sp.expand(
        TOHV(
            TOHV(
                PBSDA(
                    TODA(
                        TODA(psi, p1),
                        p2
                    ),
                    p1, p2
                ),
                p1
            ),
            p2
        )
    )
    return psi