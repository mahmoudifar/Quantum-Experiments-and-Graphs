import sympy as sp 
import numpy as np
from functools import reduce
import itertools

I = sp.I
sqrt = sp.sqrt
zero = sp.Symbol('zero') 
num_mode = 7

Paths = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'p']

ALLPaths = [f'{prefix}{i}' for i in range(0, num_mode + 1) for prefix in Paths] + Paths
          

for op in ALLPaths: # op: optical path
    globals()[op] = sp.IndexedBase(op)

n, m, l, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = map(sp.Wild, ['n', 'm', 'l', 'l1',
                                 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10']
                                                        )

theta, alpha, phi = sp.symbols('theta alpha phi', integer=True)

Labels =  [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]

def SetupToStr(setup):
    # A string created by replacing 'psi' with elements from the setup list in reverse order.
    setupstr = 'psi'
    for element in reversed(setup):
        setupstr = setupstr.replace('psi', element)
    return setupstr

def EncodedLabel(nums, labels):
    # Map each index in nums to its corresponding label in labels
    encoded_labels = [labels[i] for i in nums]
    return encoded_labels

def GetNumLabel(labels):
    num_to_label = dict((num, label) for num, label in enumerate(labels))
    return num_to_label

def Grouper(n, iterable):
    args = [iter(iterable)] * n
    return list(zip(*args))

def Sort(psi, labels=Labels):
   
    # Extract free symbols from the expression
    symbols = list(psi.free_symbols)
    
    # Identify indexed symbols and collect their bases
    bases = [sym.base for sym in symbols if isinstance(sym, sp.tensor.indexed.Indexed)]
    
    # Remove duplicates to identify unique bases
    unique_bases = list(set(bases))
    
    # Map numeric indices to labels
    mapped_labels = EncodedLabel(range(len(unique_bases)), labels)
    
    # Create a mapping between the unique bases and their corresponding labels
    mapping = list(zip(unique_bases, mapped_labels))
    
    # Construct the sorted expression 
    sorted_expression = [mapping[i][0][mapping[i][1]] for i in range(len(mapping))]
    
    sorted_result = reduce(lambda x, y: x * y, sorted_expression)
    
    return sorted_result

def PostSelaction(psi, selected_term):

    collected_terms = sp.collect(psi, [selected_term], evaluate=False)

    # Extract terms and coefficients from the dictionary
    terms, coefficients = zip(*collected_terms.items())

    # Replace coefficients corresponding to terms equal to 1
    coefficients = [0 if term == 1 else coeff for term, coeff in zip(terms, coefficients)]

    # Reconstruct the state after post-selection
    selection_list = [term * coeff for term, coeff in zip(terms, coefficients)]
    psi_out = sp.expand(sum(selection_list))

    return psi_out

# Define optical devices 

def HalfWavePlate(psi, p, n=1, dim=2):
    psi = psi.replace(p[l], lambda l: p[(l + n) % dim])
    return psi

def Absorber(psi, p):
    psi = psi.replace(p[l], 0)
    return psi

def PhaseShifter(psi, p, phi):
    psi = psi.replace(p[l], lambda l: sp.exp(I * l * phi) * p[l])
    return psi 

def OAMHologram(psi, p, n):
    psi = psi.replace(p[l], lambda l: p[l + n])
    return psi

def BS_Fun(psi, p1, p2):
    if psi.base == p1:
        psi = psi.replace(p1[l], 1 / sqrt(2) * (p2[l] + I * p1[l]))
    elif psi.base == p2:
        psi = psi.replace(p2[l], 1 / sqrt(2) * (p1[l] + I * p2[l]))
    return psi

def BeamSplitter(psi, p1, p2):
    # Extract all indexed symbols in the expression
    indexed_symbols = [sym for sym in psi.free_symbols if isinstance(sym, sp.Indexed)]

    # Filter symbols belonging to p1 or p2
    relevant_symbols = [
        sym for sym in indexed_symbols if sym.base == p1 or sym.base == p2
    ]

    replacements = {
        sym: BS_Fun(sym, p1, p2) for sym in relevant_symbols
    }
    psi = sp.expand(psi.xreplace(replacements))

    return psi

def PolarisingBeamSplitter(psi, p1, p2):
    psi = psi.subs(
        {
            p1[0]: p2[0],
            p1[1]: p1[1],
            p2[0]: p1[0],
            p2[1]: p2[1]
        },
        simultaneous=True
    )
    return psi

# Spontaneous Parametric Down-Conversion
def SPDC(psi, p1, p2, l1, l2):
    psi = psi + p1[l1] * p2[l2]
    return psi

#  Converts a graph representation to Entanglement by path identity     
def GraphtoEbPI(Graph, Paths=Paths):
   
    dictt = dict()

    # Extract graph edges and dimensions
    GraphEdges = [Grouper(2, i)[0] for i in list(Graph.keys())]
    GraphEdgesAlphabet = [EncodedLabel(path, GetNumLabel(Paths)) for path in GraphEdges]
    Dimension = [Grouper(2, i)[1] for i in list(Graph.keys())]
    #Dim = len(np.unique(list(itertools.chain(*Dimension))))
    NumMode = len(np.unique(list(itertools.chain(*GraphEdgesAlphabet))))

   # Create the setup list
    SetupList = []
    for pp in range(len(Graph)):
        SetupList.append(
            f"SPDC(psi, {GraphEdgesAlphabet[pp][0]}, {GraphEdgesAlphabet[pp][1]}, "
            f"{Dimension[pp][0]}, {Dimension[pp][1]})"
        )

    dictt['Experiment'] = SetupList
    dictt['SetupLength'] = len(SetupList)

    # Generate the output state using post-selection
    setup = SetupToStr(SetupList)
    state = sp.expand(eval(setup.replace('psi', str(0))) ** int(NumMode / 2))
    dictt['OutputState'] = PostSelaction(state, Sort(state))

    return dictt


# Converts a graph representation to a path-encoded setup for an on-chip quantum system.
def GraphtoPathEn(Graph, Paths=Paths):
   
    dictt = {}

    # Extract graph edges and dimensions
    GraphEdges = [Grouper(2, i)[0] for i in list(Graph.keys())]
    GraphEdgesAlphabet = [EncodedLabel(path, GetNumLabel(Paths)) for path in GraphEdges]
    Dimension = [Grouper(2, i)[1] for i in list(Graph.keys())]

    #Add SPDC to the setup list
    SetupList = [
        f"SPDC(psi, {GraphEdgesAlphabet[pp][0]}{pp}, {GraphEdgesAlphabet[pp][1]}{pp}, "
        f"{Dimension[pp][0]}, {Dimension[pp][1]})"
        for pp in range(len(Graph))
    ]

    # Gather all paths and dimensions
    AllPath = [f"{GraphEdgesAlphabet[pp][j]}{pp}" for pp in range(len(Graph)) for j in [0, 1]]
    AllDim = [str(Dimension[pp][j]) for pp in range(len(Graph)) for j in [0, 1]]

    # Generate possible combinations of paths and dimensions
    PossiblePath = list(itertools.combinations(AllPath, 2))
    PossibleDim = list(itertools.combinations(AllDim, 2))
    combine = list(zip(PossiblePath, PossibleDim))

    # Add the beam splitter and absorber to the setup list
    for pd in combine:
        path_combination = pd[0]
        dim_combination = pd[1]
        if path_combination[0][0] == path_combination[1][0] and dim_combination[0] == dim_combination[1]:
            SetupList.append(f"BeamSplitter(psi, {path_combination[0]}, {path_combination[1]})")
            SetupList.append(f"Absorber(psi, {path_combination[1]})")

    # Generate the output setup string
    setup = SetupToStr(SetupList)

    # Populate the dictionary with results
    dictt['Experiment'] = SetupList
    dictt['SetupLength'] = len(SetupList)
    dictt['OutputState'] = sp.expand(eval(setup.replace('psi', str(0))))

    return dictt

# Converts a graph representation to a polarisation-encoding for a bulk optics 
def GraphtoPolEn(expr):
    """
    Converts a quantum experiment setup (a path-encoded setup) to polarization encoding for bulk optics.

    The GraphtoPolEN  is correct for generating a polarization encoding setup,
    provided that the degrees of freedom (DOF) you're working with are indeed
    polarization (degrees: [0, 1]).

    Parameters:
    expr : dict
        A dictionary containing the initial setup ('Experiment') and quantum state ('OutputState').

    Returns:
    dict
        A dictionary containing the updated experiment setup, setup length, and the new output quantum state.
    """
    dictt = {}

    SetupList = expr['Experiment']
    psi = expr['OutputState']

    # Extract paths and dimensions from free symbols
    symbols = list(psi.free_symbols)
    path = []
    dimension = []
    for symbol in symbols:
        if isinstance(symbol, sp.tensor.indexed.Indexed):
            path.append(str(symbol.base))
            dimension.append(str(symbol.indices[0]))

    # Generate possible combinations of paths and dimensions
    PossiblePath = list(itertools.combinations(path, 2))
    PossibleDim = list(itertools.combinations(dimension, 2))
    combine = list(zip(PossiblePath, PossibleDim))
    combination = [combine[i][0] + combine[i][1] for i in range(len(combine))]

    # Add polarising beam splitter to the setup based on conditions
    for pd in range(len(combination)):
        if combination[pd][0][0] == combination[pd][1][0] and combination[pd][2] != combination[pd][3]:
            SetupList.append(f"PolarisingBeamSplitter(psi, {combination[pd][0]}, {combination[pd][1]})")

    # Generate the setup string and calculate the new output state
    setup = SetupToStr(SetupList)
    dictt['Experiment'] = SetupList
    dictt['SetupLength'] = len(SetupList)
    dictt['OutputState'] = sp.expand(eval(setup.replace('psi', str(0))))

    return dictt

Graph = {
    (0, 1, 0, 0): 1,
    (0, 1, 1, 1): 1,
    (1, 2, 1, 1): 1,
    (2, 3, 0, 0): 1,
    (0, 3, 1, 0): 1
}