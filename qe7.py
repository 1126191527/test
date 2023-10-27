# Import the modules
import xml.etree.ElementTree as ET
import configparser
import numpy as np
import sys
import os
import os.path

# Parse the vasprun.xml file
tree = ET.parse("vasprun.xml")
root = tree.getroot()

# Get the lattice vectors
lattice = root.find("structure/crystal/varray[@name='basis']")
lattice_vectors = np.array([[float(x) for x in v.text.split()] for v in lattice.findall("v")])

# Get the atomic positions in fractional coordinates
positions = root.find("structure/varray[@name='positions']")
atomic_positions_frac = np.array([[float(x) for x in v.text.split()] for v in positions.findall("v")])

# Get the atomic types
atoms = root.find("atominfo/array[@name='atoms']")
atomic_types = [rc.find("c").text for rc in atoms.findall("set/rc")]
    
# Get the number of atoms and types
nat = len(atomic_types)
ntyp = len(set(atomic_types))
# Get the system name
system = root.find("parameters/separator[@name='general']/i[@name='SYSTEM']").text

# Define a function to convert fractional coordinates to cartesian coordinates using the lattice vectors
def frac2cart(frac_coords, lattice_vectors):
    # Convert the frac_coords and lattice_vectors to numpy arrays
    frac_coords = np.array(frac_coords)
    lattice_vectors = np.array(lattice_vectors)
    # Calculate the cartesian coordinates by matrix multiplication
    cart_coords = np.dot(frac_coords, lattice_vectors)
    return cart_coords

# Convert the atomic positions from fractional to cartesian coordinates using the function defined before
atomic_positions_cart = frac2cart(atomic_positions_frac, lattice_vectors)

# Define a function to generate the k-points list using the nscfkmesh value
def generate_kpoints_list(nscfkmesh):
    # Parse the nscfkmesh value into three integers
    numargs = len(nscfkmesh.split())

    if numargs != 3:
        print("usage: n1 n2 n3")
        print("       n1  - divisions along 1st recip vector")
        print("       n2  - divisions along 2nd recip vector")
        print("       n3  - divisions along 3rd recip vector")
        sys.exit()

    # Check if the nscfkmesh value has single quotes, and strip them if yes
    if nscfkmesh.startswith("'") and nscfkmesh.endswith("'"):
        nscfkmesh = nscfkmesh.strip("'")

    # Convert the nscfkmesh value to three integers
    n1, n2, n3 = [int(x) for x in nscfkmesh.split()]

    if n1 <= 0:
        print("n1 must be >0")
        sys.exit()
    if n2 <= 0:
        print("n2 must be >0")
        sys.exit()
    if n3 <= 0:
        print("n3 must be >0")
        sys.exit()

    # Calculate the total number of points
    totpts = n1 * n2 * n3

    # Generate the k-points list as a string
    kpoints_list = "{}\ncrystal\n".format(totpts)
    for x in range(1, n1 + 1):
        for y in range(1, n2 + 1):
            for z in range(1, n3 + 1):
                kpoints_list += "{:.8f} {:.8f} {:.8f} {:.6e}\n".format((x - 1) / n1, (y - 1) / n2, (z - 1) / n3, 1 / totpts)

    return kpoints_list

# Define a new delimiter for the ETC file
config = configparser.ConfigParser(delimiters=":")
config.read("ETC")
pseudo_dir = config.get("pwscf", "pseudo_dir")
atomic_species_block = config.get("pwscf", "ATOMIC_SPECIES")
atomic_species_lines = atomic_species_block.split("\n")
pseudo_dict = {}
for line in atomic_species_lines:
    if not line:
        continue
    parts = line.split()
    if len(parts) == 3:
        name, mass, pseudopotential = parts
        pseudo_dict[name] = (float(mass), pseudopotential)

k_points_block = config.get("pwscf", "K_POINTS {automatic}")

# Define a function to get all the parameters from the ETC file, or use some default values if not provided
def get_parameters(config):
    # Initialize an empty dictionary to store the parameters
    parameters = {}
    # Try to get each parameter from the ETC file, or use a default value if not provided
    try:
        parameters["ecutwfc"] = config.getfloat("pwscf", "ecutwfc")
    except configparser.NoOptionError:
        parameters["ecutwfc"] = 84.0 # default value
    
    try:
        parameters["conv_thr"] = config.getfloat("pwscf", "conv_thr")
    except configparser.NoOptionError:
        parameters["conv_thr"] = 1.0e-12 # default value
    
    try:
        parameters["mixing_beta"] = config.getfloat("pwscf", "mixing_beta")
    except configparser.NoOptionError:
        parameters["mixing_beta"] = 0.7 # default value
    
    try:
        parameters["nscfkmesh"] = config.get("pwscf", "nscfkmesh")
    except configparser.NoOptionError:
        parameters["nscfkmesh"] = "6 6 6" # default value
    
    try:
        parameters["occupations"] = config.get("pwscf", "occupations")
    except configparser.NoOptionError:
        parameters["occupations"] = "'smearing'" # default value
    
    try:
        parameters["degauss"] = config.getfloat("pwscf", "degauss")
    except configparser.NoOptionError:
        parameters["degauss"] = 0.001 # default value
    
    try:
        parameters["smearing"] = config.get("pwscf", "smearing")
    except configparser.NoOptionError:
        parameters["smearing"] = "'gaussian'" # default value

    return parameters

# Get all the parameters from the ETC file using the function defined before
parameters = get_parameters(config)

# Define a function to generate the QE input file content
def generate_qe_input(calculation, pseudo_dir, pseudo_dict, k_points, parameters):
    # Define the parameters for QE calculation
    parameters_qe = {
        "CONTROL": {
            "calculation": "'{}'".format(calculation),
            "prefix": "'{}'".format(system),
            "pseudo_dir": "'{}'".format(pseudo_dir),
            "outdir": "'./'",
            "tprnfor" : ".true.",
            "tstress" : ".true.",
        },
        "SYSTEM": {
            "ibrav": 0,
            "nat": nat,
            "ntyp": ntyp,
            "ecutwfc": parameters["ecutwfc"],
            "occupations": parameters["occupations"],
            "degauss": parameters["degauss"],
            "smearing": parameters["smearing"]
        },
        "ELECTRONS": {
            "conv_thr": parameters["conv_thr"],
            "mixing_beta": parameters["mixing_beta"]
        }
    }

    # Generate the content for QE input file
    content = ""

    for name, param in parameters_qe.items():
        content += "&" + name + "\n"
        for key, value in param.items():
            content += "   {} = {}\n".format(key, value)
        content += "/\n"

    content += "CELL_PARAMETERS {angstrom}\n"
    for v in lattice_vectors:
        content += "   {:.10f} {:.10f} {:.10f}\n".format(*v)

    content += "ATOMIC_POSITIONS {angstrom}\n"
    for t, p in zip(atomic_types, atomic_positions_cart):
        content += "{} {:.10f} {:.10f} {:.10f}\n".format(t, *p)

    # Add the ATOMIC_SPECIES section to the content
    print(pseudo_dict)
    print(pseudo_dir)
    content += "ATOMIC_SPECIES\n"
    for name, value in pseudo_dict.items():
        mass, pseudo_file = value
        # Append the information to the string with proper formatting
        content += "{} {:.2f} {}\n".format(name, mass, pseudo_file)
    content += "\n" 
    content += "K_POINTS {automatic}\n"

    content += k_points
    content += "\n" 

    return content

# Get the ecutwfc value from the ETC file
ecutwfc = config.getfloat("pwscf", "ecutwfc")
# Get the conv_thr value from the ETC file, or use a default value of 1.0e-12 if not provided
conv_thr = config.getfloat("pwscf", "conv_thr", fallback=1.0e-12)
# Get the mixing_beta value from the ETC file, or use a default value of 0.7 if not provided
mixing_beta = config.getfloat("pwscf", "mixing_beta", fallback=0.7)
# Try to get the nscfkmesh value from the ETC file, use the default value 6 6 6 if not provided
nscfkmesh = config.get("pwscf", "nscfkmesh", fallback="6 6 6")

# Generate the content for scf.in file using the function defined before
scf_content = generate_qe_input("scf", pseudo_dir, pseudo_dict, k_points_block, parameters)

# Generate the content for nscf.in file using the function defined before
nscf_content = generate_qe_input("nscf", pseudo_dir, pseudo_dict, k_points_block, parameters)

# Generate the k-points list using the function defined before
kpoints_list = generate_kpoints_list(nscfkmesh)

# Replace the K_POINTS block with the k-points list in the nscf_content
nscf_content = nscf_content.replace(k_points_block, kpoints_list)

# Check if the files already exist
scf_file = "scf.in"
nscf_file = "nscf.in"

if os.path.exists(scf_file) and os.path.exists(nscf_file):
    print("The input files for QE already exist and are no longer automatically generated.")
    sys.exit()

# Write the files
with open(scf_file, "w") as f:
    f.write(scf_content)

with open(nscf_file, "w") as f:
    f.write(nscf_content)

print("The input files for QE have been generated successfully.")
