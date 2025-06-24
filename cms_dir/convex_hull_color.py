#!/usr/bin/env python3

"""
This is a basic example of how to create, plot, and analyze Phase Diagrams using the pymatgen
codebase and Materials Project database. To run this example, you should:
* have pymatgen (www.pymatgen.org) installed along with matplotlib
* obtain a Materials Project API key (https://www.materialsproject.org/open)
* paste that API key in the MAPI_KEY variable below, e.g. MAPI_KEY = "foobar1234"
For citation, see https://www.materialsproject.org/citing
For the accompanying comic book, see http://www.hackingmaterials.com/pdcomic
"""
import os
import sys
import csv
import MD_RH
from pymatgen.core import Lattice, Structure, Molecule, Element, Composition
from pymatgen.io.cif import CifWriter
import argparse

system = []  # system we want to get PD for
ene = []

# Read elemental energies from mp_element.dat


def read_elemental_energies(filename):
    elemental_energies = {}
    with open(filename, 'r') as f:
        for line in f:
            element, energy = line.replace(',', ' ').split()
            try:
                eles = Composition(element).elements
                if len(eles) == 1:
                    elemental_energies[eles[0].symbol] = float(energy)
            except BaseException:
                continue
    return elemental_energies


def read_mp(file_in):
    processed_entries = []
    ef_large0 = []
    with open(file_in, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            formula = line.split()[0]
            comp = Composition(formula)
            formreduce = comp.reduced_formula
            natom_1 = int(comp.element_composition.get(system[0]))
            natom_2 = int(comp.element_composition.get(system[1]))
            natom_3 = int(comp.element_composition.get(system[2]))
            et = float(line.split()[1])
            natom = natom_1 + natom_2 + natom_3
            ef = et - (natom_1 * ene[0] + natom_2 *
                       ene[1] + natom_3 * ene[2]) / natom
            my_entry = [natom_1, natom_2, natom_3, ef]
            processed_entries.append(my_entry)
    return processed_entries, ef_large0


def read_all(file_in):
    processed_entries = []
    ef_large0 = []
    with open(file_in, "r") as fin:
        lines = csv.reader(fin)
        for line in lines:
            comp = Composition(line[0])

            natom_1 = int(comp.element_composition.get(system[0]))
            natom_2 = int(comp.element_composition.get(system[1]))
            natom_3 = int(comp.element_composition.get(system[2]))

            et = float(line[1])
            natom = natom_1 + natom_2 + natom_3
            ef = et - (natom_1 * ene[0] + natom_2 *
                       ene[1] + natom_3 * ene[2]) / natom
            my_entry = [natom_1, natom_2, natom_3, ef]
            if (ef > 0):
                ef_large0.append(my_entry)
                continue
            elif ef < -1:
                continue
            processed_entries.append(my_entry)
    return processed_entries, ef_large0


def main():
    global system, ene

    parser = argparse.ArgumentParser(
        description='Calculate formation energies relative to convex hull.')
    parser.add_argument("--elements", type=str, required=True, help="Elements")
    parser.add_argument('--stable', required=True, type=str,
                        help='Path to the stable phases file (e.g., mp_int_stable.dat)')
    parser.add_argument('--input', required=True, type=str,
                        help='Path to the calculated results CSV file')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Maximum Ehull (eV/atom) to display for metastable phases (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file name (e.g., hull_plot.png). If not provided, shows plot interactively.')

    args = parser.parse_args()
    elements = [Element(ele) for ele in args.elements.split('-')]
    eles = [ele.symbol for ele in elements]

    elename = ''.join(eles)
    system.append(eles[2])
    system.append(eles[0])
    system.append(eles[1])

    # Create phase diagram!
    ef_file = args.stable
    elemental_energies = read_elemental_energies(ef_file)
    ene = [float(elemental_energies[i]) for i in system]

    mp_file = args.input
    ef_l0 = []
    pre_xyze, ef_l0 = read_mp(ef_file)
    mp_xyze, ef_l0 = read_all(mp_file)

    if args.threshold <= 0:
        mp_xyze = []
    all_xyze = mp_xyze

    for j in range(len(pre_xyze)):
        all_xyze.append(pre_xyze[j])

    aga_xyze = []
    exp_xyze = []

    MD_RH.draw_ternary_convex(
        all_xyze,
        aga_xyze,
        pre_xyze,
        mp_xyze,
        ef_l0,
        system,
        elename,
        args.threshold,
        args.output)


if __name__ == "__main__":
    main()
