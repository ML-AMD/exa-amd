#!/usr/bin/env python3
import os
import sys
import csv
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


# input: pts as list with element [nA,nB,nC,Ef]
def area(a, b, c):
    import numpy as np
    from numpy.linalg import norm
    if (isinstance(a, list)):
        a = np.array(a, dtype=np.float32)
    if (isinstance(b, list)):
        b = np.array(b, dtype=np.float32)
    if (isinstance(c, list)):
        c = np.array(c, dtype=np.float32)
    return 0.5 * norm(np.cross(b - a, c - a))


def get_plane(p1, p2, p3):
    import numpy as np
    if (isinstance(p1, list)):
        p1 = np.array(p1, dtype=np.float32)
    if (isinstance(p2, list)):
        p2 = np.array(p2, dtype=np.float32)
    if (isinstance(p3, list)):
        p3 = np.array(p3, dtype=np.float32)
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d


def draw_ternary_convex(pts, pts_aga, pts_exp, pts_mp,
                        pts_l0, ele, string, hullmax=0.1, output_file=None):
    import matplotlib
    import ternary
    import numpy as np
    from scipy.spatial import ConvexHull
    matplotlib.rcParams['figure.dpi'] = 200
    matplotlib.rcParams['figure.figsize'] = (4, 4)

    # scales
    figure, tax = ternary.figure(scale=1.0)
    # boundary
    tax.boundary(linewidth=0.5)
    tax.gridlines(color="grey", multiple=0.1)

    fontsize = 12
    ter = ele[0] + "-" + ele[1] + "-" + ele[2]
    tax.right_corner_label(ele[0], fontsize=fontsize + 2)
    tax.top_corner_label(ele[1], fontsize=fontsize + 2)
    tax.left_corner_label(ele[2], fontsize=fontsize + 2)

    # convert data to trianle set
    pts = np.array(pts)
    pts_aga = np.array(pts_aga)
    pts_exp = np.array(pts_exp)
    pts_mp = np.array(pts_mp)
    pts_l0 = np.array(pts_l0)
    tpts = []
    mpts = []
    for ipt in pts:
        comp = np.array([int(ii) for ii in ipt[:3]])
        comp = comp / sum(comp)
        x = comp[0] + comp[1] / 2.
        y = comp[1] * np.sqrt(3) / 2
        tpts.append([x, y, float(ipt[3])])

    tpts_l0 = []
    for ipt in pts_l0:
        comp = np.array([int(ii) for ii in ipt[:3]])
        comp = comp / sum(comp)
        x = comp[0] + comp[1] / 2.
        y = comp[1] * np.sqrt(3) / 2
        tpts_l0.append([x, y, float(ipt[3])])

    comps = []
    ehulls = []

    hull = ConvexHull(tpts)
    fout = open("./convex-hull.dat", "w+")
    print("# of stable structures", len(hull.vertices), ":", end=" ", file=fout)
    fout.write("\n")
    print(*ele, "Ef(eV/atom)", end=" ", file=fout)
    fout.write("\n")
    # plot data
    pdata = []

    for pt in pts:
        mm = np.array([int(ii) for ii in pt[:3]])
        pdata.append(1.0 * mm / sum(mm))

    # 1 plot stable and connect them
    for isimp in hull.simplices:
        tax.line(pdata[isimp[0]], pdata[isimp[1]], linewidth=0.7,
                 marker='.', markersize=8., color='black')
        tax.line(pdata[isimp[0]], pdata[isimp[2]], linewidth=0.7,
                 marker='.', markersize=8., color='black')
        tax.line(pdata[isimp[1]], pdata[isimp[2]], linewidth=0.7,
                 marker='.', markersize=8., color='black')

    stables = []
    stables_extra = []
    for iv in hull.vertices:
        name = ele[0] + str(int(pts[iv][0])) + ele[1] + \
            str(int(pts[iv][1])) + ele[2] + str(int(pts[iv][2]))
        aaa = pts[iv]
        # still not sure how to plot names on the figure 06/24
        stables.append([tpts[iv][0], tpts[iv][1], name])
        name = ele[0] + str(int(pts[iv][0])) + ele[1] + \
            str(int(pts[iv][1])) + ele[2] + str(int(pts[iv][2]))

        matches_first_three = np.all(np.isclose(
            pts_exp[:, :3], aaa[:3], atol=1e-4), axis=1)
        matches_fourth = np.isclose(pts_exp[:, 3], aaa[3], atol=1e-1)
        # if not np.any(np.all(np.isclose(pts_exp, aaa, atol=1e-1), axis=1)):
        if not np.any(matches_first_three & matches_fourth):
            tax.scatter([pdata[iv]], marker='.', s=64., color='red', zorder=10)
            comps.append(iv)
            ehulls.append(0)
        else:
            formula = Composition(name).reduced_formula
            fout.write(formula + '\n')

    # 2 get meta-stable phases
    mstables = []
    for i in range(len(pdata)):
        if (i not in hull.vertices):
            mstables.append(pdata[i])

    aga_meta_stables = []
    exp_meta_stables = []
    mp_meta_stables = []
    l0_meta_stables = []
    # 4 find the distance to the convex hull
    print("# of metastable structures", len(mstables), ":", end=" ", file=fout)
    fout.write("\n")
    print(*ele, "Ef(eV/atom) E_to_convex_hull(eV/atom)", end=" ", file=fout)
    fout.write("\n")
    #  4.1 get nearest 3 points
    for k in range(len(tpts)):
        if (k in hull.vertices):
            h = 0
            # continue # jump the stable ones
        else:
            x = tpts[k][:2]  # metastable, as [x,y,Ef]
            for isimp in hull.simplices:  # loop the simplices
                A = tpts[isimp[0]][:2]
                B = tpts[isimp[1]][:2]
                C = tpts[isimp[2]][:2]
                # find if x in the A-B-C triangle
                area_ABC = area(A, B, C)
                sum_a = area(A, B, x) + area(A, C, x) + area(B, C, x)
                if (sum_a - area_ABC <= 0.001):
                    # in the ABC, get the ABC plane
                    a, b, c, d = get_plane(
                        tpts[isimp[0]], tpts[isimp[1]], tpts[isimp[2]])
                    if (a == 0 and b == 0 and d == 0):
                        continue
                    if (c == 0):
                        continue
                    # get the cross point with ABC plane
                    z = (d - a * x[0] - b * x[1]) / c
                    # height to convex hull
                    h = tpts[k][2] - z

            name = ele[0] + str(int(pts[k][0])) + ele[1] + \
                str(int(pts[k][1])) + ele[2] + str(int(pts[k][2]))
            formula = Composition(name).reduced_formula
            comps.append(k)
            ehulls.append(h)
        # judge the label for aga, exp and mp
        label = "000"
        for ss in range(len(pts_aga)):
            if (pts[k][-1] == pts_aga[ss][-1]):
                label = "aga"
                aga_meta_stables.append([float(pts[k][0]), float(
                    pts[k][1]), float(pts[k][2]), pts[k][3], h])
                break
        for ss in range(len(pts_exp)):
            if (pts[k][-1] == pts_exp[ss][-1]):
                label = "exp"
                exp_meta_stables.append([float(pts[k][0]), float(
                    pts[k][1]), float(pts[k][2]), pts[k][3], h])
        for ss in range(len(pts_mp)):
            if (pts[k][-1] == pts_mp[ss][-1]):
                label = "mp"
                mp_meta_stables.append([float(pts[k][0]), float(
                    pts[k][1]), float(pts[k][2]), pts[k][3], h])

    for ka in range(len(tpts_l0)):
        x = tpts_l0[ka][:2]  # metastable, as [x,y,Ef]
        for isimp in hull.simplices:  # loop the simplices
            A = tpts[isimp[0]][:2]
            B = tpts[isimp[1]][:2]
            C = tpts[isimp[2]][:2]
            # find if x in the A-B-C triangle
            area_ABC = area(A, B, C)
            sum_a = area(A, B, x) + area(A, C, x) + area(B, C, x)
            if (sum_a - area_ABC <= 0.001):
                # in the ABC, get the ABC plane
                a, b, c, d = get_plane(
                    tpts[isimp[0]], tpts[isimp[1]], tpts[isimp[2]])
                if (a == 0 and b == 0 and d == 0):
                    continue
                if (c == 0):
                    continue
                # get the cross point with ABC plane
                z = (d - a * x[0] - b * x[1]) / c
                # height to convex hull
                h = tpts_l0[ka][2] - z
                l0_meta_stables.append([float(pts_l0[ka][0]), float(pts_l0[ka][1]), float(pts_l0[ka][2]),
                                        pts_l0[ka][3], h])

    all_meta_stables = [
        aga_meta_stables,
        exp_meta_stables,
        mp_meta_stables,
        l0_meta_stables]
    if hullmax > 0:
        ehulls, comps = zip(*sorted(zip(ehulls, comps)))
        for eh, kkk in zip(ehulls, comps):
            aaa = pts[kkk]
            name = ele[0] + str(int(aaa[0])) + ele[1] + \
                str(int(aaa[1])) + ele[2] + str(int(aaa[2]))
            formula = Composition(name).reduced_formula

            fout.write(formula + '   ' + str(eh * 1000) + '\n')

    fout.close()

    marker_vec = [6, 7, ".", "."]
    s_vec = [50, 50, 50, 50]
    all_color_data = []
    all_meta_data = []
    # cm = ternary.plt.cm.get_cmap('tab20c')
    cm = ternary.plt.cm.get_cmap('rainbow')

    for ms in range(len(all_meta_stables)):
        # all_meta_stables[ms]=np.array(all_meta_stables[ms])
        if (len(all_meta_stables[ms]) != 0):
            meta_data = []
            color_data = []
            for mpt in all_meta_stables[ms]:
                if mpt[-1] < hullmax:
                    mm = np.array([float(ii) for ii in mpt[:3]])
                    point_t = 1.0 * mm / sum(mm)
                    meta_data.append(point_t)
                    all_meta_data.append(point_t)
                    color_data.append(mpt[-1])
                    all_color_data.append(mpt[-1])

    if hullmax > 0:
        tax.scatter(
            all_meta_data,
            s=7,
            marker='s',
            colormap=cm.reversed(),
            vmin=0,
            vmax=hullmax,
            colorbar=False,
            c=all_color_data,
            cmap=cm.reversed())

    # remove matplotlib axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    ternary.plt.show()

    if output_file:
        ternary.plt.savefig(output_file, dpi=300)
    else:
        ternary.plt.show()


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

    draw_ternary_convex(
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
