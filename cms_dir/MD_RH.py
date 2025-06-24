from pymatgen.core import Lattice, Structure, Molecule, Element, Composition


def AcrossB(a, b):
    c = [0.0 for i in range(3)]
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


def AdotB(a, b):
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return c


def volume(a, b, c):
    x = AcrossB(b, c)
    v = AdotB(a, x)
    return v


def R2K(x):
    # x = x[3][3]
    y = [[0 for i in range(3)] for j in range(3)]
    v = volume(x[0], x[1], x[2])
    b1 = AcrossB(x[1], x[2])
    b2 = AcrossB(x[2], x[0])
    b3 = AcrossB(x[0], x[1])
    for i in range(3):
        b1[i] = b1[i] / v
        b2[i] = b2[i] / v
        b3[i] = b3[i] / v
    y[0] = b1
    y[1] = b2
    y[2] = b3
    return y


def rotation_matrix(x, y, z, ranx, rany, ranz):
    import numpy as np
    thx = ranx * np.pi / 180
    thy = ranx * np.pi / 180
    thz = ranx * np.pi / 180
    length = len(x)
    xn = []
    for i in range(length):
        xr1 = x[i] * np.cos(thz) + y[i] * np.sin(thz)
        yr1 = -x[i] * np.sin(thz) + y[i] * np.cos(thz)
        zr1 = z[i]
        xr2 = xr1 * np.cos(thy) - zr1 * np.sin(thy)
        yr2 = yr1
        zr2 = xr1 * np.sin(thy) + zr1 * np.cos(thy)
        xr1 = xr2
        yr1 = yr2 * np.cos(thx) + zr2 * np.sin(thx)
        zr1 = -yr2 * np.sin(thx) + zr2 * np.cos(thx)
        xn.append([xr1, yr1, zr1])
    return (xn)


def draw_ternary(pts, ele, string):
    # https://github.com/marcharper/python-ternary
    import matplotlib
    import ternary
    matplotlib.rcParams['figure.dpi'] = 200
    matplotlib.rcParams['figure.figsize'] = (4, 4)

    # scales
    figure, tax = ternary.figure(scale=1.0)
    # boundary
    tax.boundary(linewidth=1.0)
    tax.gridlines(color="grey", multiple=0.1)
    # tax.gridlines(color="blue", multiple=2, linewidth=0.5)

    # labels and title
    # 57
    fontsize = 12
    # tax.set_title("ternary phase diagram\n", fontsize=fontsize)
    tax.set_title(" ", fontsize=fontsize)
    tax.left_axis_label("$x_C$", fontsize=fontsize, offset=0.14)
    tax.right_axis_label("$x_B$", fontsize=fontsize, offset=0.14)
    tax.bottom_axis_label("$x_A$", fontsize=fontsize, offset=0.14)

    # ticks
    tax.ticks(
        axis='lbr',
        linewidth=1,
        multiple=0.2,
        tick_formats="%.1f",
        offset=0.03)

    # text
    figure.text(.89, 0.12, ele[0], fontsize=12)
    figure.text(.48, 0.93, ele[1], fontsize=12)
    figure.text(.02, 0.15, ele[2], fontsize=12)

    # plot data
    pdata = []
    for pt in pts:
        mm = pt[:3]
        pdata.append(1.0 * mm / sum(mm))
    tax.scatter(pdata, color='red', s=8.0)

    # remove matplotlib axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    # figure.tight_layout()

#    ternary.plt.show()
    ternary.plt.savefig(string + ".png")


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


def draw_binary_convex(pts, ele, string):
    from scipy.spatial import ConvexHull
    import numpy as np
    import pylab as plt
    from matplotlib.ticker import MultipleLocator

    apts = []
    apts_names = []
    unstables = []
    unstables_names = []
    idx = 0
    for ipt in pts:
        if (ipt[2] <= 0):
            apts.append([ipt[1] * 1.0 / (ipt[0] + ipt[1]), ipt[2]])
            apts_names.append(ele[0] + str(int(ipt[0])) +
                              ele[1] + str(int(ipt[1])))
            idx += 1
        else:
            unstables.append([ipt[1] / (ipt[0] + ipt[1]), ipt[2]])
            unstables_names.append(
                ele[0] + str(int(ipt[0])) + ele[1] + str(int(ipt[1])))
    # get stables
    hull = ConvexHull(apts)
    stables = []
    stables_names = []
    print("stable structures:", len(hull.vertices))
    for iv in hull.vertices:
        #        stables.append(apts[iv])
        #        stables_names.append(apts_names[iv])
        stables.append([apts[iv][0], apts[iv][1], apts_names[iv]])
        print(pts[iv])
    stables_sorted = sorted(stables, key=lambda entry: entry[0])
    for vvv in stables_sort:
        print(vvv)
    # get metastables
    mstables = []
    mstables_names = []
    # print("metastable:")
    for i in range(len(apts)):
        if (i not in hull.vertices):
            mstables.append(apts[i])
            mstables_names.append(apts_names[iv])
        #    print(apts[i])
    # draw
    plt.figure(figsize=(8, 6))
    mstables_plt = np.array(mstables)
    unstables_plt = np.array(unstables)
    # meta_stable
    if (len(mstables) != 0):
        plt.plot(mstables_plt[:, 0], mstables_plt[:, 1],
                 "x", color="g", markersize=6)

    # unstable
    if (len(unstables) != 0):
        plt.plot(unstables_plt[:, 0], unstables_plt[:, 1],
                 "x", color='r', markersize=6)

    # stables
    stables_plt = np.array(stables_sort, dtype=object)
    plt.plot(stables_plt[:, 0], stables_plt[:, 1],
             "o-", color='k', markersize=6)
    # print(len(stables))
    for kkk in range(len(stables)):
        plt.annotate(
            stables_plt[kkk, 2], (stables_plt[kkk, 0], stables_plt[kkk, 1]), size=12)

    # dash line 0 to 1
    dash = np.array([[0, 0], [1, 0]])
    plt.plot(dash[:, 0], dash[:, 1], '--', color='k')

    fc = 16
    xr = [k * 0.1 for k in range(0, 11, 1)]
    ir = np.array(xr)
    xr = ["%.1f" % (k * 0.1) for k in range(0, 11, 1)]
    xr[0] = ele[0]
    xr[-1] = ele[1]
    plt.xticks(ir, xr, fontsize=fc)
    plt.yticks(fontsize=fc)
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.grid(axis="x", which="major")
    plt.ylabel(r'$E_f\ (eV/atom)$', fontsize=fc)
    plt.xlabel(r'$x_{' + ele[1] + '}$', fontsize=fc)
    plt.tight_layout()
    plt.savefig(string + ".png")
    plt.show()


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


def draw_ternary_old(pts, pts_aga, pts_exp, pts_mp, pts_l0, ele, string):
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

    # labels and title
    # 57
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
        comp = ipt[:3]
        comp = comp / sum(comp)
        x = comp[0] + comp[1] / 2.
        y = comp[1] * np.sqrt(3) / 2
        tpts.append([x, y, ipt[3]])
    for ipt in pts_exp:
        comp = ipt[:3]
        comp = comp / sum(comp)
        x = comp[0] + comp[1] / 2.
        y = comp[1] * np.sqrt(3) / 2
        mpts.append([x, y, ipt[3]])
    tpts_l0 = []
    for ipt in pts_l0:
        comp = ipt[:3]
        comp = comp / sum(comp)
        x = comp[0] + comp[1] / 2.
        y = comp[1] * np.sqrt(3) / 2
        tpts_l0.append([x, y, ipt[3]])

    # get convex hull
    # hll.vertices is the indice of stable compounds in pts,
    # hull.simplices is the connections among stable compounds
    hull = ConvexHull(mpts)
    fout = open("./convex-hull.dat", "w+")
    print("# of stable structures", len(hull.vertices), ":", end=" ", file=fout)
    fout.write("\n")
    print(*ele, "Ef(eV/atom)", end=" ", file=fout)
    fout.write("\n")
    stables = []
    for iv in hull.vertices:
        print(
            "%2d%3d%3d%12.6f" %
            (int(
                pts[iv][0]), int(
                pts[iv][1]), int(
                pts[iv][2]), pts[iv][3]), end=" ", file=fout)
        fout.write("\n")
        name = ele[0] + str(int(pts[iv][0])) + ele[1] + \
            str(int(pts[iv][1])) + ele[2] + str(int(pts[iv][2]))
        # still not sure how to plot names on the figure 06/24
        stables.append([mpts[iv][0], mpts[iv][1], name])

    # plot data
    pdata = []
    # print(pts)
    for pt in pts_exp:
        mm = pt[:3]
        pdata.append(1.0 * mm / sum(mm))

    for isimp in hull.simplices:
        tax.line(pdata[isimp[0]], pdata[isimp[1]], linewidth=1.,
                 marker='.', markersize=8., color='black')
        tax.line(pdata[isimp[0]], pdata[isimp[2]], linewidth=1.,
                 marker='.', markersize=8., color='black')
        tax.line(pdata[isimp[1]], pdata[isimp[2]], linewidth=1.,
                 marker='.', markersize=8., color='black')

    mstables = []
    for i in range(len(pdata)):
        if (i not in hull.vertices):
            mstables.append(pdata[i])
    '''
    #3 plot meta-stable phases
    mstables=np.array(mstables)
    if(len(mstables) != 0):
        tax.scatter(mstables, color='blue', marker='^', s=12.0, linewidth=1.5)
    '''

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
            print("%2d%3d%3d%12.6f%11.6f" % (int(pts[k][0]), int(pts[k][1]), int(pts[k][2]),
                  pts[k][3], h), end=" ", file=fout)
            fout.write("\n")
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

    # 3_color plot meta-stable phases
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
        print("%2d%3d%3d%12.6f%11.6f" % (int(pts_l0[ka][0]), int(pts_l0[ka][1]), int(pts_l0[ka][2]),
              pts_l0[ka][3], h), end=" ", file=fout)
        fout.write("\n")

    all_meta_stables = [mp_meta_stables, l0_meta_stables]
    fout.close()

    marker_vec = [6, 7, ".", "."]
    s_vec = [50, 50, 50, 50]
    all_color_data = []
    all_meta_data = []

    cm = ternary.plt.cm.get_cmap('rainbow')
    for ms in range(len(all_meta_stables)):
        all_meta_stables[ms] = np.array(all_meta_stables[ms])
        if (len(all_meta_stables[ms]) != 0):
            meta_data = []
            color_data = []
            for mpt in all_meta_stables[ms]:
                mm = mpt[:3]
                point_t = 1.0 * mm / sum(mm)
                meta_data.append(point_t)
                all_meta_data.append(point_t)
                color_data.append(mpt[-1])
                all_color_data.append(mpt[-1])

    tax.scatter(
        all_meta_data,
        s=7,
        colormap=cm,
        vmin=0,
        vmax=0.5,
        colorbar=True,
        c=all_color_data,
        cmap=cm)
    # remove matplotlib axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    ternary.plt.savefig(string + "_newhull.png", bbox_inches='tight')
    ternary.plt.show()
