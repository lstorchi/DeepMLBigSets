import math

import numpy as np

######################################################################################

def check_float(potential_float):

    try:
        float(potential_float)
        return True
    except ValueError:
        return False

######################################################################################


def readlabel(path, files, label="BBB", filtertu=True):

    from rdkit.Chem import PandasTools

    frame = PandasTools.LoadSDF(path+files,
                                smilesName='SMILES',
                                molColName='Molecule',
                                includeFingerprints=False)
    name = files.split(".")[0]
    basename = name.replace("_0", "")

    value = None
    if filtertu:
        value = int(float(frame[label][0]))
        if value == -1:
            value = 0
    else:
        if check_float(frame[label][0]):
            value = float(frame[label][0])
        else:
            print("Error for ", frame[label], file=sys.stderr)

    return basename, value

######################################################################################


def readfeature(path, files, cn):

    data = np.load(path+files)
    lst = data.files

    mol_channel = data["channels"]
    mol_coordinates = data["coords"]

    num_rows, num_cols = mol_coordinates.shape

    dimx = dimy = dimz = int(round(math.pow(float(num_rows), float(1/3))))

    treedobject = np.zeros((dimx, dimy, dimz, cn), dtype=np.float32)

    coords_to_idx = {}
    i = 0
    for iz in range(dimz):
        for iy in range(dimy):
            for ix in range(dimx):
                k = "%f_%f_%f" % (mol_coordinates[i][0],
                                  mol_coordinates[i][1],
                                  mol_coordinates[i][2])
                coords_to_idx[k] = (ix, iy, iz)
                #print(k, ix, iy, iz, i)
                i = i + 1

    for i in range(0, num_rows):
        try:
            k = "%f_%f_%f" % (mol_coordinates[i][0],
                              mol_coordinates[i][1],
                              mol_coordinates[i][2])
            ix, iy, iz = coords_to_idx[k]
            treedobject[ix, iy, iz] = np.float32(mol_channel[i])
            #print( mol_channel[i])
        except:
            return None, None, None, None,

    return treedobject, dimx, dimy, dimz

######################################################################################
