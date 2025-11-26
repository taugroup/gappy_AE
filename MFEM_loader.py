import glob
import meshio
import numpy as np

# adjust this pattern if your files are .vtu
files = sorted(glob.glob("ex9_*.vtk"))

snapshots = []
points = None
cells = None

for f in files:
    m = meshio.read(f)

    # take the first scalar field; print keys the first time
    if not snapshots:
        print("point data keys:", m.point_data.keys())

    # many MFEM examples call it "u"
    if "u" in m.point_data:
        field = m.point_data["u"]
    else:
        # fall back to the first one
        key = list(m.point_data.keys())[0]
        field = m.point_data[key]

    snapshots.append(field.astype(np.float32))

    if points is None:
        points = m.points
        cells = m.cells

X = np.stack(snapshots, axis=0)   # shape: (T, Npoints)

print("dataset shape:", X.shape)
np.save("mfem_ex9_dataset.npy", X)
