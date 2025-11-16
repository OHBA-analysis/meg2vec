"""Get representations using the trained model."""

import numpy as np
from modules.data import load_fif
from modules.ts2vec import TS2Vec

n_timestamps = 500
output_dims = 32
batch_size = 128

# Load data
files = [
    "data/mf2pt2_sub-CC110033_ses-rest_task-rest_megtransdef.fif",
    "data/mf2pt2_sub-CC110037_ses-rest_task-rest_megtransdef.fif",
    "data/mf2pt2_sub-CC110045_ses-rest_task-rest_megtransdef.fif",
]
data = load_fif(
    files,
    n_timestamps=n_timestamps,
    zscore=True,
    n_jobs=4,
)
print("data.shape =", data.shape)

# Load model
model = TS2Vec(
    input_dims=data.shape[2],
    output_dims=output_dims,
    batch_size=batch_size,
    device="cuda:0",
)
model.load("output/model.pt")

# Get representations
repr_ts = model.encode(data)
repr_inst = model.encode(data, encoding_window="full_series")

# Save
np.save("output/repr_ts.npy", repr_ts)
np.save("output/repr_inst.npy", repr_inst)

print("repr_ts.shape =", repr_ts.shape)
print("repr_inst.shape =", repr_inst.shape)
