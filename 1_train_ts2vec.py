"""Train TS2Vec."""

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
    n_timestamps=500,
    zscore=True,
    n_jobs=4,
)
print("data.shape =", data.shape)

# Train TS2Vec
model = TS2Vec(
    input_dims=data.shape[2],
    output_dims=output_dims,
    batch_size=batch_size,
    device="cuda:0",
)
loss_log = model.fit(data, verbose=True)

# Save
model.save("output/model.pt")
np.save("results/loss_log.npy", loss_log)
