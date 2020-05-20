# crystal-gnn

These use the environment /projects/rlmolecule/pstjohn/envs/tf (let me know if you have read issues).
Major packages there include tensorflow 2.2, pymatgen, and `nfp`.

nfp is the package I put together, available at [nrel/nfp](https://github.com/nrel/nfp). Note this project uses the tf2 branch, which is not currently pip-installable. 

To re-create the inputs and trained model, first run `preprocess_crystals.py`, then `train_model.py`.

I ran `preprocess_crystals.py` on the dav/login nodes (I'm bad), but submitted `train_model.py` via SLURM using `gpu_submit.sh`.

One note is that the GPU compute nodes have a different driver version than the DAV nodes currently, and therefore can only run tensorflow 2.0. (CUDA compute capacity <= 10.0)
