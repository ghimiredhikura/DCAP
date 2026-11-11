# Local release validation

Checked on Ubuntu with one RTX 5090 selected for GPU inference. Exact package versions are in [tested_environment.json](tested_environment.json). This record concerns release correctness, not the pending RTX 4080/Jetson Orin manuscript measurements.

## Completed checks

| Check | Outcome |
|---|---|
| Data installation and SHA-256 verification | All three archives and 2,640 rasters passed; 1,320 chips with 990/132/198 partitions |
| Model installation | All six tensor-only checkpoints passed catalog checks |
| Compact statistical replay | All main endpoint values, DCAP paired intervals and all 72 sensitivity rows match the frozen evidence |
| Released-weight validation/test inference | Dense and DCAP AP, Dice, small-component recall and boundary F1 match exactly; absolute Brier differences <2×10⁻⁹ |
| Fresh-checkpoint threshold path | Re-fitting DCAP validation thresholds matches the stored global and block thresholds; test consumes that validation result |
| Inventory-to-mask reconstruction | All 1,320 masks match pixel for pixel, including georeferencing |
| Live source preparation | One chip downloaded from public Sentinel-2/DEM assets; mask matches exactly; image maximum absolute difference 7.63×10⁻⁶ |
| Dense training and resume | Two-epoch checks on eight fitting/four validation chips matched with and without interruption; restart before the first saved epoch also passed |
| DCAP construction and recovery | Full 18-epoch gate learning, physical removal, training-only inheritance/BN selection and one recovery epoch completed; measured MAC reduction 70.0003% |
| Ubuntu inference benchmark | All 12 model/mode/batch combinations passed CPU parity and changed-input checks; full timing protocol passed initially; all 12 functional configurations passed after cleanup |
| Automated regression suite | Twelve checks passed in both the CUDA host environment and a fresh isolated CPU environment, including scientific replay, split/threshold provenance, checkpoint/statistics integrity and package exclusions |
| Figures | Figures 6–8 regenerated as PDF/PNG; rendering inspected |
| Fresh environment | Python 3.13 venv without system site packages; official CPU PyTorch 2.13.0, pinned dependencies and editable checkout installed; `pip check`, replay and plots passed |
| Campaign collection | Complete collection and figure generation checked with synthetic completed-run fixtures; test fixtures are excluded from the release |
| Source quality and distribution | Static checks passed; unused helpers/duplicate imports removed; manifest and portable-checkout verification passed |

## What these checks do not establish

- **Full retraining:** the 63-model campaign was not repeated as a packaging test. Training, inference and collection were checked separately; synthetic/smoke outputs are excluded from scientific evidence.
- **Complete source reprocessing:** live downloading/rebuilding was checked on one chip. All packaged raster hashes and all 1,320 reconstructed masks were checked locally.
- **Bitwise portability:** training trajectories and upstream source reprocessing can vary across machines or provider updates.
- **Target-device performance:** RTX 4080 and Jetson Orin measurements remain pending.

Use the supplied checkpoints, frozen rasters and compact statistics for direct replay of the reported values. See the [reproduction routes](reproduction.md).
