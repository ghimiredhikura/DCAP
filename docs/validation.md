# Local release validation

Checked on Ubuntu with one RTX 5090 selected for GPU inference. Exact package versions are in [tested_environment.json](tested_environment.json). This record concerns release correctness, not the pending RTX 4080/Jetson Orin manuscript measurements.

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

The full 63-model campaign was not retrained as a packaging test. The recheck changed packaging, documentation and run-integrity checks; published numerical evidence, checkpoints and processed rasters are unchanged. Its training, inference and analysis components were exercised as above; shortened training outputs are excluded from distributed scientific results. Remote preparation was checked on one chip, while the full mask lineage and all packaged raster hashes were checked locally. Cross-machine training trajectories and remote source reprocessing are not guaranteed to be bitwise identical. The supplied checkpoints, frozen raster assets and compact statistics are the direct route to reproducing the reported values.
