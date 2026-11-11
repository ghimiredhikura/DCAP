# Documentation

Start with **[Setup](setup.md)**, then choose a task:

| Guide | Use it to |
|---|---|
| [Reproduction](reproduction.md) | Replay statistics, evaluate released checkpoints or retrain the main comparison |
| [Data](data.md) | Install processed chips, rebuild source data and retain required attribution |
| [Deployment and benchmark](benchmark.md) | Replay target-device results or measure GPU inference with the released ensembles |
| [Validation](validation.md) | Check what was tested and the limits of those checks |
| [Release](release.md) | Find assets, checksums and source-packaging instructions |

For a quick review, follow **Route 1** in the reproduction guide: no model or full-dataset download is needed. **Route 2** checks predictions from the released weights; **Route 3** retrains the main comparison.

Exact tested versions are recorded in [`tested_environment.json`](tested_environment.json). Run commands from the repository root with your environment activated.
