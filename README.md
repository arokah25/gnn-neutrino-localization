# Neutrino Interaction Localization with EdgeConv Graph Neural Networks

This project explores the use of Graph Neural Networks (GNNs), specifically Dynamic Edge Convolution layers, to reconstruct the position of neutrino interactions in a 2D simulated detector environment. The model is trained to predict the interaction vertex using only the spatial and temporal distribution of detected Cherenkov light pulses.

---

## Dataset

The dataset simulates a simplified IceCube-like detector: a 4×4 grid of sensors in two dimensions. Each event contains a variable number of photon hits, each described by:

- Detection time  
- (x, y) position of the sensor

Files are provided in `.parquet` format and include training, validation, and test splits. Data is not included in this repository due to size but can be downloaded [via this link](#) and placed in a `data/` directory.

---

## Model Architecture

The model uses the [DynamicEdgeConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.DynamicEdgeConv.html) layer from PyTorch Geometric to dynamically build local graphs from hit features in each event. The network consists of:

- Two EdgeConv layers with shared MLPs  
- Global mean pooling across graph nodes  
- Final MLP mapping to `[xpos, ypos]` regression output  

Features are normalized globally across the training set, and predictions are denormalized before evaluation.

---

## Results

- **Test RMSE** (real-world units):
  - `xpos`: 1.23 meters
  - `ypos`: 1.57 meters

These results are based on evaluation over 10,000 unseen test events. Most predictions are within 2 meters of the true interaction point.

### Example visualizations:

<p align="center">
  <img src="plots/pred_vs_true_xy.png" width="550"/>
  <br><em>True vs. Predicted x and y positions</em>
</p>

<p align="center">
  <img src="plots/position_overlay.png" width="480"/>
  <br><em>Overlay of predicted and true (x, y) positions</em>
</p>

More plots are available in the [plots/](plots/) directory.

---

## Usage

### Requirements

- Python 3.9+
- torch==2.5.0
- torch_geometric
- torch_cluster
- numpy, scikit-learn, matplotlib, awkward

Install dependencies:

```bash
pip install -r requirements.txt
