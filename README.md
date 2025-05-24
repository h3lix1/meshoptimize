# Meshtastic Network Optimizer

A Python GUI application for optimizing router placement in Meshtastic mesh networks. This tool helps identify which nodes should be routers and which should be clients to maximize network efficiency while maintaining redundancy.

## Features

- **Real-time Network Visualization**: Interactive graph visualization showing nodes and their connections
- **Router Optimization**: Automatically identify redundant routers and suggest optimizations
- **Scoring System**: Customizable scoring algorithms for evaluating node importance
- **Redundancy Control**: Configurable redundancy levels (2-3 connections per node)
- **Hop Count Analysis**: Ensures all nodes can reach each other within 5 hops
- **Interactive GUI**: Sliders and controls for real-time network adjustments
- **Color-coded Visualization**: Visual indicators for node types and connection quality
- **Performance Optimization**: Multi-threaded processing for large networks
- **Live Data Fetching**: Can download and analyze data from meshview.bayme.sh

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python mesh_optimizer.py
```

### GUI Controls

- **Redundancy Slider**: Set the desired redundancy level (2-3 connections)
- **Max Hops Slider**: Maximum allowed hops between any two nodes (default: 5)
- **Coverage Weight**: Importance of node coverage in scoring
- **Centrality Weight**: Importance of network centrality
- **Redundancy Weight**: Importance of connection redundancy
- **Load Data**: Fetch latest data from meshview.bayme.sh
- **Optimize**: Run the optimization algorithm
- **Export**: Save optimization results

### Color Coding

- **Blue**: Router nodes
- **Green**: Client nodes
- **Yellow**: Nodes with single connection (vulnerable)
- **Red**: Disconnected nodes or problematic links
- **Orange**: Suggested router promotions
- **Purple**: Suggested router demotions

## Architecture

The application consists of several modules:

- `mesh_optimizer.py`: Main application entry point and GUI
- `network_analyzer.py`: Core network analysis algorithms
- `data_fetcher.py`: Handles data retrieval from Meshtastic sources
- `visualization.py`: Network visualization components
- `scoring_engine.py`: Node scoring and evaluation algorithms
- `optimization_worker.py`: Background workers for optimization tasks

## Algorithms

### Node Scoring

Nodes are scored based on:
1. **Coverage**: Number of unique nodes reachable
2. **Centrality**: Betweenness centrality in the network
3. **Redundancy**: Connection redundancy provided
4. **Hop Reduction**: Impact on average hop count
5. **Critical Path**: Presence in critical network paths

### Optimization Strategy

1. Identify redundant routers with overlapping coverage
2. Find clients that would improve network connectivity as routers
3. Ensure minimum redundancy requirements are met
4. Minimize hop count while reducing router count
5. Maintain network connectivity for all nodes

## Requirements

- Python 3.8+
- PyQt6 for GUI
- NetworkX for graph algorithms
- Matplotlib/Plotly for visualization
- See `requirements.txt` for full list