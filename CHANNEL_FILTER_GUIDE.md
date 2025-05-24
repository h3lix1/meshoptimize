# Channel Filter Feature Guide

## Overview
The Meshtastic Network Optimizer now includes a channel filter feature that allows you to analyze and optimize networks based on specific Meshtastic channels.

## Features Added

### 1. Channel Filter Dropdown
- Located in the "Optimization Controls" panel
- Shows "All Channels" by default
- Automatically populates with available channels from loaded data
- Common channels include: MediumFast, MediumSlow, LongFast, LongSlow, VeryLongSlow

### 2. Enhanced Node Information on Hover
When you hover over a node in the visualization, you'll now see:
- **Node Label** (bold at the top)
- **ID**: Node identifier
- **Role**: Router or Client
- **Channel**: The Meshtastic channel this node uses
- **Hardware**: Device type (e.g., T-Beam, Heltec V3, RAK4631)
- **Short Name**: Abbreviated node name
- **Long Name**: Full descriptive name
- **Connections**: Number of connected nodes
- **Connected to**: Names of up to 5 connected nodes
- **Optimization Scores**: If optimization has been run

### 3. Visual Improvements
- **Larger node sizes**: Minimum size increased from 5 to 8 pixels for better visibility
- **Better scaling**: Node size scales more dramatically with connections
- **Color Legend**: Shows what each node color represents:
  - Blue: Router
  - Green: Client
  - Red: Isolated node
  - Yellow: Vulnerable (only 1 connection)
  - Purple: Proposed for demotion
  - Orange: Proposed for promotion

### 4. Channel-Based Analysis
All analysis and optimization functions now respect the channel filter:
- **Network Health**: Shows statistics only for the selected channel
- **Optimization**: Only optimizes nodes within the selected channel
- **Node Scores**: Calculated based on the filtered network

## How to Use

1. **Load Network Data**: Use "Load Data" button or File menu
2. **Select Channel**: Choose a channel from the dropdown
   - The visualization will update to show only nodes on that channel
   - Edge connections are filtered to show only intra-channel links
3. **Analyze**: View network health metrics for the selected channel
4. **Optimize**: Run optimization on the filtered network
5. **Reset**: Use "Reset View" to return to showing all channels

## Use Cases

- **Multi-Channel Networks**: Analyze each channel separately to ensure optimal coverage
- **Channel Migration**: Identify which nodes should move to different channels
- **Isolated Analysis**: Find network issues specific to certain channels
- **Performance Testing**: Compare network efficiency across different channel configurations

## Tips

- Hover over any node to see detailed information
- Use the pan/zoom controls to navigate large networks
- Export visualizations with channel-specific views
- The status bar shows how many nodes/edges are currently displayed 