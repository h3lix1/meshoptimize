# Meshtastic Network Optimizer - Usage Guide

## Quick Start

1. **Launch the Application**
   ```bash
   ./run_optimizer.sh
   ```
   Or directly:
   ```bash
   source venv/bin/activate
   python mesh_optimizer.py
   ```

2. **Load Network Data**
   - Click "Load Data" to fetch from meshview.bayme.sh
   - Or use File → Load Data from File for local JSON files
   - Sample data loads automatically on startup for testing

3. **Run Optimization**
   - Adjust settings (redundancy, weights) as needed
   - Click "Optimize Network" (green button)
   - Review results in the tabs

4. **Apply Changes**
   - Review recommendations
   - Click "Apply Changes" if satisfied
   - Export results for later use

## Interface Overview

### Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│                    Menu Bar                             │
├─────────────────────────────────────────────────────────┤
│                 Control Panel                           │
│  [Redundancy] [Max Hops] [Scoring Weights] [Actions]   │
├──────────────────────────┬──────────────────────────────┤
│                          │                              │
│    Network Graph         │    Information Tabs         │
│    Visualization         │  • Network Health           │
│                          │  • Node Scores              │
│                          │  • Recommendations          │
│                          │  • Simulation               │
│                          │                              │
└──────────────────────────┴──────────────────────────────┘
```

### Control Panel

#### Redundancy Slider (1-5)
- Sets minimum router connections each node should have
- Default: 2 (recommended for reliability)
- Higher values increase network resilience but need more routers

#### Max Hops Spinner (3-10)
- Maximum hops allowed between any two nodes
- Default: 5 (Meshtastic standard)
- Lower values require more routers

#### Scoring Weights (0.0-2.0)
- **Coverage**: How many nodes a router can reach
- **Centrality**: How important for shortest paths
- **Redundancy**: Alternative paths provided
- **Hop Reduction**: Impact on reducing hop counts
- **Critical Path**: Presence in critical connections

#### Action Buttons
- **Load Data**: Fetch fresh data from meshview.bayme.sh
- **Optimize Network**: Run optimization algorithm
- **Apply Changes**: Apply proposed router changes
- **Reset View**: Clear highlights and selections

### Network Visualization

#### Node Colors
- 🔵 **Blue**: Router nodes
- 🟢 **Green**: Client nodes
- 🟡 **Yellow**: Vulnerable (only 1 connection)
- 🔴 **Red**: Isolated (no connections)
- 🟠 **Orange**: Client promotion candidates
- 🟣 **Purple**: Router demotion candidates

#### Node Interaction
- Click nodes to select and see details
- Selected nodes show black border
- Node size indicates connection count

#### Edge Appearance
- Thickness: Connection strength/weight
- Color: Signal quality (SNR-based)
  - Dark gray: Strong signal (SNR > 0)
  - Light gray: Normal signal
  - Light red: Weak signal (SNR < -10)

### Information Tabs

#### Network Health Tab
Shows overall network statistics:
- Node and edge counts
- Router/client distribution
- Connectivity metrics
- Problem nodes (isolated, vulnerable)

#### Node Scores Tab
Sortable table showing:
- Node ID and label
- Current role (Router/Client)
- Individual score components
- Total optimization score

Click table rows to highlight nodes in the graph.

#### Recommendations Tab
Optimization suggestions:
- Routers to demote (redundant)
- Clients to promote (beneficial)
- Network warnings
- Improvement suggestions

#### Simulation Tab
Shows before/after comparison:
- Router count changes
- Path length impact
- Connectivity changes
- Summary of improvements

## Optimization Process

### 1. Understanding the Algorithm

The optimizer evaluates each node based on:

1. **Coverage Score**: How many unique nodes it can reach
2. **Centrality Score**: How often it appears in shortest paths
3. **Redundancy Score**: Alternative paths it provides
4. **Hop Reduction**: How much it reduces average hop count
5. **Critical Path**: Whether removing it disconnects the network

### 2. Adjusting Parameters

**For Dense Networks (many routers):**
- Increase redundancy requirement (3-4)
- Increase centrality weight
- Focus on reducing router count

**For Sparse Networks (few routers):**
- Lower redundancy requirement (1-2)
- Increase coverage weight
- Focus on maintaining connectivity

**For Long Networks (high hop counts):**
- Increase hop reduction weight
- Consider more router promotions
- Monitor average path length

### 3. Interpreting Results

**Good Optimization:**
- ✓ Reduced router count
- ✓ Maintained/improved average path length
- ✓ No new isolated nodes
- ✓ Maintained connectivity

**Warning Signs:**
- ⚠ Increased vulnerable nodes
- ⚠ Higher average path length
- ⚠ New isolated nodes
- ⚠ Disconnected components

## Advanced Features

### Network Monitoring
Tools → Start Network Monitor
- Checks for network changes every 60 seconds
- Alerts when topology changes
- Useful for dynamic networks

### Interactive Visualization
Tools → Open Interactive View
- Creates HTML file with Plotly
- Zoom, pan, hover for details
- Better for large networks

### Data Export/Import

**Export Options:**
- Results: JSON with recommendations
- Visualization: PNG or PDF image
- Interactive: HTML with full interactivity

**Import Options:**
- JSON files with nodes/edges structure
- Previously exported network data

## Workflow Examples

### Example 1: Reducing Router Count

1. Load network data
2. Set redundancy to 2
3. Increase centrality weight to 1.5
4. Run optimization
5. Review routers marked for demotion
6. Check simulation shows no connectivity loss
7. Apply changes

### Example 2: Improving Coverage

1. Load network data
2. Check Network Health for isolated nodes
3. Increase coverage weight to 1.8
4. Run optimization
5. Review promotion candidates
6. Verify they connect isolated areas
7. Apply changes

### Example 3: Balancing Network

1. Load network data
2. Keep all weights at 1.0
3. Set redundancy to 3
4. Run optimization
5. Review balanced recommendations
6. Export results for team review
7. Apply after consensus

## Troubleshooting

### Application Won't Start
- Ensure Python 3.8+ installed
- Check all dependencies installed
- Try: `pip install -r requirements.txt`

### Can't Load Data from URL
- Check internet connection
- Verify meshview.bayme.sh is accessible
- Try loading sample data first

### Optimization Takes Too Long
- Large networks may take time
- Reduce node count if possible
- Use sampling for initial analysis

### Visualization Issues
- Update matplotlib: `pip install --upgrade matplotlib`
- Check PyQt6 installation
- Try interactive view instead

## Best Practices

1. **Start Conservative**: Begin with default settings
2. **Incremental Changes**: Apply a few changes at a time
3. **Monitor Impact**: Check network health after changes
4. **Document Decisions**: Export results for records
5. **Test First**: Use simulation before applying
6. **Regular Reviews**: Re-optimize as network grows

## Tips for Large Networks

- Use interactive visualization for better performance
- Focus on specific problem areas
- Export data and work offline
- Consider clustering similar nodes
- Run optimization in batches

## Network Patterns

### Hub and Spoke
- Few central routers serve many clients
- High centrality weight
- Low redundancy acceptable

### Mesh Grid
- Evenly distributed routers
- Balanced weights
- Medium redundancy (2-3)

### Linear Chain
- Routers form backbone
- High hop reduction weight
- Critical path protection important

### Clustered Groups
- Dense local clusters
- Inter-cluster routers critical
- Variable redundancy by area 