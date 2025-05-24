# Enhanced Interactive Visualization Guide

This guide explains the new enhanced interactive visualization features that make your mesh network much clearer and easier to analyze.

## 🚀 Key Improvements

### **Visual Clarity**
- **Multiple Layout Algorithms**: Choose from 6 different layout algorithms to find the best view for your network
- **Critical Path Highlighting**: Automatically identify and highlight the most important connections
- **Intelligent Filtering**: Filter nodes by type, connection count, and other properties
- **Zoom-Aware Labeling**: Node labels that adapt to zoom level for better readability

### **Enhanced Interactivity**
- **Detailed Tooltips**: Rich hover information including SNR, connection quality, and network metrics
- **Node Selection**: Click to select nodes and see detailed information in a dedicated panel
- **Real-time Controls**: Adjust layout parameters, filtering, and display options on-the-fly
- **Multiple View Modes**: Pan, zoom, and select modes for different interaction needs

## 🎛️ Control Panel Features

### **Layout Options**
Choose from these advanced layout algorithms:

1. **Spring Enhanced** (Default)
   - Best for most networks
   - Uses node weights and connection quality
   - Balances node distribution and readability

2. **Force Atlas**
   - Great for large networks
   - Emphasizes network structure
   - Good separation between clusters

3. **Circular Clustered**
   - Groups related nodes in circles
   - Excellent for community detection
   - Clear cluster visualization

4. **Hierarchical**
   - Shows network hierarchy based on importance
   - Router nodes at top levels
   - Good for understanding network structure

5. **Community Based**
   - Organizes nodes by detected communities
   - Grid-based community layout
   - Helps identify network segments

6. **Edge Bundled**
   - Reduces visual clutter in dense networks
   - Groups similar connections
   - Cleaner appearance for complex networks

**Layout Controls:**
- **Iterations Slider**: Control layout quality vs. speed (10-500 iterations)
- ****: Adjust spacing between nodes (0.5x - 3.0x)

### **Display Filters**
Fine-tune what you see:

**Node Type Filters:**
- ☐ Show Routers (blue nodes)
- ☐ Show Clients (green nodes)  
- ☐ Show Isolated (red nodes)
- ☐ Show Vulnerable (yellow nodes - only 1 connection)

**Connection Filters:**
- **Min Connections**: Hide nodes with fewer connections
- **Max Connections**: Hide nodes with more connections
- **Critical Paths**: Highlight most important network paths

### **Node Labels**
- **Show Node Labels**: Toggle node text display
- **Label Size**: Adjust text size (8-24pt) for better readability

### **Actions**
- **Recalculate Layout**: Apply current settings to regenerate layout
- **Find Critical Paths**: Identify and highlight critical network connections
- **Center on Selected**: Focus view on selected nodes
- **Export HTML**: Save interactive visualization for sharing

## 🖱️ Interaction Guide

### **Node Interaction**
- **Single Click**: Select node and show details in sidebar
- **Double Click**: Center view on that node
- **Hover**: Show detailed tooltip with connection info

### **View Controls**
- **Pan Mode**: Drag to move around the network
- **Zoom Mode**: Scroll or drag to zoom in/out
- **Select Mode**: Draw selection box to select multiple nodes

### **Keyboard Shortcuts**
- **Scroll**: Zoom in/out
- **Drag**: Pan (in pan mode)
- **Box Select**: Select multiple nodes (in select mode)

## 📊 Node Information Display

### **Tooltip Information**
When you hover over a node, you'll see:
- **Basic Info**: Name, ID, type (Router/Client)
- **Hardware**: Device model and channel
- **Connections**: Number and quality of connections
- **Connection List**: Up to 5 connected nodes with SNR values
- **Network Metrics**: Centrality and optimization scores

### **Node Details Panel**
Selected nodes show detailed information including:
- Complete connection list with signal quality
- Optimization scores breakdown
- Network analysis metrics
- Hardware and configuration details

## 🎨 Visual Legend

### **Node Colors**
- 🔵 **Blue**: Router nodes (mesh repeaters)
- 🟢 **Green**: Client nodes (end devices)  
- 🔴 **Red**: Isolated nodes (no connections)
- 🟡 **Yellow**: Vulnerable nodes (only 1 connection)
- 🟠 **Orange**: Highlighted/selected nodes

### **Node Sizes**
Node size indicates importance:
- **Base Size**: All nodes have minimum readable size
- **Connection Bonus**: +2 pixels per connection
- **Score Bonus**: +10 pixels per optimization score point

### **Edge Styles**
- **Thin Gray**: Normal connections
- **Thick Orange**: Critical path connections (when enabled)
- **Edge Opacity**: Based on connection quality/SNR

## 🔍 Advanced Features

### **Critical Path Detection**
Click "Find Critical Paths" to:
- Identify the most important connections in your network
- Highlight paths between high-importance nodes
- Show which connections are essential for network connectivity
- Help prioritize which nodes need the most attention

### **Filtering for Focus**
Use filters to reduce visual complexity:
- **Hide isolated nodes** when troubleshooting connectivity
- **Show only routers** to understand backbone structure  
- **Filter by connection count** to find over/under-connected nodes
- **Critical path mode** to see only essential connections

### **Layout Optimization**
Different algorithms work better for different scenarios:
- **Small networks (< 30 nodes)**: Spring Enhanced or Circular
- **Medium networks (30-100 nodes)**: Force Atlas or Community Based
- **Large networks (100+ nodes)**: Hierarchical or Community Based
- **Dense networks**: Edge Bundled to reduce clutter

## 🛠️ Troubleshooting Tips

### **Performance**
- For large networks (100+ nodes), use fewer iterations (50-100)
- Try Community Based or Hierarchical layouts for better performance
- Use filtering to reduce displayed nodes

### **Visual Clarity**
- Increase node separation if nodes overlap
- Try different layout algorithms if structure isn't clear
- Use critical path highlighting to focus on important connections
- Adjust label size based on zoom level

### **Network Analysis**
- Look for isolated (red) nodes that need connection
- Identify vulnerable (yellow) nodes that need redundancy
- Use critical path highlighting to find essential connections
- Check node details for poor SNR connections

## 📤 Export and Sharing

### **HTML Export**
The "Export HTML" feature creates a standalone file that:
- Contains all the interactivity of the main application
- Can be opened in any modern web browser
- Includes all current filter and layout settings
- Perfect for sharing analysis with others
- Works offline once saved

### **Best Practices for Export**
1. Set up your ideal view (layout, filters, highlights)
2. Click "Export HTML" 
3. Choose a descriptive filename
4. The file will automatically open in your browser
5. Share the HTML file with colleagues or documentation

## 🎯 Use Cases

### **Network Planning**
- Use hierarchical layout to understand network topology
- Identify areas needing more routers (isolated client clusters)
- Find optimal router placement using critical path analysis

### **Troubleshooting**
- Filter to show only problematic nodes (isolated, vulnerable)
- Check connection quality through detailed tooltips
- Use critical path highlighting to prioritize fixes

### **Network Optimization**
- Identify redundant routers (low optimization scores)
- Find clients that should be promoted to routers
- Analyze connection quality and plan improvements

### **Documentation**
- Export HTML visualizations for network documentation
- Use different layouts for different audiences
- Include in reports and presentations

---

## Quick Start Checklist

1. ✅ **Load your network data** in the main application
2. ✅ **Try different layout algorithms** to find the clearest view
3. ✅ **Enable critical path highlighting** to see important connections
4. ✅ **Use filters** to focus on specific node types or problems
5. ✅ **Click nodes** to see detailed information
6. ✅ **Adjust node separation and label size** for optimal readability
7. ✅ **Export to HTML** when you have a good view to share

The enhanced visualization transforms your "rats nest" into a clear, interactive tool for understanding and optimizing your mesh network!

### Enhanced Click Interactivity

The network visualization now supports advanced click functionality that provides detailed insights into node connections:

#### Node Click Behavior
- **Single Click**: Click on any node to highlight it and all its connections
  - The clicked node turns **orange** to indicate selection
  - Connected nodes remain in their original colors and show labels
  - All other nodes are **dimmed** (gray, semi-transparent) to focus attention
  - Edges connected to the selected node are **highlighted in orange** with increased thickness
  - Other edges are dimmed to reduce visual clutter

#### Visual Feedback
- **Selected Node**: Highlighted in bright orange (#FF6F00)
- **Connected Nodes**: Retain original colors (blue for routers, green for clients, etc.)
- **Connected Node Labels**: Always visible when a node is selected, showing node IDs or names
- **Highlighted Edges**: Orange color with 3x thickness for edges connected to the selected node
- **Dimmed Elements**: Unconnected nodes and edges fade to gray with reduced opacity

#### Clearing Selection
- **Click Same Node**: Click the already-selected node to clear the selection
- **Double-Click Anywhere**: Double-click on the visualization to clear all highlighting
- **Clear Highlights Button**: Use the "Clear Highlights" button in the control panel

#### Benefits
- **Topology Understanding**: Quickly see which nodes are directly connected
- **Network Analysis**: Identify hub nodes and understand local connectivity patterns
- **Troubleshooting**: Focus on specific node neighborhoods for debugging
- **Performance Assessment**: Visualize the impact of individual nodes on network connectivity

## Layout Algorithms 