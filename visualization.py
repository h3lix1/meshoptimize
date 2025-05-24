"""
Visualization Module for Meshtastic Network Optimizer
Handles network graph visualization with interactive features.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.offline as pyo
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
import colorsys
import re

logger = logging.getLogger(__name__)


def sanitize_label(label: str) -> str:
    """Remove or replace problematic Unicode characters from labels."""
    if not label:
        return ""
    
    # Remove emoji and other problematic Unicode characters
    # This regex matches most emoji and special Unicode symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000027BF"  # Miscellaneous symbols
        "]+", 
        flags=re.UNICODE
    )
    
    # Replace emojis with nothing or a simple text representation
    label = emoji_pattern.sub('', label)
    
    # Also ensure we only have printable ASCII and common Unicode
    # This keeps letters, numbers, spaces, and common punctuation
    label = ''.join(char for char in label if ord(char) < 127 or char.isalnum())
    
    return label.strip()


class NetworkColors:
    """Color scheme for network visualization."""
    ROUTER = '#1E88E5'  # Blue
    CLIENT = '#43A047'  # Green
    VULNERABLE = '#FDD835'  # Yellow
    ISOLATED = '#E53935'  # Red
    PROMOTION_CANDIDATE = '#FF6F00'  # Orange
    DEMOTION_CANDIDATE = '#8E24AA'  # Purple
    EDGE_NORMAL = '#CCCCCC'  # Light gray
    EDGE_WEAK = '#FFCCCC'  # Light red
    EDGE_STRONG = '#666666'  # Dark gray
    BACKGROUND = '#FFFFFF'  # White
    

class NetworkVisualizer(QWidget):
    """Qt widget for network visualization using matplotlib."""
    
    node_clicked = pyqtSignal(str)  # Signal emitted when a node is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = {}
        self.edges = []
        self.graph = None
        self.pos = None
        self.selected_node = None
        self.highlight_nodes = set()
        self.node_colors = {}
        self.edge_colors = {}
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        
    def update_network(self, nodes: Dict, edges: List[Dict], 
                      highlight_nodes: Optional[Set[str]] = None,
                      node_colors: Optional[Dict[str, str]] = None,
                      edge_colors: Optional[Dict[Tuple[str, str], str]] = None):
        """Update the network visualization."""
        self.nodes = nodes
        self.edges = edges
        self.highlight_nodes = highlight_nodes or set()
        self.node_colors = node_colors or {}
        self.edge_colors = edge_colors or {}
        
        # Build NetworkX graph
        self.graph = nx.Graph()
        for node_id, node_data in nodes.items():
            self.graph.add_node(node_id, **node_data)
        for edge in edges:
            self.graph.add_edge(edge['from'], edge['to'], **edge)
            
        # Calculate layout if not already done
        if self.pos is None or set(self.graph.nodes()) != set(self.pos.keys()):
            self._calculate_layout()
            
        self._draw_network()
        
    def _calculate_layout(self):
        """Calculate node positions using spring layout."""
        # Use spring layout with custom parameters for better visualization
        if len(self.graph.nodes()) > 0:
            self.pos = nx.spring_layout(
                self.graph,
                k=2/np.sqrt(max(1, len(self.graph.nodes()))),
                iterations=50,
                seed=42  # For consistent layout
            )
        else:
            self.pos = {}
        
    def _draw_network(self):
        """Draw the network graph."""
        self.ax.clear()
        
        if not self.graph:
            return
            
        # Draw edges first (so they appear behind nodes)
        edge_collection = []
        edge_colors_list = []
        edge_widths = []
        
        for edge in self.edges:
            from_node = edge['from']
            to_node = edge['to']
            
            if from_node in self.pos and to_node in self.pos:
                x1, y1 = self.pos[from_node]
                x2, y2 = self.pos[to_node]
                edge_collection.append([(x1, y1), (x2, y2)])
                
                # Determine edge color
                edge_key = (from_node, to_node)
                edge_key_rev = (to_node, from_node)
                
                if edge_key in self.edge_colors:
                    color = self.edge_colors[edge_key]
                elif edge_key_rev in self.edge_colors:
                    color = self.edge_colors[edge_key_rev]
                else:
                    # Default color based on edge weight/SNR
                    snr = edge.get('snr')
                    if snr is None:
                        color = NetworkColors.EDGE_NORMAL
                    elif snr < -10:
                        color = NetworkColors.EDGE_WEAK
                    elif snr > 0:
                        color = NetworkColors.EDGE_STRONG
                    else:
                        color = NetworkColors.EDGE_NORMAL
                        
                edge_colors_list.append(color)
                
                # Edge width based on weight
                weight = edge.get('weight', 1.0)
                edge_widths.append(1 + weight * 2)
                
        # Draw edges
        if edge_collection:
            from matplotlib.collections import LineCollection
            lc = LineCollection(edge_collection, colors=edge_colors_list, 
                              linewidths=edge_widths, alpha=0.6)
            self.ax.add_collection(lc)
            
        # Draw nodes
        for node_id, node_data in self.nodes.items():
            if node_id not in self.pos:
                continue
                
            x, y = self.pos[node_id]
            
            # Determine node color
            if node_id in self.node_colors:
                color = self.node_colors[node_id]
            elif node_id in self.highlight_nodes:
                if node_data['is_router']:
                    color = NetworkColors.DEMOTION_CANDIDATE
                else:
                    color = NetworkColors.PROMOTION_CANDIDATE
            elif len(list(self.graph.neighbors(node_id))) == 0:
                color = NetworkColors.ISOLATED
            elif len(list(self.graph.neighbors(node_id))) == 1:
                color = NetworkColors.VULNERABLE
            elif node_data['is_router']:
                color = NetworkColors.ROUTER
            else:
                color = NetworkColors.CLIENT
                
            # Node size based on importance (number of connections)
            node_size = 300 + len(list(self.graph.neighbors(node_id))) * 100
            
            # Draw node
            circle = plt.Circle((x, y), 0.03, color=color, 
                              ec='black' if node_id == self.selected_node else 'white',
                              linewidth=2 if node_id == self.selected_node else 1,
                              zorder=10)
            self.ax.add_patch(circle)
            
            # Draw label
            label = node_data.get('label', node_id)
            # Sanitize label to remove emojis and problematic characters
            label = sanitize_label(label)
            if len(label) > 10:
                label = label[:10] + '...'
            self.ax.text(x, y, label, ha='center', va='center', 
                        fontsize=8, zorder=11)
                        
        # Set axis properties
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=NetworkColors.ROUTER, label='Router'),
            mpatches.Patch(color=NetworkColors.CLIENT, label='Client'),
            mpatches.Patch(color=NetworkColors.VULNERABLE, label='Vulnerable (1 connection)'),
            mpatches.Patch(color=NetworkColors.ISOLATED, label='Isolated'),
            mpatches.Patch(color=NetworkColors.PROMOTION_CANDIDATE, label='Promotion Candidate'),
            mpatches.Patch(color=NetworkColors.DEMOTION_CANDIDATE, label='Demotion Candidate')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add title
        self.ax.set_title('Meshtastic Network Topology', fontsize=16, fontweight='bold')
        
        # Refresh canvas
        self.canvas.draw()
        
    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
            
        # Find closest node to click
        min_dist = float('inf')
        clicked_node = None
        
        for node_id in self.pos:
            x, y = self.pos[node_id]
            dist = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            if dist < min_dist and dist < 0.05:  # 0.05 is the click threshold
                min_dist = dist
                clicked_node = node_id
                
        if clicked_node:
            self.selected_node = clicked_node
            self.node_clicked.emit(clicked_node)
            self._draw_network()
            
    def _on_hover(self, event):
        """Handle mouse hover events."""
        # Could implement tooltips here
        pass
        
    def highlight_path(self, path: List[str]):
        """Highlight a path through the network."""
        # Color edges in the path
        self.edge_colors.clear()
        for i in range(len(path) - 1):
            self.edge_colors[(path[i], path[i+1])] = '#FF0000'  # Red for path
            
        self._draw_network()
        
    def clear_highlights(self):
        """Clear all highlights."""
        self.highlight_nodes.clear()
        self.node_colors.clear()
        self.edge_colors.clear()
        self.selected_node = None
        self._draw_network()
        
    def export_to_file(self, filename: str):
        """Export the current visualization to a file."""
        self.figure.savefig(filename, dpi=300, bbox_inches='tight')
        

class PlotlyNetworkVisualizer:
    """Create interactive Plotly visualization of the network."""
    
    @staticmethod
    def create_interactive_plot(nodes: Dict, edges: List[Dict], 
                               node_scores: Optional[Dict] = None,
                               highlight_nodes: Optional[Set[str]] = None) -> go.Figure:
        """Create an interactive Plotly figure of the network."""
        # Build NetworkX graph for layout
        G = nx.Graph()
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)
        for edge in edges:
            G.add_edge(edge['from'], edge['to'], **edge)
            
        # Calculate layout
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2/np.sqrt(max(1, len(G.nodes()))), iterations=50, seed=42)
        else:
            pos = {}
        
        # Create edge traces
        edge_trace = []
        for edge in edges:
            if edge['from'] in pos and edge['to'] in pos:
                x0, y0 = pos[edge['from']]
                x1, y1 = pos[edge['to']]
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                ))
                
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node_id, node_data in nodes.items():
            if node_id in pos:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                # Create hover text
                sanitized_label = sanitize_label(node_data.get('label', node_id))
                text = f"<b>{sanitized_label}</b><br>"
                text += f"Role: {node_data.get('role', 'Unknown')}<br>"
                text += f"Connections: {len(list(G.neighbors(node_id)))}<br>"
                
                if node_scores and node_id in node_scores:
                    score = node_scores[node_id]
                    text += f"Total Score: {score.total_score:.3f}<br>"
                    text += f"Coverage: {score.coverage_score:.3f}<br>"
                    text += f"Centrality: {score.centrality_score:.3f}<br>"
                    text += f"Redundancy: {score.redundancy_score:.3f}<br>"
                    
                node_text.append(text)
                
                # Determine color
                if highlight_nodes and node_id in highlight_nodes:
                    if node_data['is_router']:
                        node_color.append('purple')
                    else:
                        node_color.append('orange')
                elif len(list(G.neighbors(node_id))) == 0:
                    node_color.append('red')
                elif len(list(G.neighbors(node_id))) == 1:
                    node_color.append('yellow')
                elif node_data['is_router']:
                    node_color.append('blue')
                else:
                    node_color.append('green')
                    
                # Size based on connections
                node_size.append(10 + len(list(G.neighbors(node_id))) * 3)
                
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[sanitize_label(n.get('label', n.get('id', '')))[:10] for n in nodes.values() if n.get('id') in pos],
            textposition='top center',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title='Interactive Meshtastic Network Map',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
        
    @staticmethod
    def save_interactive_plot(fig: go.Figure, filename: str = 'network_visualization.html'):
        """Save the interactive plot to an HTML file."""
        pyo.plot(fig, filename=filename, auto_open=False) 