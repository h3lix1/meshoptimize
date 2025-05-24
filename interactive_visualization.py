"""
Interactive Visualization Module for Large Networks
Uses Plotly with WebGL for better performance with many nodes/edges
"""

import plotly.graph_objects as go
import plotly.offline as pyo
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSignal, QUrl
import tempfile
import os
import colorsys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LayoutOptions:
    """Options for network layout algorithms."""
    algorithm: str = "fruchterman_reingold"
    k: float = None  # Optimal distance between nodes
    iterations: int = 50
    seed: int = 42
    dimension: int = 2
    
    
def sanitize_label(label: str) -> str:
    """Remove problematic Unicode characters from labels."""
    if not label:
        return ""
    # Keep only ASCII and basic alphanumeric
    return ''.join(char for char in label if ord(char) < 127 or char.isalnum()).strip()


class InteractiveNetworkVisualizer(QWidget):
    """Interactive network visualization using Plotly with WebGL."""
    
    node_clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = {}
        self.edges = []
        self.graph = None
        self.pos = None
        self.current_html = None
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Control buttons
        controls = QHBoxLayout()
        
        self.reset_layout_btn = QPushButton("Reset Layout")
        self.reset_layout_btn.clicked.connect(self._reset_layout)
        controls.addWidget(self.reset_layout_btn)
        
        self.force_layout_btn = QPushButton("Force Layout")
        self.force_layout_btn.clicked.connect(lambda: self._recalculate_layout("spring"))
        controls.addWidget(self.force_layout_btn)
        
        self.circular_layout_btn = QPushButton("Circular Layout")
        self.circular_layout_btn.clicked.connect(lambda: self._recalculate_layout("circular"))
        controls.addWidget(self.circular_layout_btn)
        
        self.hierarchical_layout_btn = QPushButton("Hierarchical Layout")
        self.hierarchical_layout_btn.clicked.connect(lambda: self._recalculate_layout("hierarchical"))
        controls.addWidget(self.hierarchical_layout_btn)
        
        controls.addWidget(QWidget())  # Spacer
        
        self.export_btn = QPushButton("Export HTML")
        self.export_btn.clicked.connect(self._export_html)
        controls.addWidget(self.export_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Web view for Plotly
        self.web_view = QWebEngineView()
        # Enable JavaScript and local content access
        settings = self.web_view.settings()
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        layout.addWidget(self.web_view)
        
    def update_network(self, nodes: Dict, edges: List[Dict], 
                      highlight_nodes: Optional[Set[str]] = None,
                      node_scores: Optional[Dict] = None):
        """Update the network visualization."""
        self.nodes = nodes
        self.edges = edges
        self.highlight_nodes = highlight_nodes or set()
        self.node_scores = node_scores or {}
        
        # Build NetworkX graph
        self.graph = nx.Graph()
        for node_id, node_data in nodes.items():
            self.graph.add_node(node_id, **node_data)
        for edge in edges:
            if edge['from'] in nodes and edge['to'] in nodes:
                self.graph.add_edge(edge['from'], edge['to'], **edge)
                
        # Calculate layout if needed
        if self.pos is None or set(self.graph.nodes()) != set(self.pos.keys()):
            self._calculate_layout()
            
        self._create_plotly_figure()
        
    def _calculate_layout(self, algorithm="spring"):
        """Calculate node positions using various layout algorithms."""
        if len(self.graph.nodes()) == 0:
            self.pos = {}
            return
            
        if algorithm == "spring":
            # Use Fruchterman-Reingold algorithm for better large graph layout
            self.pos = nx.spring_layout(
                self.graph,
                k=3/np.sqrt(len(self.graph.nodes())),
                iterations=50,
                seed=42,
                weight=None  # Ignore weights for layout
            )
        elif algorithm == "circular":
            # Group by connectivity
            self.pos = nx.circular_layout(self.graph)
        elif algorithm == "hierarchical":
            # Try to create hierarchical layout based on node degree
            try:
                # Create a directed graph for hierarchy
                DG = nx.DiGraph()
                for edge in self.edges:
                    DG.add_edge(edge['from'], edge['to'])
                # Use graphviz layout if available
                self.pos = nx.nx_agraph.graphviz_layout(DG, prog='dot')
            except:
                # Fallback to spring layout
                self.pos = nx.spring_layout(self.graph, k=3/np.sqrt(len(self.graph.nodes())))
        else:
            self.pos = nx.spring_layout(self.graph)
            
    def _recalculate_layout(self, algorithm):
        """Recalculate layout with specified algorithm."""
        self._calculate_layout(algorithm)
        self._create_plotly_figure()
        
    def _reset_layout(self):
        """Reset to default layout."""
        self.pos = None
        self._calculate_layout()
        self._create_plotly_figure()
        
    def _create_plotly_figure(self):
        """Create an optimized Plotly figure for large networks."""
        if not self.graph or not self.pos:
            return
            
        # Check if we have nodes to display
        if len(self.nodes) == 0:
            logger.warning("No nodes to display")
            return
            
        # Prepare edge traces using scattergl for better performance
        edge_x = []
        edge_y = []
        
        for edge in self.edges:
            if edge['from'] in self.pos and edge['to'] in self.pos:
                x0, y0 = self.pos[edge['from']]
                x1, y1 = self.pos[edge['to']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
        edge_trace = go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            name='Edges'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_customdata = []
        node_color = []
        node_size = []
        
        for node_id, node_data in self.nodes.items():
            if node_id in self.pos:
                x, y = self.pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                # Prepare hover text
                label = sanitize_label(node_data.get('label', node_id))
                text = f"<b>{label}</b><br>"
                text += f"<b>ID:</b> {node_id}<br>"
                text += f"<b>Role:</b> {node_data.get('role', 'Unknown')}<br>"
                
                # Add channel information
                channel = node_data.get('channel', '')
                if channel:
                    text += f"<b>Channel:</b> {channel}<br>"
                
                # Add hardware info
                hardware = node_data.get('hardware', '')
                if hardware:
                    text += f"<b>Hardware:</b> {hardware}<br>"
                
                # Add name info
                short_name = node_data.get('short_name', '')
                long_name = node_data.get('long_name', '')
                if short_name:
                    text += f"<b>Short Name:</b> {short_name}<br>"
                if long_name and long_name != label:
                    text += f"<b>Long Name:</b> {sanitize_label(long_name)}<br>"
                
                # Connection info
                text += f"<b>Connections:</b> {len(list(self.graph.neighbors(node_id)))}<br>"
                
                # Add neighbor information
                neighbors = list(self.graph.neighbors(node_id))
                if neighbors:
                    neighbor_names = []
                    for n_id in neighbors[:5]:  # Show first 5 neighbors
                        n_data = self.nodes.get(n_id, {})
                        n_label = sanitize_label(n_data.get('label', n_id))
                        neighbor_names.append(n_label)
                    text += f"<b>Connected to:</b> {', '.join(neighbor_names)}"
                    if len(neighbors) > 5:
                        text += f" and {len(neighbors) - 5} more"
                    text += "<br>"
                
                if self.node_scores and node_id in self.node_scores:
                    score = self.node_scores[node_id]
                    text += f"<br><b>Optimization Scores:</b><br>"
                    text += f"Total: {score.total_score:.3f}<br>"
                    text += f"Coverage: {score.coverage_score:.3f}<br>"
                    text += f"Centrality: {score.centrality_score:.3f}<br>"
                    text += f"Redundancy: {score.redundancy_score:.3f}<br>"
                    
                node_text.append(text)
                node_customdata.append(node_id)
                
                # Determine color
                if node_id in self.highlight_nodes:
                    if node_data.get('is_router'):
                        node_color.append('#8E24AA')  # Purple for demotion
                    else:
                        node_color.append('#FF6F00')  # Orange for promotion
                elif len(list(self.graph.neighbors(node_id))) == 0:
                    node_color.append('#E53935')  # Red for isolated
                elif len(list(self.graph.neighbors(node_id))) == 1:
                    node_color.append('#FDD835')  # Yellow for vulnerable
                elif node_data.get('is_router'):
                    node_color.append('#1E88E5')  # Blue for router
                else:
                    node_color.append('#43A047')  # Green for client
                    
                # Size based on degree with better scaling
                degree = len(list(self.graph.neighbors(node_id)))
                # Minimum size of 8 for better visibility, max of 30
                node_size.append(8 + min(degree * 3, 22))
                
        node_trace = go.Scattergl(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text',
            customdata=node_customdata,
            name='Nodes',
            showlegend=False
        )
        
        # Create figure with optimizations
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Add legend traces for node types
        legend_traces = [
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='#1E88E5'),
                      name='Router', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers', 
                      marker=dict(size=10, color='#43A047'),
                      name='Client', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='#E53935'),
                      name='Isolated', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='#FDD835'),
                      name='Vulnerable (1 connection)', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='#8E24AA'),
                      name='Proposed Demotion', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='#FF6F00'),
                      name='Proposed Promotion', showlegend=True),
        ]
        
        for trace in legend_traces:
            fig.add_trace(trace)
        
        # Update layout for better performance and interactivity
        fig.update_layout(
            title=dict(
                text=f'Meshtastic Network ({len(self.nodes)} nodes, {len(self.edges)} edges)',
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="Black",
                borderwidth=1
            ),
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_x) - 0.1, max(node_x) + 0.1] if node_x else [-1, 1]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_y) - 0.1, max(node_y) + 0.1] if node_y else [-1, 1]
            ),
            plot_bgcolor='white',
            dragmode='pan',  # Default to pan mode
            clickmode='event+select',
            uirevision='constant'  # Preserve zoom/pan state
        )
        
        # Add custom buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"dragmode": "pan"}],
                            label="Pan",
                            method="relayout"
                        ),
                        dict(
                            args=[{"dragmode": "zoom"}],
                            label="Zoom",
                            method="relayout"
                        ),
                        dict(
                            args=[{"dragmode": "select"}],
                            label="Select",
                            method="relayout"
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            # Add custom JavaScript for node clicks
            # Use 'inline' to embed the entire Plotly library in the HTML
            html_str = fig.to_html(
                include_plotlyjs='inline',
                config={'displayModeBar': True, 'displaylogo': False}
            )
            
            # Inject custom JavaScript for handling clicks
            custom_js = """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                var plot = document.getElementsByClassName('plotly-graph-div')[0];
                plot.on('plotly_click', function(data) {
                    if (data.points.length > 0) {
                        var point = data.points[0];
                        if (point.customdata) {
                            // Send node ID back to Python
                            window.pyqtBridge = window.pyqtBridge || {};
                            window.pyqtBridge.nodeClicked = point.customdata;
                        }
                    }
                });
            });
            </script>
            """
            
            html_str = html_str.replace('</body>', custom_js + '</body>')
            f.write(html_str)
            self.current_html = f.name
            
        # Debug: verify file was created and has content
        import os
        if os.path.exists(self.current_html):
            file_size = os.path.getsize(self.current_html)
            logger.info(f"HTML file created: {self.current_html} ({file_size} bytes)")
            
        # Load in web view
        self.web_view.setUrl(QUrl.fromLocalFile(self.current_html))
        
        # Debug: print the file path
        logger.info(f"Loading HTML from: {self.current_html}")
        
        # Connect to loading finished signal for debugging
        self.web_view.loadFinished.connect(self._on_load_finished)
        
    def _on_load_finished(self, success):
        """Handle when the page finishes loading."""
        if success:
            logger.info("Interactive plot loaded successfully")
        else:
            logger.error("Failed to load interactive plot")
            
    def _export_html(self):
        """Export the visualization to an HTML file and open in browser."""
        if not self.current_html or not os.path.exists(self.current_html):
            logger.warning("No visualization to export")
            return
            
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", "network_visualization.html", "HTML Files (*.html)"
        )
        
        if filename:
            import shutil
            shutil.copy(self.current_html, filename)
            logger.info(f"Exported visualization to {filename}")
            
            # Try to open in browser
            import webbrowser
            webbrowser.open(f"file://{filename}")
            
    def clear_highlights(self):
        """Clear all highlights and redraw."""
        self.highlight_nodes = set()
        self._create_plotly_figure()
        
    def export_to_file(self, filename: str):
        """Export the current visualization."""
        if self.current_html:
            import shutil
            shutil.copy(self.current_html, filename)
            
    def __del__(self):
        """Clean up temporary files."""
        if self.current_html and os.path.exists(self.current_html):
            try:
                os.unlink(self.current_html)
            except:
                pass 