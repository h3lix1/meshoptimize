"""
Enhanced Interactive Visualization Module for Large Networks
Features: Advanced interactivity, critical path highlighting, filtering, and zoom-aware labeling
PERFORMANCE OPTIMIZED with caching, threading, and JIT compilation
"""

import plotly.graph_objects as go
import plotly.offline as pyo
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QHBoxLayout, 
                           QCheckBox, QSlider, QLabel, QComboBox, QSpinBox,
                           QGroupBox, QGridLayout, QTextEdit, QSplitter)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSignal, QUrl, Qt, QThread, pyqtSlot
import tempfile
import os
import colorsys
import json
from dataclasses import dataclass
from collections import defaultdict
import concurrent.futures
import threading
import time

# JIT compilation for performance
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

logger = logging.getLogger(__name__)


@dataclass
class LayoutOptions:
    """Enhanced options for network layout algorithms."""
    algorithm: str = "spring_enhanced"
    k: float = None  # Optimal distance between nodes
    iterations: int = 100
    seed: int = 42
    dimension: int = 2
    edge_bundling: bool = False
    hierarchical_direction: str = "TB"  # Top-Bottom, Left-Right
    cluster_separation: float = 1.5
    

@dataclass
class FilterOptions:
    """Options for filtering network display."""
    min_degree: int = 0
    max_degree: int = 100
    show_routers: bool = True
    show_clients: bool = True
    show_isolated: bool = True
    show_vulnerable: bool = True
    highlight_critical_paths: bool = False
    edge_weight_threshold: float = 0.0
    show_node_labels: bool = True
    node_label_size: int = 12
    
    
def sanitize_label(label: str) -> str:
    """Remove problematic Unicode characters from labels."""
    if not label:
        return ""
    # Keep only ASCII and basic alphanumeric
    return ''.join(char for char in label if ord(char) < 127 or char.isalnum()).strip()


class EnhancedNetworkVisualizer(QWidget):
    """Enhanced interactive network visualization with advanced features and performance optimizations."""
    
    node_clicked = pyqtSignal(str)
    node_selected = pyqtSignal(list)  # For multiple node selection
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = {}
        self.edges = []
        self.graph = None
        self.pos = None
        self.current_html = None
        self.critical_paths = []
        self.selected_nodes = set()
        self.highlight_nodes = set()  # Track currently highlighted nodes
        
        # Performance optimizations
        self.performance_cache = PerformanceCache()
        self.layout_thread = None
        self._last_update_hash = None
        self._cached_node_traces = None
        self._cached_edge_traces = None
        
        # Filter and layout options
        self.filter_options = FilterOptions()
        self.layout_options = LayoutOptions()
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the enhanced UI components."""
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for controls
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        # Right panel for visualization
        viz_panel = self._create_visualization_panel()
        splitter.addWidget(viz_panel)
        
        # Set initial sizes (control panel smaller)
        splitter.setSizes([300, 1000])
        
        main_layout.addWidget(splitter)
        
    def _create_control_panel(self) -> QWidget:
        """Create the control panel with various options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Layout controls
        layout_group = QGroupBox("Layout Options")
        layout_grid = QGridLayout(layout_group)
        
        # Layout algorithm selector
        layout_grid.addWidget(QLabel("Algorithm:"), 0, 0)
        self.layout_combo = QComboBox()
        self.layout_combo.addItems([
            "spring_enhanced", "spring_enhanced_fast", "force_atlas", "circular_clustered", 
            "hierarchical", "edge_bundled", "community_based"
        ])
        self.layout_combo.currentTextChanged.connect(self._on_layout_changed)
        layout_grid.addWidget(self.layout_combo, 0, 1)
        
        # Iterations slider
        layout_grid.addWidget(QLabel("Iterations:"), 1, 0)
        self.iterations_slider = QSlider(Qt.Orientation.Horizontal)
        self.iterations_slider.setRange(10, 1000)
        self.iterations_slider.setValue(500)
        self.iterations_slider.valueChanged.connect(self._on_iterations_changed)
        layout_grid.addWidget(self.iterations_slider, 1, 1)
        self.iterations_label = QLabel("100")
        layout_grid.addWidget(self.iterations_label, 1, 2)
        
        # Node separation
        layout_grid.addWidget(QLabel("Node Separation:"), 2, 0)
        self.separation_slider = QSlider(Qt.Orientation.Horizontal)
        self.separation_slider.setRange(50, 3000)
        self.separation_slider.setValue(2500)
        self.separation_slider.valueChanged.connect(self._on_separation_changed)
        layout_grid.addWidget(self.separation_slider, 2, 1)
        self.separation_label = QLabel("1.5")
        layout_grid.addWidget(self.separation_label, 2, 2)
        
        layout.addWidget(layout_group)
        
        # Filter controls
        filter_group = QGroupBox("Display Filters")
        filter_grid = QGridLayout(filter_group)
        
        # Node type filters
        self.show_routers_cb = QCheckBox("Show Routers")
        self.show_routers_cb.setChecked(True)
        self.show_routers_cb.toggled.connect(self._on_filter_changed)
        filter_grid.addWidget(self.show_routers_cb, 0, 0)
        
        self.show_clients_cb = QCheckBox("Show Clients")
        self.show_clients_cb.setChecked(True)
        self.show_clients_cb.toggled.connect(self._on_filter_changed)
        filter_grid.addWidget(self.show_clients_cb, 0, 1)
        
        self.show_isolated_cb = QCheckBox("Show Isolated")
        self.show_isolated_cb.setChecked(True)
        self.show_isolated_cb.toggled.connect(self._on_filter_changed)
        filter_grid.addWidget(self.show_isolated_cb, 1, 0)
        
        self.show_vulnerable_cb = QCheckBox("Show Vulnerable")
        self.show_vulnerable_cb.setChecked(True)
        self.show_vulnerable_cb.toggled.connect(self._on_filter_changed)
        filter_grid.addWidget(self.show_vulnerable_cb, 1, 1)
        
        # Degree range filter
        filter_grid.addWidget(QLabel("Min Connections:"), 2, 0)
        self.min_degree_spin = QSpinBox()
        self.min_degree_spin.setRange(0, 100)
        self.min_degree_spin.valueChanged.connect(self._on_filter_changed)
        filter_grid.addWidget(self.min_degree_spin, 2, 1)
        
        filter_grid.addWidget(QLabel("Max Connections:"), 3, 0)
        self.max_degree_spin = QSpinBox()
        self.max_degree_spin.setRange(1, 100)
        self.max_degree_spin.setValue(100)
        self.max_degree_spin.valueChanged.connect(self._on_filter_changed)
        filter_grid.addWidget(self.max_degree_spin, 3, 1)
        
        # Critical path highlighting
        self.highlight_critical_cb = QCheckBox("Highlight Critical Paths")
        self.highlight_critical_cb.toggled.connect(self._on_filter_changed)
        filter_grid.addWidget(self.highlight_critical_cb, 4, 0, 1, 2)
        
        layout.addWidget(filter_group)
        
        # Node label controls
        label_group = QGroupBox("Node Labels")
        label_grid = QGridLayout(label_group)
        
        self.show_labels_cb = QCheckBox("Show Node Labels")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.toggled.connect(self._on_filter_changed)
        label_grid.addWidget(self.show_labels_cb, 0, 0, 1, 2)
        
        label_grid.addWidget(QLabel("Label Size:"), 1, 0)
        self.label_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.label_size_slider.setRange(8, 24)
        self.label_size_slider.setValue(12)
        self.label_size_slider.valueChanged.connect(self._on_label_size_changed)
        label_grid.addWidget(self.label_size_slider, 1, 1)
        self.label_size_label = QLabel("12")
        label_grid.addWidget(self.label_size_label, 1, 2)
        
        layout.addWidget(label_group)
        
        # Action buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_group)
        
        self.recalculate_btn = QPushButton("Recalculate Layout")
        self.recalculate_btn.clicked.connect(self._recalculate_layout)
        button_layout.addWidget(self.recalculate_btn)
        
        self.find_critical_btn = QPushButton("Find Critical Paths")
        self.find_critical_btn.clicked.connect(self._find_critical_paths)
        button_layout.addWidget(self.find_critical_btn)
        
        self.center_selected_btn = QPushButton("Center on Selected")
        self.center_selected_btn.clicked.connect(self._center_on_selected)
        button_layout.addWidget(self.center_selected_btn)
        
        self.export_btn = QPushButton("Export HTML")
        self.export_btn.clicked.connect(self._export_html)
        button_layout.addWidget(self.export_btn)
        
        layout.addWidget(button_group)
        
        # Node details area
        details_group = QGroupBox("Node Details")
        details_layout = QVBoxLayout(details_group)
        
        self.node_details = QTextEdit()
        self.node_details.setMaximumHeight(200)
        self.node_details.setReadOnly(True)
        # Fix styling to ensure text is visible
        self.node_details.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.node_details.setPlainText("Click a node to view its details...")
        details_layout.addWidget(self.node_details)
        
        layout.addWidget(details_group)
        
        layout.addStretch()
        return panel
        
    def _create_visualization_panel(self) -> QWidget:
        """Create the visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Web view for Plotly
        self.web_view = QWebEngineView()
        # Enable JavaScript and local content access
        settings = self.web_view.settings()
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        
        # Connect title change signal to handle node details updates
        self.web_view.titleChanged.connect(self._on_title_changed)
        
        layout.addWidget(self.web_view)
        
        return panel
        
    def update_network(self, nodes: Dict, edges: List[Dict], 
                      highlight_nodes: Optional[Set[str]] = None,
                      node_scores: Optional[Dict] = None):
        """Update the network visualization with enhanced features and proper synchronization."""
        logger.info(f"Updating network with {len(nodes)} nodes and {len(edges)} edges")
        
        self.nodes = nodes
        self.edges = edges
        self.highlight_nodes = highlight_nodes or set()
        self.node_scores = node_scores or {}
        
        # Clear caches when data changes
        self.performance_cache.clear()
        
        # Build NetworkX graph
        self.graph = nx.Graph()
        for node_id, node_data in nodes.items():
            self.graph.add_node(node_id, **node_data)
            
        for edge in edges:
            if edge['from'] in nodes and edge['to'] in nodes:
                # Create a copy of edge data to avoid modifying original
                edge_attrs = edge.copy()
                
                # Calculate weight based on SNR if available
                weight = edge.get('weight', 1.0)
                if 'snr' in edge and edge['snr'] is not None:
                    # Higher SNR = lower weight (better connection)
                    weight = max(0.1, 1.0 - (edge['snr'] + 10) / 30)
                
                # Remove weight from attributes to avoid conflict
                if 'weight' in edge_attrs:
                    del edge_attrs['weight']
                
                # Add edge with weight as explicit parameter
                self.graph.add_edge(edge['from'], edge['to'], weight=weight, **edge_attrs)
        
        logger.info(f"Built graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        
        # Calculate layout - this now includes proper validation and fallbacks
        self._calculate_enhanced_layout()
        
        # Only proceed with visualization if we have valid positions
        if not self.pos or len(self.pos) != len(self.graph.nodes()):
            logger.error("Failed to generate valid layout positions, cannot update visualization")
            return
            
        # Create the visualization
        self._create_enhanced_plotly_figure()
        
    def _calculate_enhanced_layout(self):
        """Calculate node positions using enhanced layout algorithms with performance optimizations."""
        if len(self.graph.nodes()) == 0:
            self.pos = {}
            logger.warning("No nodes in graph for layout calculation")
            return
            
        algorithm = self.layout_options.algorithm
        logger.info(f"Calculating layout using {algorithm} for {len(self.graph.nodes())} nodes")
        
        # Check cache first - but be more selective about what we cache
        graph_hash = f"{len(self.graph.nodes())}_{len(self.graph.edges())}_{algorithm}_{self.layout_options.iterations}"
        cached_pos = self.performance_cache.get(f"layout_{graph_hash}")
        
        if cached_pos is not None and len(cached_pos) == len(self.graph.nodes()):
            logger.info(f"Using cached layout for {len(self.graph.nodes())} nodes")
            self.pos = cached_pos
            return
            
        # For better responsiveness, use synchronous calculation for smaller graphs
        # or when we need immediate results
        if len(self.graph.nodes()) <= 200 or algorithm not in ["spring_enhanced", "spring_enhanced_fast"]:
            self._calculate_layout_sync()
        else:
            # Use background thread only for very large graphs with expensive algorithms
            self._calculate_layout_threaded()
    
    def _calculate_layout_threaded(self):
        """Calculate layout in background thread - with proper synchronization."""
        if self.layout_thread and self.layout_thread.isRunning():
            self.layout_thread.terminate()
            self.layout_thread.wait()
        
        # Start with a temporary layout so nodes aren't scattered
        logger.info("Generating temporary layout while calculating optimal positions...")
        self.pos = nx.spring_layout(self.graph, seed=42, iterations=10)  # Quick temporary layout
        
        self.layout_thread = LayoutCalculationThread(
            self.graph, 
            self.layout_options.algorithm, 
            self.layout_options
        )
        self.layout_thread.layout_finished.connect(self._on_layout_finished)
        self.layout_thread.start()
        
        logger.info("Background layout calculation started...")
    
    @pyqtSlot(dict)
    def _on_layout_finished(self, pos):
        """Handle completion of background layout calculation."""
        if pos and len(pos) == len(self.graph.nodes()):
            logger.info(f"Background layout completed with {len(pos)} node positions")
            self.pos = pos
            
            # Cache the result with validation
            graph_hash = f"{len(self.graph.nodes())}_{len(self.graph.edges())}_{self.layout_options.algorithm}_{self.layout_options.iterations}"
            self.performance_cache.set(f"layout_{graph_hash}", pos)
            
            # Clear node trace cache to force regeneration with new positions
            self.performance_cache.clear()
            
            # Update visualization with new positions
            self._create_enhanced_plotly_figure()
        else:
            logger.error(f"Background layout calculation failed or returned invalid positions: {len(pos) if pos else 0} positions for {len(self.graph.nodes())} nodes")
    
    def _calculate_layout_sync(self):
        """Calculate layout synchronously with proper error handling."""
        try:
            algorithm = self.layout_options.algorithm
            start_time = time.time()
            
            if algorithm == "spring_enhanced_fast" and NUMBA_AVAILABLE and len(self.graph.nodes()) > 50:
                # Use JIT-optimized spring layout for medium-sized graphs
                nodes = list(self.graph.nodes())
                n = len(nodes)
                
                # Create adjacency matrix
                adj_matrix = nx.adjacency_matrix(self.graph, nodelist=nodes).toarray()
                
                # Initialize positions more thoughtfully
                pos_array = np.random.RandomState(42).random((n, 2)) * 10 - 5
                
                # Parameters
                k = self.layout_options.cluster_separation / np.sqrt(n) if n > 0 else 1.0
                iterations = min(self.layout_options.iterations, 50)
                
                # Optimize
                optimized_pos = optimize_spring_layout_step(pos_array, adj_matrix, k, iterations)
                
                # Convert to dict
                self.pos = {nodes[i]: (float(optimized_pos[i, 0]), float(optimized_pos[i, 1])) for i in range(n)}
                
            elif algorithm == "spring_enhanced":
                # Standard NetworkX spring layout with optimizations
                k_value = self.layout_options.cluster_separation / np.sqrt(len(self.graph.nodes()))
                self.pos = nx.spring_layout(
                    self.graph,
                    k=k_value,
                    iterations=min(self.layout_options.iterations, 100),
                    seed=self.layout_options.seed,
                    weight='weight'
                )
                
            elif algorithm == "circular_clustered":
                # Optimized circular clustered layout
                try:
                    communities = nx.community.greedy_modularity_communities(self.graph)
                    self.pos = {}
                    
                    if len(communities) > 0:
                        for i, community in enumerate(communities):
                            subgraph = self.graph.subgraph(community)
                            angle = 2 * np.pi * i / len(communities)
                            center_x = 3 * np.cos(angle)
                            center_y = 3 * np.sin(angle)
                            
                            if len(community) > 0:
                                sub_pos = nx.circular_layout(subgraph, scale=0.5)
                                for node, (x, y) in sub_pos.items():
                                    self.pos[node] = (x + center_x, y + center_y)
                    else:
                        # Fallback if no communities found
                        self.pos = nx.circular_layout(self.graph)
                except Exception as e:
                    logger.warning(f"Circular clustered layout failed: {e}, using fallback")
                    self.pos = nx.circular_layout(self.graph)
                    
            elif algorithm == "force_atlas":
                # Force Atlas style layout
                self.pos = nx.spring_layout(
                    self.graph,
                    k=2/np.sqrt(len(self.graph.nodes())),
                    iterations=min(self.layout_options.iterations * 2, 200),
                    seed=self.layout_options.seed,
                    weight='weight'
                )
                    
            elif algorithm == "community_based":
                # Layout based on community detection
                try:
                    communities = nx.community.greedy_modularity_communities(self.graph)
                    self.pos = {}
                    
                    if len(communities) > 0:
                        # Position communities in a grid
                        grid_size = int(np.ceil(np.sqrt(len(communities))))
                        
                        for i, community in enumerate(communities):
                            if len(community) == 0:
                                continue
                                
                            grid_x = (i % grid_size) * 4
                            grid_y = (i // grid_size) * 4
                            
                            subgraph = self.graph.subgraph(community)
                            sub_pos = nx.spring_layout(subgraph, scale=1.5, seed=42)
                            
                            for node, (x, y) in sub_pos.items():
                                self.pos[node] = (x + grid_x, y + grid_y)
                    else:
                        self.pos = nx.spring_layout(self.graph, seed=42)
                except Exception as e:
                    logger.warning(f"Community-based layout failed: {e}, using fallback")
                    self.pos = nx.spring_layout(self.graph, seed=42)
                    
            else:
                # Default fallback - always works
                self.pos = nx.spring_layout(
                    self.graph, 
                    seed=self.layout_options.seed,
                    iterations=min(self.layout_options.iterations, 100)
                )
            
            # Validate positions
            if not self.pos or len(self.pos) != len(self.graph.nodes()):
                logger.error(f"Layout calculation produced invalid positions: {len(self.pos) if self.pos else 0} for {len(self.graph.nodes())} nodes")
                # Emergency fallback
                self.pos = nx.spring_layout(self.graph, seed=42, iterations=30)
            
            # Cache the result with validation
            if self.pos and len(self.pos) == len(self.graph.nodes()):
                graph_hash = f"{len(self.graph.nodes())}_{len(self.graph.edges())}_{algorithm}_{self.layout_options.iterations}"
                self.performance_cache.set(f"layout_{graph_hash}", self.pos)
                
            calc_time = time.time() - start_time
            logger.info(f"Layout calculation complete in {calc_time:.2f}s. Generated positions for {len(self.pos)} nodes")
                
        except Exception as e:
            logger.error(f"Error calculating layout with {algorithm}: {e}")
            # Emergency fallback to ensure we always have valid positions
            try:
                self.pos = nx.spring_layout(self.graph, seed=42, iterations=30)
                logger.info(f"Used emergency fallback layout for {len(self.pos)} nodes")
            except Exception as e2:
                logger.error(f"Even fallback layout failed: {e2}")
                # Last resort: random positions
                self.pos = {node: (np.random.random(), np.random.random()) for node in self.graph.nodes()}
        
    def _create_enhanced_plotly_figure(self):
        """Create an enhanced Plotly figure with advanced interactivity."""
        if not self.graph or not self.pos:
            logger.warning("Cannot create figure: missing graph or position data")
            return
            
        # Validate that we have positions for all nodes
        if len(self.pos) != len(self.graph.nodes()):
            logger.error(f"Position mismatch: {len(self.pos)} positions for {len(self.graph.nodes())} nodes")
            # Force recalculation of layout
            self._calculate_enhanced_layout()
            if not self.pos or len(self.pos) != len(self.graph.nodes()):
                logger.error("Failed to generate valid positions, aborting figure creation")
                return
            
        # Apply filters
        filtered_nodes = self._get_filtered_nodes()
        
        if len(filtered_nodes) == 0:
            logger.warning("No nodes to display after filtering")
            return
            
        logger.info(f"Creating figure with {len(filtered_nodes)} filtered nodes out of {len(self.graph.nodes())} total")
        
        # Store the currently selected node if any, so we can restore it
        previously_selected = getattr(self, 'currently_selected_node', None)
        
        # Create traces
        traces = []
        
        # Enhanced edge traces with weight-based styling
        edge_traces = self._create_edge_traces(filtered_nodes)
        traces.extend(edge_traces)
        
        # Enhanced node traces with better labeling
        node_traces = self._create_node_traces(filtered_nodes)
        traces.extend(node_traces)
        
        if not traces:
            logger.error("No traces created, cannot generate figure")
            return
        
        # Create figure
        fig = go.Figure(data=traces)
        
        # Enhanced layout
        self._configure_enhanced_layout(fig, filtered_nodes)
        
        # Generate HTML with enhanced JavaScript
        self._generate_enhanced_html(fig, previously_selected)
        
    def _get_filtered_nodes(self) -> Set[str]:
        """Get nodes that pass current filters."""
        filtered = set()
        
        logger.info(f"Filtering {len(self.nodes)} nodes with filters: "
                   f"routers={self.filter_options.show_routers}, "
                   f"clients={self.filter_options.show_clients}, "
                   f"isolated={self.filter_options.show_isolated}, "
                   f"vulnerable={self.filter_options.show_vulnerable}, "
                   f"min_degree={self.filter_options.min_degree}, "
                   f"max_degree={self.filter_options.max_degree}")
        
        for node_id, node_data in self.nodes.items():
            degree = self.graph.degree(node_id)
            
            # Degree filter
            if not (self.filter_options.min_degree <= degree <= self.filter_options.max_degree):
                continue
                
            # Node type filters - be more permissive about router detection
            is_router = node_data.get('is_router', False) or node_data.get('role') == 'ROUTER'
            
            if is_router and not self.filter_options.show_routers:
                continue
            if not is_router and not self.filter_options.show_clients:
                continue
                
            # Special cases
            if degree == 0 and not self.filter_options.show_isolated:
                continue
            if degree == 1 and not self.filter_options.show_vulnerable:
                continue
                
            filtered.add(node_id)
            
        logger.info(f"After filtering: {len(filtered)} nodes will be displayed")
        return filtered
        
    def _create_edge_traces(self, filtered_nodes: Set[str]) -> List:
        """Create enhanced edge traces with weight-based styling."""
        if not filtered_nodes or not self.pos:
            logger.warning("Cannot create edge traces: no filtered nodes or positions")
            return []
            
        traces = []
        
        # Regular edges
        edge_x, edge_y = [], []
        critical_edge_x, critical_edge_y = [], []
        
        edges_processed = 0
        for edge in self.edges:
            if edge['from'] not in filtered_nodes or edge['to'] not in filtered_nodes:
                continue
                
            if edge['from'] not in self.pos or edge['to'] not in self.pos:
                continue
                
            try:
                x0, y0 = self.pos[edge['from']]
                x1, y1 = self.pos[edge['to']]
                
                # Convert to float to ensure compatibility
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                
                edges_processed += 1
                
                # Check if this is a critical edge
                is_critical = self._is_critical_edge(edge['from'], edge['to'])
                
                if is_critical and self.filter_options.highlight_critical_paths:
                    critical_edge_x.extend([x0, x1, None])
                    critical_edge_y.extend([y0, y1, None])
                else:
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
            except (TypeError, ValueError) as e:
                logger.debug(f"Invalid position data for edge {edge['from']}-{edge['to']}: {e}")
                continue
                
        logger.info(f"Processed {edges_processed} edges for visualization")
        
        # Regular edges trace
        if edge_x:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='rgba(150,150,150,0.6)'),
                hoverinfo='none',
                name='Connections',
                showlegend=True
            )
            traces.append(edge_trace)
            
        # Critical edges trace
        if critical_edge_x:
            critical_trace = go.Scatter(
                x=critical_edge_x, y=critical_edge_y,
                mode='lines',
                line=dict(width=3, color='rgba(255,69,0,0.8)'),
                hoverinfo='none',
                name='Critical Paths',
                showlegend=True
            )
            traces.append(critical_trace)
            
        return traces
        
    def _create_node_traces(self, filtered_nodes: Set[str]) -> List:
        """Create enhanced node traces with detailed information - PERFORMANCE OPTIMIZED."""
        start_time = time.time()
        
        # Validate inputs first
        if not filtered_nodes or not self.pos:
            logger.warning("Cannot create node traces: no filtered nodes or positions")
            return []
            
        # Check if all filtered nodes have positions
        nodes_with_positions = [node for node in filtered_nodes if node in self.pos]
        if len(nodes_with_positions) != len(filtered_nodes):
            logger.warning(f"Some filtered nodes missing positions: {len(nodes_with_positions)}/{len(filtered_nodes)}")
            filtered_nodes = set(nodes_with_positions)
            
        if not filtered_nodes:
            logger.warning("No nodes have positions available")
            return []
        
        # Create a more specific cache key that includes position data hash
        try:
            # Convert positions to a hashable format
            pos_items = [(node, (float(x), float(y))) for node, (x, y) in sorted(self.pos.items())]
            pos_hash = hash(tuple(pos_items))
        except (TypeError, ValueError):
            # Fallback if positions contain unhashable types
            pos_hash = len(self.pos)
            
        current_hash = self._generate_data_hash()
        cache_key = f"node_traces_{current_hash}_{len(filtered_nodes)}_{pos_hash}"
        
        # Only use cache if we have substantial nodes to avoid caching empty results
        if len(filtered_nodes) > 10:
            cached_traces = self.performance_cache.get(cache_key)
            if cached_traces is not None and len(cached_traces) > 0:
                logger.info(f"Using cached node traces ({len(cached_traces)} traces)")
                return cached_traces
        
        logger.info(f"Creating optimized node traces for {len(filtered_nodes)} filtered nodes")
        
        # Pre-allocate data structures for better performance
        node_groups = {
            'router': {'x': [], 'y': [], 'text': [], 'customdata': [], 'size': [], 'color': []},
            'client': {'x': [], 'y': [], 'text': [], 'customdata': [], 'size': [], 'color': []},
            'isolated': {'x': [], 'y': [], 'text': [], 'customdata': [], 'size': [], 'color': []},
            'vulnerable': {'x': [], 'y': [], 'text': [], 'customdata': [], 'size': [], 'color': []},
            'highlighted': {'x': [], 'y': [], 'text': [], 'customdata': [], 'size': [], 'color': []}
        }
        
        # Process nodes directly instead of using threading for better reliability
        nodes_with_positions = 0
        for node_id in filtered_nodes:
            if node_id not in self.pos:
                continue
                
            nodes_with_positions += 1
            node_data = self.nodes[node_id]
            x, y = self.pos[node_id]
            
            # Create optimized hover text (simplified for performance)
            hover_text = self._create_fast_hover_text(node_id, node_data)
            
            # Determine node size based on importance (optimized)
            degree = self.graph.degree(node_id)
            base_size = 12
            size_bonus = min(degree * 2, 20)
            node_size = base_size + size_bonus
            
            # Determine node group
            group = self._get_node_group(node_id, node_data, degree)
            
            # Add to group
            node_groups[group]['x'].append(x)
            node_groups[group]['y'].append(y)
            node_groups[group]['text'].append(hover_text)
            node_groups[group]['customdata'].append(node_id)
            node_groups[group]['size'].append(node_size)
            node_groups[group]['color'].append('#000000')  # Placeholder
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {nodes_with_positions} nodes with valid positions in {processing_time:.2f}s")
        
        # Create traces for each group - optimized
        traces = []
        group_config = {
            'router': {'color': '#1E88E5', 'name': 'Routers'},
            'client': {'color': '#43A047', 'name': 'Clients'},
            'isolated': {'color': '#E53935', 'name': 'Isolated'},
            'vulnerable': {'color': '#FDD835', 'name': 'Vulnerable'},
            'highlighted': {'color': '#FF6F00', 'name': 'Highlighted'}
        }
        
        total_nodes_in_traces = 0
        for group, data in node_groups.items():
            if not data['x']:
                continue
                
            total_nodes_in_traces += len(data['x'])
            config = group_config[group]
            
            # Generate node labels
            node_labels = [self._get_node_label(cdata) for cdata in data['customdata']]
            
            # Create optimized trace with pre-computed data
            node_trace = go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers+text' if self.filter_options.show_node_labels else 'markers',
                marker=dict(
                    size=data['size'],
                    color=config['color'],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=node_labels,
                textposition="middle center",
                textfont=dict(
                    size=self.filter_options.node_label_size, 
                    color='white',
                    family='Arial, sans-serif'
                ),
                hovertext=data['text'],
                hoverinfo='text',
                customdata=data['customdata'],
                name=config['name'],
                showlegend=True
            )
            traces.append(node_trace)
            logger.info(f"Created {group} trace with {len(data['x'])} nodes")
        
        total_time = time.time() - start_time
        logger.info(f"Created {len(traces)} node traces with {total_nodes_in_traces} total nodes in {total_time:.2f}s")
        
        # Only cache if we have substantial results
        if len(traces) > 0 and total_nodes_in_traces > 10:
            self.performance_cache.set(cache_key, traces)
        
        return traces
    
    def _create_fast_hover_text(self, node_id: str, node_data: Dict) -> str:
        """Create optimized hover text for performance."""
        # Simplified hover text for better performance
        hex_name = node_data.get('hex_name', node_id)
        short_name = node_data.get('short_name', '')
        hardware = node_data.get('hardware', '')
        degree = self.graph.degree(node_id)
        
        # Build text efficiently
        parts = [f"<b>{hex_name}</b>"]
        
        if short_name and short_name.strip():
            parts.append(f"Name: {short_name}")
        
        if hardware:
            parts.append(f"HW: {hardware}")
            
        parts.append(f"Connections: {degree}")
        
        # Add neighbor count only (not full list for performance)
        if degree > 0:
            parts.append(f"Neighbors: {degree}")
        
        return "<br>".join(parts)
        
    def _create_node_hover_text(self, node_id: str, node_data: Dict) -> str:
        """Create detailed hover text for a node."""
        # Use hex_name for display if available
        hex_name = node_data.get('hex_name', node_id)
        original_name = node_data.get('original_name', '')
        
        # Create title with hex name
        text = f"<b>{hex_name}</b><br>"
        
        # Show original numeric name if different
        if original_name and original_name != hex_name:
            text += f"<b>Original ID:</b> {original_name}<br>"
        
        # Basic node info
        short_name = node_data.get('short_name', '')
        long_name = node_data.get('long_name', '')
        
        if short_name and short_name.strip():
            text += f"<b>Short Name:</b> {short_name}<br>"
        if long_name and long_name.strip() and long_name != short_name:
            text += f"<b>Long Name:</b> {sanitize_label(long_name)}<br>"
            
        # Node type
        is_router = node_data.get('is_router', False) or node_data.get('role') == 'ROUTER'
        text += f"<b>Type:</b> {'Router' if is_router else 'Client'}<br>"
        
        # Hardware and channel info
        hardware = node_data.get('hardware', '')
        if hardware:
            text += f"<b>Hardware:</b> {hardware}<br>"
        
        channel = node_data.get('channel', '')
        if channel:
            text += f"<b>Channel:</b> {channel}<br>"
        
        # Connection info
        degree = self.graph.degree(node_id)
        text += f"<b>Connections:</b> {degree}<br>"
        
        # Neighbor information with quality
        neighbors = list(self.graph.neighbors(node_id))
        if neighbors:
            text += f"<b>Connected to:</b><br>"
            for i, n_id in enumerate(neighbors[:5]):  # Show first 5
                n_data = self.nodes.get(n_id, {})
                n_hex_name = n_data.get('hex_name', n_id)
                n_short_name = n_data.get('short_name', '')
                
                # Use short name if available, otherwise hex name
                display_name = n_short_name if n_short_name and n_short_name.strip() else n_hex_name[:8]
                
                # Get edge weight/SNR if available
                edge_data = self.graph.get_edge_data(node_id, n_id, {})
                snr = edge_data.get('snr')
                if snr is not None:
                    text += f"  • {display_name} (SNR: {snr:.1f})<br>"
                else:
                    text += f"  • {display_name}<br>"
            if len(neighbors) > 5:
                text += f"  • ... and {len(neighbors) - 5} more<br>"
        
        # Optimization scores
        if self.node_scores and node_id in self.node_scores:
            score = self.node_scores[node_id]
            text += f"<br><b>Optimization Scores:</b><br>"
            text += f"  Total: {score.total_score:.3f}<br>"
            text += f"  Coverage: {score.coverage_score:.3f}<br>"
            text += f"  Centrality: {score.centrality_score:.3f}<br>"
            text += f"  Redundancy: {score.redundancy_score:.3f}<br>"
            
        # Network metrics
        if nx.is_connected(self.graph):
            try:
                centrality = nx.betweenness_centrality(self.graph).get(node_id, 0)
                text += f"<br><b>Betweenness Centrality:</b> {centrality:.3f}<br>"
            except:
                pass
                
        return text
        
    def _get_node_label(self, node_id: str) -> str:
        """Get display label for a node."""
        if not self.filter_options.show_node_labels:
            return ""
            
        node_data = self.nodes.get(node_id, {})
        
        # Use hex_name if available, otherwise use short_name or the node_id itself
        hex_name = node_data.get('hex_name', node_id)
        short_name = node_data.get('short_name', '')
        
        # Prefer short_name if it exists and is meaningful, otherwise use hex
        if short_name and short_name.strip() and short_name != 'Meshtastic':
            # For display, show short name with hex in parentheses
            return f"{short_name[:6]}"
        else:
            # Show hex name, truncated for display
            return hex_name[:8] if len(hex_name) > 8 else hex_name
        
    def _get_node_group(self, node_id: str, node_data: Dict, degree: int) -> str:
        """Determine which group a node belongs to for styling."""
        if node_id in self.highlight_nodes:
            return 'highlighted'
        elif degree == 0:
            return 'isolated'
        elif degree == 1:
            return 'vulnerable'
        elif node_data.get('is_router'):
            return 'router'
        else:
            return 'client'
            
    def _is_critical_edge(self, node1: str, node2: str) -> bool:
        """Check if an edge is on a critical path."""
        if not self.critical_paths:
            return False
            
        for path in self.critical_paths:
            for i in range(len(path) - 1):
                if (path[i] == node1 and path[i+1] == node2) or \
                   (path[i] == node2 and path[i+1] == node1):
                    return True
        return False
        
    def _configure_enhanced_layout(self, fig, filtered_nodes: Set[str]):
        """Configure enhanced layout with better controls."""
        node_positions = [self.pos[node] for node in filtered_nodes if node in self.pos]
        if not node_positions:
            x_range, y_range = [-1, 1], [-1, 1]
        else:
            x_coords, y_coords = zip(*node_positions)
            x_margin = (max(x_coords) - min(x_coords)) * 0.1
            y_margin = (max(y_coords) - min(y_coords)) * 0.1
            x_range = [min(x_coords) - x_margin, max(x_coords) + x_margin]
            y_range = [min(y_coords) - y_margin, max(y_coords) + y_margin]
        
        fig.update_layout(
            title=dict(
                text=f'Enhanced Meshtastic Network ({len(filtered_nodes)} nodes visible)',
                font=dict(size=18)
            ),
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="Black", borderwidth=1
            ),
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),
            xaxis=dict(
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                zeroline=False, showticklabels=False,
                range=x_range
            ),
            yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                zeroline=False, showticklabels=False,
                range=y_range
            ),
            plot_bgcolor='white',
            dragmode='pan',
            clickmode='event+select',
            uirevision='constant',
            height=800
        )
        
        # Enhanced control buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(args=[{"dragmode": "pan"}], label="Pan", method="relayout"),
                        dict(args=[{"dragmode": "zoom"}], label="Zoom", method="relayout"),
                        dict(args=[{"dragmode": "select"}], label="Select", method="relayout"),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0, xanchor="left",
                    y=1.05, yanchor="top"
                ),
            ]
        )
        
    def _generate_enhanced_html(self, fig, restore_selected_node=None):
        """Generate HTML with enhanced JavaScript for better interactivity."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            html_str = fig.to_html(
                include_plotlyjs='inline',  # Use inline instead of CDN to avoid loading issues
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'doubleClick': False  # Disable built-in double-click behavior
                }
            )
            
            # Add the previously selected node info to JavaScript if available
            restore_js = ""
            if restore_selected_node:
                restore_js = f"let restoreSelectedNode = '{restore_selected_node}';"
            else:
                restore_js = "let restoreSelectedNode = null;"
            
            # Add enhanced JavaScript for click handling and node highlighting
            enhanced_js = """
            <script>
                // Global variables
                let selectedNodeId = null;
                let plotDiv = null;
                let nodeConnections = {};
                let originalTraces = null;
                let retryCount = 0;
                let maxRetries = 20;
                """ + restore_js + """
                
                // Main initialization
                function initializeEnhancedClicking() {
                    console.log('Starting enhanced click initialization, attempt:', retryCount + 1);
                    
                    // Try multiple selectors to find the plot
                    plotDiv = document.querySelector('.plotly-graph-div') || 
                             document.querySelector('[id*="plotly"]') ||
                             document.querySelector('div[data-plotly]');
                    
                    if (!plotDiv) {
                        console.log('Plot div not found, will retry...');
                        if (retryCount < maxRetries) {
                            retryCount++;
                            setTimeout(initializeEnhancedClicking, 1000);
                        }
                        return;
                    }
                    
                    console.log('Found plot div:', plotDiv);
                    
                    // Check if Plotly is available and plot has data
                    if (typeof Plotly === 'undefined') {
                        console.log('Plotly not loaded yet, will retry...');
                        if (retryCount < maxRetries) {
                            retryCount++;
                            setTimeout(initializeEnhancedClicking, 1000);
                        }
                        return;
                    }
                    
                    // Check if plot has data
                    if (!plotDiv.data || plotDiv.data.length === 0) {
                        console.log('Plot data not ready, will retry...');
                        if (retryCount < maxRetries) {
                            retryCount++;
                            setTimeout(initializeEnhancedClicking, 1000);
                        }
                        return;
                    }
                    
                    console.log('Plot ready with', plotDiv.data.length, 'traces');
                    
                    try {
                        // Store original data before any modifications - ensure deep copy
                        originalTraces = JSON.parse(JSON.stringify(plotDiv.data));
                        console.log('Stored original traces:', originalTraces.length, 'traces');
                        
                        // Validate that original traces were stored correctly
                        let nodeTraceCount = 0;
                        let edgeTraceCount = 0;
                        for (let i = 0; i < originalTraces.length; i++) {
                            if (originalTraces[i].mode && originalTraces[i].mode.includes('markers')) {
                                nodeTraceCount++;
                            }
                            if (originalTraces[i].mode && originalTraces[i].mode.includes('lines')) {
                                edgeTraceCount++;
                            }
                        }
                        console.log('Original traces contain:', nodeTraceCount, 'node traces,', edgeTraceCount, 'edge traces');
                        
                        // Build node connections
                        buildNodeConnections();
                        
                        // Set up event handlers
                        setupEventHandlers();
                        
                        console.log('Enhanced clicking initialized successfully!');
                        console.log('Node connections built for', Object.keys(nodeConnections).length, 'nodes');
                        
                        // Add visual indicator that enhanced clicking is active
                        addStatusIndicator();
                        
                        // Restore previously selected node if any
                        if (restoreSelectedNode && nodeConnections[restoreSelectedNode]) {
                            console.log('Restoring selection for:', restoreSelectedNode);
                            setTimeout(() => {
                                highlightNodeAndConnections(restoreSelectedNode);
                                selectedNodeId = restoreSelectedNode;
                            }, 500);
                        }
                        
                    } catch (error) {
                        console.error('Error during initialization:', error);
                        if (retryCount < maxRetries) {
                            retryCount++;
                            setTimeout(initializeEnhancedClicking, 2000);
                        }
                    }
                }
                
                function addStatusIndicator() {
                    let indicator = document.createElement('div');
                    indicator.id = 'enhanced-click-indicator';
                    indicator.innerHTML = '✓ Enhanced Click Active';
                    indicator.style.cssText = `
                        position: fixed; 
                        top: 10px; 
                        left: 10px; 
                        background: #4CAF50; 
                        color: white; 
                        padding: 5px 10px; 
                        border-radius: 3px; 
                        font-size: 12px; 
                        z-index: 10000;
                        font-family: Arial, sans-serif;
                    `;
                    document.body.appendChild(indicator);
                    
                    // Remove after 3 seconds
                    setTimeout(() => {
                        if (indicator.parentNode) {
                            indicator.parentNode.removeChild(indicator);
                        }
                    }, 3000);
                }
                
                function buildNodeConnections() {
                    nodeConnections = {};
                    
                    // Find all node traces and store their info
                    for (let i = 0; i < plotDiv.data.length; i++) {
                        let trace = plotDiv.data[i];
                        
                        // Node traces have markers and customdata
                        if (trace.mode && trace.mode.includes('markers') && trace.customdata) {
                            console.log('Found node trace', i, 'with', trace.customdata.length, 'nodes');
                            for (let j = 0; j < trace.customdata.length; j++) {
                                let nodeId = trace.customdata[j];
                                if (nodeId) {
                                    nodeConnections[nodeId] = {
                                        traceIndex: i,
                                        pointIndex: j,
                                        x: trace.x[j],
                                        y: trace.y[j],
                                        neighbors: new Set()
                                    };
                                }
                            }
                        }
                    }
                    
                    console.log('Found nodes:', Object.keys(nodeConnections));
                    
                    // Build connections from edge traces
                    let edgeTraceCount = 0;
                    let totalConnections = 0;
                    
                    for (let i = 0; i < plotDiv.data.length; i++) {
                        let trace = plotDiv.data[i];
                        
                        // Edge traces have lines
                        if (trace.mode && trace.mode.includes('lines') && trace.x && trace.y) {
                            edgeTraceCount++;
                            console.log('Processing edge trace', i, 'with', trace.x.length, 'coordinates');
                            
                            // Process edge coordinates (x1, x2, null, x3, x4, null, ...)
                            let edgeCount = 0;
                            for (let j = 0; j < trace.x.length - 1; j += 3) {
                                let x1 = trace.x[j];
                                let y1 = trace.y[j];
                                let x2 = trace.x[j + 1];
                                let y2 = trace.y[j + 1];
                                
                                if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {
                                    let node1 = findNodeByCoordinates(x1, y1);
                                    let node2 = findNodeByCoordinates(x2, y2);
                                    
                                    if (node1 && node2 && node1 !== node2) {
                                        nodeConnections[node1].neighbors.add(node2);
                                        nodeConnections[node2].neighbors.add(node1);
                                        edgeCount++;
                                        totalConnections++;
                                        
                                        if (edgeCount <= 3) { // Log first few edges
                                            console.log('  Connected:', node1, '<->', node2);
                                        }
                                    }
                                }
                            }
                            console.log('  Found', edgeCount, 'connections in this trace');
                        }
                    }
                    
                    // Convert Sets to Arrays for easier handling
                    for (let nodeId in nodeConnections) {
                        nodeConnections[nodeId].neighbors = Array.from(nodeConnections[nodeId].neighbors);
                    }
                    
                    console.log('Processed', edgeTraceCount, 'edge traces');
                    console.log('Total connections built:', totalConnections);
                    console.log('Sample connections:');
                    let sampleCount = 0;
                    for (let nodeId in nodeConnections) {
                        if (nodeConnections[nodeId].neighbors.length > 0 && sampleCount < 5) {
                            console.log('  ', nodeId, 'connects to:', nodeConnections[nodeId].neighbors);
                            sampleCount++;
                        }
                    }
                }
                
                function findNodeByCoordinates(x, y, tolerance = 0.001) {
                    let matches = [];
                    for (let nodeId in nodeConnections) {
                        let node = nodeConnections[nodeId];
                        let distance = Math.sqrt(Math.pow(node.x - x, 2) + Math.pow(node.y - y, 2));
                        if (distance < tolerance) {
                            matches.push({nodeId: nodeId, distance: distance});
                        }
                    }
                    
                    if (matches.length === 0) {
                        return null;
                    } else if (matches.length === 1) {
                        return matches[0].nodeId;
                    } else {
                        // Multiple matches - return the closest one
                        matches.sort((a, b) => a.distance - b.distance);
                        console.log('Multiple coordinate matches for', x, y, ':', matches.map(m => m.nodeId));
                        return matches[0].nodeId;
                    }
                }
                
                function setupEventHandlers() {
                    // Remove any existing handlers first
                    if (plotDiv._clickHandler) {
                        plotDiv.removeListener('plotly_click', plotDiv._clickHandler);
                    }
                    if (plotDiv._doubleClickHandler) {
                        plotDiv.removeListener('plotly_doubleclick', plotDiv._doubleClickHandler);
                    }
                    if (plotDiv._relayoutHandler) {
                        plotDiv.removeListener('plotly_relayout', plotDiv._relayoutHandler);
                    }
                    
                    // Simple click handling
                    plotDiv._clickHandler = handleNodeClick;
                    plotDiv._doubleClickHandler = clearSelection;
                    plotDiv._relayoutHandler = function() {
                        if (selectedNodeId) {
                            console.log('Relayout event detected, re-highlighting selected node:', selectedNodeId);
                            highlightNodeAndConnections(selectedNodeId);
                        }
                    };
                    
                    plotDiv.on('plotly_click', plotDiv._clickHandler);
                    plotDiv.on('plotly_doubleclick', plotDiv._doubleClickHandler);
                    plotDiv.on('plotly_relayout', plotDiv._relayoutHandler);
                    
                    console.log('Event handlers (click, doubleclick, relayout) set up');
                }
                
                function handleNodeClick(data) {
                    console.log('=== HANDLING SINGLE CLICK ===');
                    
                    if (!data.points || data.points.length === 0) {
                        console.log('No points in click data - clicked on empty space, clearing selection');
                        if (selectedNodeId) {
                            selectedNodeId = null;
                            clearSelection();
                            sendNodeToDetails(null);
                        }
                        return;
                    }
                    
                    let point = data.points[0];
                    let clickedNodeId = point.customdata;
                    
                    console.log('Clicked node:', clickedNodeId);
                    console.log('Current selected node:', selectedNodeId);
                    
                    if (!clickedNodeId || !nodeConnections[clickedNodeId]) {
                        console.log('Node not found in connections map, treating as empty space click');
                        if (selectedNodeId) {
                            selectedNodeId = null;
                            clearSelection();
                            sendNodeToDetails(null);
                        }
                        return;
                    }
                    
                    if (selectedNodeId === clickedNodeId) {
                        console.log('Same node clicked, clearing selection');
                        selectedNodeId = null;
                        clearSelection();
                        sendNodeToDetails(null);
                    } else {
                        console.log('New node selected, highlighting');
                        selectedNodeId = clickedNodeId;
                        highlightNodeAndConnections(clickedNodeId);
                        sendNodeToDetails(clickedNodeId);
                    }
                    
                    console.log('Updated selected node to:', selectedNodeId);
                }
                
                function sendNodeToDetails(nodeId) {
                    // Send node ID to Python side for details update
                    // We'll trigger this by changing the page title which Python can detect
                    let message = nodeId ? `SHOW_NODE_DETAILS:${nodeId}` : `CLEAR_NODE_DETAILS`;
                    console.log('Sending to Python:', message);
                    document.title = message;
                }
                
                // ensureLabelsVisible function was removed
                
                function highlightNodeAndConnections(nodeId) {
                    console.log('=== HIGHLIGHTING NODE (Prioritizing Original Labels) ===');
                    console.log('Selected node:', nodeId);
                    
                    let neighbors = nodeConnections[nodeId].neighbors;
                    let connectedNodes = new Set([nodeId, ...neighbors]);
                    
                    let newData = [];
                    
                    for (let i = 0; i < originalTraces.length; i++) {
                        let trace = originalTraces[i];
                        
                        if (trace.mode && trace.mode.includes('markers') && trace.customdata) {
                            let newTrace = JSON.parse(JSON.stringify(trace));
                            let markerColors = [];
                            let markerOpacities = [];
                            let texts = [];
                            let currentTraceTextPositions = [];
                            let currentTraceTextFonts = []; 
                            let newMode = trace.mode; 

                            for (let j = 0; j < trace.customdata.length; j++) {
                                let currentNodeId = trace.customdata[j];
                                let labelToShow = '';

                                // Prioritize original label (should be short_name if available from Python)
                                if (originalTraces[i] && originalTraces[i].text && originalTraces[i].text[j] && originalTraces[i].text[j].trim() !== '') {
                                    labelToShow = originalTraces[i].text[j];
                                } else if (currentNodeId) { // Fallback to currentNodeId if original text is missing
                                    labelToShow = currentNodeId.substring(0, 8);
                                }

                                let textFont = {
                                    size: (trace.textfont && trace.textfont.size) ? trace.textfont.size : 12,
                                    color: 'rgba(0,0,0,1)', 
                                    outlinecolor: 'rgba(255,255,255,0.7)',
                                    outlinewidth: 2
                                }; 

                                if (currentNodeId === nodeId) { // Selected node
                                    markerColors.push('#FF6F00'); 
                                    markerOpacities.push(1.0);
                                    texts.push(labelToShow); 
                                } else if (connectedNodes.has(currentNodeId)) { // Connected node
                                    markerColors.push(trace.marker.color); 
                                    markerOpacities.push(1.0);
                                    texts.push(labelToShow); 
                                } else { // Unconnected node
                                    markerColors.push('rgba(200,200,200,0.3)');
                                    markerOpacities.push(0.3);
                                    texts.push(''); 
                                    textFont.color = 'rgba(0,0,0,0)'; 
                                    textFont.outlinecolor = 'rgba(0,0,0,0)'; 
                                }
                                currentTraceTextFonts.push(textFont);
                                currentTraceTextPositions.push(trace.textposition ? trace.textposition[j] || trace.textposition : "middle center");
                            }
                            
                            newTrace.marker.color = markerColors;
                            newTrace.marker.opacity = markerOpacities;
                            newTrace.text = texts;
                            newTrace.textfont = currentTraceTextFonts; 
                            newTrace.textposition = currentTraceTextPositions;

                            const hasVisibleText = texts.some(t => t !== '');
                            if (hasVisibleText) {
                                if (!newMode.includes('text')) {
                                    newMode = newMode + '+text';
                                }
                            } else {
                                if (newMode.includes('text')) {
                                    newMode = newMode.replace('+text', '').replace('text+', '');
                                    if (newMode === 'text') newMode = 'markers';
                                }
                            }
                            newTrace.mode = newMode;
                            
                            newData.push(newTrace);
                        }
                    }
                    
                    // Edge handling (remains the same)
                    let highlightedEdges = {x: [], y: []};
                    let dimmedEdges = {x: [], y: []};
                    for (let i = 0; i < originalTraces.length; i++) {
                        let trace = originalTraces[i];
                        if (trace.mode && trace.mode.includes('lines') && trace.x && trace.y) {
                            for (let j = 0; j < trace.x.length - 1; j += 3) {
                                let x1 = trace.x[j], y1 = trace.y[j];
                                let x2 = trace.x[j+1], y2 = trace.y[j+1];
                                if (x1 !== null && x2 !== null) {
                                    let node1 = findNodeByCoordinates(x1, y1);
                                    let node2 = findNodeByCoordinates(x2, y2);
                                    let isConnectedEdge = false;
                                    if (node1 && node2) {
                                        if ((node1 === nodeId && neighbors.includes(node2)) || (node2 === nodeId && neighbors.includes(node1))) {
                                            isConnectedEdge = true;
                                        }
                                    }
                                    if (isConnectedEdge) {
                                        highlightedEdges.x.push(x1, x2, null);
                                        highlightedEdges.y.push(y1, y2, null);
                                    } else {
                                        dimmedEdges.x.push(x1, x2, null);
                                        dimmedEdges.y.push(y1, y2, null);
                                    }
                                }
                            }
                        }
                    }
                    if (highlightedEdges.x.length > 0) {
                        newData.push({
                            x: highlightedEdges.x, y: highlightedEdges.y, mode: 'lines',
                            line: {color: '#FF6F00', width: 3}, hoverinfo: 'none', name: 'Connected Edges', showlegend: false
                        });
                    }
                    if (dimmedEdges.x.length > 0) {
                        newData.push({
                            x: dimmedEdges.x, y: dimmedEdges.y, mode: 'lines',
                            line: {color: 'rgba(200,200,200,0.2)', width: 0.5}, hoverinfo: 'none', name: 'Other Edges', showlegend: false
                        });
                    }
                    
                    Plotly.react(plotDiv, newData, plotDiv.layout, plotDiv.config);
                    console.log('Applied highlighting, prioritizing original (short_name) labels.');
                }
                
                // Debug function
                function debugPlot() {
                    console.log('=== Debug Info ===');
                    console.log('Plot div:', plotDiv);
                    console.log('Plotly available:', typeof Plotly !== 'undefined');
                    console.log('Plot data available:', plotDiv && plotDiv.data ? plotDiv.data.length : 'No');
                    console.log('Node connections:', Object.keys(nodeConnections).length);
                    console.log('Selected node:', selectedNodeId);
                    console.log('Node connections map:', nodeConnections);
                }
                
                // Make debug available globally
                window.debugPlot = debugPlot;
                
                // Start initialization when page is ready
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', () => {
                        setTimeout(initializeEnhancedClicking, 1000);
                    });
                } else {
                    setTimeout(initializeEnhancedClicking, 1000);
                }
                
                function clearSelection() {
                    console.log('=== CLEARING SELECTION ===');
                    console.log('Previous selected node:', selectedNodeId);
                    selectedNodeId = null;
                    
                    if (!originalTraces || originalTraces.length === 0) {
                        console.log('No original traces to restore, rebuilding from current data');
                        // If no original traces, try to restore visibility to all elements
                        if (plotDiv && plotDiv.data) {
                            let restoredData = [];
                            for (let i = 0; i < plotDiv.data.length; i++) {
                                let trace = JSON.parse(JSON.stringify(plotDiv.data[i]));
                                
                                // Restore node visibility and colors
                                if (trace.mode && trace.mode.includes('markers') && trace.customdata) {
                                    // Reset all nodes to visible with original colors
                                    let numNodes = trace.customdata.length;
                                    trace.marker.opacity = new Array(numNodes).fill(0.8);
                                    
                                    // Reset text visibility based on filter settings
                                    if (trace.text) {
                                        // Ensure text mode is enabled if we have text
                                        if (!trace.mode.includes('text')) {
                                            trace.mode = trace.mode + '+text';
                                        }
                                        // Reset text fonts to visible
                                        if (trace.textfont && Array.isArray(trace.textfont)) {
                                            for (let j = 0; j < trace.textfont.length; j++) {
                                                trace.textfont[j].color = trace.textfont[j].color || 'rgba(0,0,0,1)';
                                            }
                                        }
                                    }
                                }
                                
                                // Restore edge visibility
                                if (trace.mode && trace.mode.includes('lines')) {
                                    trace.line.color = trace.line.color || 'rgba(150,150,150,0.6)';
                                    trace.line.width = trace.line.width || 1;
                                }
                                
                                restoredData.push(trace);
                            }
                            
                            try {
                                Plotly.react(plotDiv, restoredData, plotDiv.layout, plotDiv.config);
                                console.log('Restored visibility without original traces');
                            } catch (error) {
                                console.error('Failed to restore without original traces:', error);
                            }
                        }
                        sendNodeToDetails(null);
                        return;
                    }
                    
                    // Restore original data
                    try {
                        let restoredTraces = JSON.parse(JSON.stringify(originalTraces));
                        Plotly.react(plotDiv, restoredTraces, plotDiv.layout, plotDiv.config);
                        console.log('Restored original state successfully with', restoredTraces.length, 'traces');
                        console.log('Selected node after clear:', selectedNodeId);
                    } catch (error) {
                        console.error('Error restoring state:', error);
                        // Fallback: try direct data update
                        try {
                            plotDiv.data = JSON.parse(JSON.stringify(originalTraces));
                            Plotly.redraw(plotDiv);
                            console.log('Restored using redraw');
                        } catch (error2) {
                            console.error('Fallback restore also failed:', error2);
                            // Last resort: reload the page
                            console.log('Attempting to rebuild visualization...');
                            if (typeof rebuildVisualization === 'function') {
                                rebuildVisualization();
                            }
                        }
                    }
                    
                    // Clear node details
                    sendNodeToDetails(null);
                }
                
            </script>
            """
            
            # Insert the enhanced JavaScript before the closing body tag
            enhanced_html = html_str.replace('</body>', f'{enhanced_js}</body>')
            f.write(enhanced_html)
            self.current_html = f.name
            
        # Load in web view
        self.web_view.setUrl(QUrl.fromLocalFile(self.current_html))
        self.web_view.loadFinished.connect(self._on_load_finished)
        
    # Event handlers for UI controls
    def _on_layout_changed(self, algorithm: str):
        """Handle layout algorithm change."""
        self.layout_options.algorithm = algorithm
        self._recalculate_layout()
        
    def _on_iterations_changed(self, value: int):
        """Handle iterations slider change."""
        self.layout_options.iterations = value
        self.iterations_label.setText(str(value))
        
    def _on_separation_changed(self, value: int):
        """Handle separation slider change."""
        self.layout_options.cluster_separation = value / 100.0
        self.separation_label.setText(f"{value/100:.1f}")
        
    def _on_filter_changed(self):
        """Handle filter changes."""
        self.filter_options.show_routers = self.show_routers_cb.isChecked()
        self.filter_options.show_clients = self.show_clients_cb.isChecked()
        self.filter_options.show_isolated = self.show_isolated_cb.isChecked()
        self.filter_options.show_vulnerable = self.show_vulnerable_cb.isChecked()
        self.filter_options.highlight_critical_paths = self.highlight_critical_cb.isChecked()
        self.filter_options.min_degree = self.min_degree_spin.value()
        self.filter_options.max_degree = self.max_degree_spin.value()
        self.filter_options.show_node_labels = self.show_labels_cb.isChecked()
        
        if hasattr(self, 'graph') and self.graph:
            self._create_enhanced_plotly_figure()
            
    def _on_label_size_changed(self, value: int):
        """Handle label size change."""
        self.filter_options.node_label_size = value
        self.label_size_label.setText(str(value))
        if hasattr(self, 'graph') and self.graph:
            self._create_enhanced_plotly_figure()
            
    def _recalculate_layout(self):
        """Recalculate layout with current settings."""
        if hasattr(self, 'graph') and self.graph:
            self.pos = None
            self._calculate_enhanced_layout()
            self._create_enhanced_plotly_figure()
            
    def _find_critical_paths(self):
        """Find and highlight critical paths in the network."""
        if not self.graph or len(self.graph.nodes()) < 2:
            return
            
        try:
            # Find paths between important nodes (high degree nodes)
            degrees = dict(self.graph.degree())
            important_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            self.critical_paths = []
            
            for i, (node1, _) in enumerate(important_nodes):
                for node2, _ in important_nodes[i+1:]:
                    try:
                        if nx.has_path(self.graph, node1, node2):
                            path = nx.shortest_path(self.graph, node1, node2)
                            if len(path) > 2:  # Only paths with intermediate nodes
                                self.critical_paths.append(path)
                    except:
                        continue
                        
            logger.info(f"Found {len(self.critical_paths)} critical paths")
            
            # Update visualization if critical path highlighting is enabled
            if self.filter_options.highlight_critical_paths:
                self._create_enhanced_plotly_figure()
                
        except Exception as e:
            logger.error(f"Error finding critical paths: {e}")
            
    def _center_on_selected(self):
        """Center the view on selected nodes."""
        if self.selected_nodes and self.pos:
            # Calculate center of selected nodes
            positions = [self.pos[node] for node in self.selected_nodes if node in self.pos]
            if positions:
                x_coords, y_coords = zip(*positions)
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                logger.info(f"Centering on selected nodes at ({center_x:.2f}, {center_y:.2f})")
                
    def _on_load_finished(self, success: bool):
        """Handle when the page finishes loading."""
        if success:
            logger.info("Enhanced interactive plot loaded successfully")
        else:
            logger.error("Failed to load enhanced interactive plot")
            
    def _on_title_changed(self, title: str):
        """Handle title changes from JavaScript for node details communication."""
        if title.startswith("SHOW_NODE_DETAILS:"):
            node_id = title.replace("SHOW_NODE_DETAILS:", "")
            logger.info(f"JavaScript requested node details for: {node_id}")
            self.update_node_details(node_id)
        elif title == "CLEAR_NODE_DETAILS":
            logger.info("JavaScript requested clearing node details")
            self.node_details.setPlainText("Click a node to view its details...")
            
    def _export_html(self):
        """Export the visualization to an HTML file."""
        if not self.current_html or not os.path.exists(self.current_html):
            logger.warning("No visualization to export")
            return
            
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Enhanced Visualization", "enhanced_network_visualization.html", 
            "HTML Files (*.html)"
        )
        
        if filename:
            import shutil
            shutil.copy(self.current_html, filename)
            logger.info(f"Exported enhanced visualization to {filename}")
            
            # Try to open in browser
            import webbrowser
            webbrowser.open(f"file://{filename}")
            
    def update_node_details(self, node_id: str):
        """Update the node details panel."""
        if not node_id or node_id not in self.nodes:
            self.node_details.setPlainText("Click a node to view its details...")
            return
            
        node_data = self.nodes[node_id]
        details = self._create_node_hover_text(node_id, node_data)
        
        # Convert to plain text for the text widget
        import re
        details = re.sub(r'<br>', '\n', details)
        details = re.sub(r'<[^>]+>', '', details)  # Remove HTML tags
        
        self.node_details.setPlainText(details)
            
    def clear_highlights(self):
        """Clear all highlights and redraw."""
        self.highlight_nodes = set()
        self.selected_nodes = set()
        self.critical_paths = []
        if hasattr(self, 'graph') and self.graph:
            self._create_enhanced_plotly_figure()
            
    def handle_node_click(self, node_id: str):
        """Handle node click events from the visualization."""
        if node_id in self.nodes:
            # Update selected nodes
            if node_id in self.selected_nodes:
                self.selected_nodes.remove(node_id)
                self.currently_selected_node = None
            else:
                self.selected_nodes = {node_id}  # Single selection for now
                self.currently_selected_node = node_id
            
            # Update highlighted nodes to include the selected node and its neighbors
            if node_id in self.selected_nodes:
                neighbors = set(self.graph.neighbors(node_id)) if self.graph else set()
                self.highlight_nodes = {node_id} | neighbors
            else:
                self.highlight_nodes = set()
            
            # Emit signal for external handling
            self.node_clicked.emit(node_id)
            
            logger.info(f"Node clicked: {node_id}, neighbors: {len(self.highlight_nodes) - 1 if self.highlight_nodes else 0}")
        
    def get_node_neighbors(self, node_id: str) -> Set[str]:
        """Get the neighbors of a specific node."""
        if self.graph and node_id in self.graph:
            return set(self.graph.neighbors(node_id))
        return set()
        
    def set_selected_node(self, node_id: str):
        """Set the currently selected node (called from JavaScript or elsewhere)."""
        if node_id in self.nodes:
            self.currently_selected_node = node_id
            self.selected_nodes = {node_id}
            neighbors = set(self.graph.neighbors(node_id)) if self.graph else set()
            self.highlight_nodes = {node_id} | neighbors
        
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

    def _generate_data_hash(self):
        """Generate a hash of the current data for caching."""
        import hashlib
        
        # Create a safe string representation of the data state
        try:
            # Convert filter options to a safe string representation
            filter_str = str({
                'min_degree': self.filter_options.min_degree,
                'max_degree': self.filter_options.max_degree,
                'show_routers': self.filter_options.show_routers,
                'show_clients': self.filter_options.show_clients,
                'show_isolated': self.filter_options.show_isolated,
                'show_vulnerable': self.filter_options.show_vulnerable,
                'show_node_labels': self.filter_options.show_node_labels,
                'node_label_size': self.filter_options.node_label_size
            })
            
            data_str = f"{len(self.nodes)}_{len(self.edges)}_{filter_str}"
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.debug(f"Hash generation failed: {e}, using simple hash")
            # Fallback to a simple hash
            return f"{len(self.nodes)}_{len(self.edges)}_{id(self.filter_options)}"


# Keep the old class name for backward compatibility
InteractiveNetworkVisualizer = EnhancedNetworkVisualizer 

# JIT-compiled performance functions
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def compute_distances_fast(pos_array, indices):
        """Fast distance computation using numba JIT."""
        n = len(indices)
        distances = np.zeros((n, n))
        for i in prange(n):
            for j in range(i + 1, n):
                dx = pos_array[i, 0] - pos_array[j, 0]
                dy = pos_array[i, 1] - pos_array[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    @jit(nopython=True)
    def optimize_spring_layout_step(pos, adj_matrix, k, iterations):
        """JIT-optimized spring layout algorithm."""
        n = pos.shape[0]
        displacement = np.zeros_like(pos)
        
        for iteration in range(iterations):
            displacement.fill(0.0)
            
            # Repulsive forces
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            force = k * k / dist
                            displacement[i, 0] += (dx / dist) * force
                            displacement[i, 1] += (dy / dist) * force
            
            # Attractive forces
            for i in range(n):
                for j in range(n):
                    if adj_matrix[i, j] > 0:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            force = dist * dist / k
                            displacement[i, 0] -= (dx / dist) * force
                            displacement[i, 1] -= (dy / dist) * force
            
            # Apply displacement with cooling
            temp = k * (1.0 - iteration / iterations)
            for i in range(n):
                disp_len = np.sqrt(displacement[i, 0]**2 + displacement[i, 1]**2)
                if disp_len > 0:
                    factor = min(disp_len, temp) / disp_len
                    pos[i, 0] += displacement[i, 0] * factor
                    pos[i, 1] += displacement[i, 1] * factor
        
        return pos
else:
    # Fallback implementations without JIT
    def compute_distances_fast(pos_array, indices):
        """Fallback distance computation."""
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(pos_array))
    
    def optimize_spring_layout_step(pos, adj_matrix, k, iterations):
        """Fallback spring layout."""
        return pos


class PerformanceCache:
    """Thread-safe cache for expensive computations."""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()
        self._cache_times = {}
        self._max_age = 300  # 5 minutes
    
    def get(self, key):
        with self._lock:
            if key in self._cache:
                # Check if cache is still valid
                if time.time() - self._cache_times[key] < self._max_age:
                    return self._cache[key]
                else:
                    # Remove expired cache
                    del self._cache[key]
                    del self._cache_times[key]
            return None
    
    def set(self, key, value):
        with self._lock:
            self._cache[key] = value
            self._cache_times[key] = time.time()
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._cache_times.clear()


class LayoutCalculationThread(QThread):
    """Background thread for layout calculations."""
    
    layout_finished = pyqtSignal(dict)
    
    def __init__(self, graph, algorithm, options):
        super().__init__()
        self.graph = graph.copy()  # Work with a copy
        self.algorithm = algorithm
        self.options = options
        
    def run(self):
        """Calculate layout in background thread."""
        try:
            start_time = time.time()
            
            if self.algorithm == "spring_enhanced_fast" and NUMBA_AVAILABLE:
                pos = self._fast_spring_layout()
            else:
                pos = self._fallback_layout()
                
            calc_time = time.time() - start_time
            logger.info(f"Layout calculation completed in {calc_time:.2f}s using {self.algorithm}")
            
            self.layout_finished.emit(pos)
            
        except Exception as e:
            logger.error(f"Layout calculation failed: {e}")
            # Emit empty dict to signal failure
            self.layout_finished.emit({})
    
    def _fast_spring_layout(self):
        """Fast spring layout using JIT compilation."""
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=nodes).toarray()
        
        # Initialize positions randomly
        pos_array = np.random.random((n, 2)) * 10 - 5
        
        # Parameters
        k = self.options.cluster_separation / np.sqrt(n) if n > 0 else 1.0
        iterations = min(self.options.iterations, 50)  # Limit for performance
        
        # Optimize layout
        optimized_pos = optimize_spring_layout_step(pos_array, adj_matrix, k, iterations)
        
        # Convert back to dict
        return {nodes[i]: (optimized_pos[i, 0], optimized_pos[i, 1]) for i in range(n)}
    
    def _fallback_layout(self):
        """Fallback to NetworkX layouts."""
        try:
            if self.algorithm in ["spring_enhanced", "spring_enhanced_fast"]:
                k_value = self.options.cluster_separation / np.sqrt(len(self.graph.nodes()))
                return nx.spring_layout(
                    self.graph,
                    k=k_value,
                    iterations=min(self.options.iterations, 100),
                    seed=self.options.seed
                )
            else:
                return nx.spring_layout(self.graph, seed=42)
        except Exception as e:
            logger.error(f"Fallback layout failed: {e}")
            return {} 