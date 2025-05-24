#!/usr/bin/env python3
"""
Meshtastic Network Optimizer
Main GUI application for optimizing router placement in mesh networks.
"""

import sys
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QSplitter, QTabWidget, QProgressBar, QTextEdit, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QHeaderView, QCheckBox,
    QDoubleSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QAction, QFont
import pandas as pd
from datetime import datetime
import json

from data_fetcher import MeshDataFetcher, DataProcessor
from network_analyzer import NetworkAnalyzer
from network_analyzer_optimized import OptimizedNetworkAnalyzer
from visualization import NetworkVisualizer, PlotlyNetworkVisualizer
from interactive_visualization import InteractiveNetworkVisualizer
from optimization_worker import OptimizationWorker, NetworkMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeshOptimizerGUI(QMainWindow):
    """Main application window for Mesh Network Optimizer."""
    
    def __init__(self):
        super().__init__()
        self.data_fetcher = MeshDataFetcher()
        self.current_nodes = {}
        self.current_edges = []
        self.current_results = None
        self.optimization_worker = None
        self.network_monitor = None
        self.available_channels = set()
        self.selected_channel = "All Channels"
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Meshtastic Network Optimizer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Create main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Network visualization
        # Use interactive visualization for better performance with large networks
        self.network_viz = InteractiveNetworkVisualizer()
        self.network_viz.node_clicked.connect(self._on_node_clicked)
        splitter.addWidget(self.network_viz)
        
        # Right side - Tabs for different views
        self.tab_widget = QTabWidget()
        
        # Network Health Tab
        self.health_widget = self._create_health_widget()
        self.tab_widget.addTab(self.health_widget, "Network Health")
        
        # Node Scores Tab
        self.scores_table = self._create_scores_table()
        self.tab_widget.addTab(self.scores_table, "Node Scores")
        
        # Recommendations Tab
        self.recommendations_widget = self._create_recommendations_widget()
        self.tab_widget.addTab(self.recommendations_widget, "Recommendations")
        
        # Simulation Tab
        self.simulation_widget = self._create_simulation_widget()
        self.tab_widget.addTab(self.simulation_widget, "Simulation")
        
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([800, 600])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("Ready")
        
        # Load sample data on startup
        self._load_sample_data()
        
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Data from URL", self)
        load_action.triggered.connect(self._load_data_from_url)
        file_menu.addAction(load_action)
        
        load_file_action = QAction("Load Data from File", self)
        load_file_action.triggered.connect(self._load_data_from_file)
        file_menu.addAction(load_file_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        export_viz_action = QAction("Export Visualization", self)
        export_viz_action.triggered.connect(self._export_visualization)
        file_menu.addAction(export_viz_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        monitor_action = QAction("Start Network Monitor", self)
        monitor_action.setCheckable(True)
        monitor_action.triggered.connect(self._toggle_network_monitor)
        tools_menu.addAction(monitor_action)
        
        interactive_action = QAction("Open Interactive View", self)
        interactive_action.triggered.connect(self._open_interactive_view)
        tools_menu.addAction(interactive_action)
        
    def _create_control_panel(self):
        """Create the control panel with sliders and buttons."""
        panel = QGroupBox("Optimization Controls")
        layout = QHBoxLayout()
        
        # Channel filter
        channel_group = QGroupBox("Channel Filter")
        channel_layout = QVBoxLayout()
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("All Channels")
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        
        channel_layout.addWidget(QLabel("Select Channel:"))
        channel_layout.addWidget(self.channel_combo)
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Redundancy control
        redundancy_group = QGroupBox("Redundancy")
        redundancy_layout = QVBoxLayout()
        
        self.redundancy_slider = QSlider(Qt.Orientation.Horizontal)
        self.redundancy_slider.setMinimum(1)
        self.redundancy_slider.setMaximum(5)
        self.redundancy_slider.setValue(2)
        self.redundancy_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.redundancy_slider.setTickInterval(1)
        self.redundancy_slider.valueChanged.connect(self._on_redundancy_changed)
        
        self.redundancy_label = QLabel("Required Redundancy: 2")
        redundancy_layout.addWidget(self.redundancy_label)
        redundancy_layout.addWidget(self.redundancy_slider)
        redundancy_group.setLayout(redundancy_layout)
        layout.addWidget(redundancy_group)
        
        # Max hops control
        hops_group = QGroupBox("Max Hops")
        hops_layout = QVBoxLayout()
        
        self.hops_spin = QSpinBox()
        self.hops_spin.setMinimum(3)
        self.hops_spin.setMaximum(10)
        self.hops_spin.setValue(5)
        
        hops_layout.addWidget(QLabel("Maximum Hops:"))
        hops_layout.addWidget(self.hops_spin)
        hops_group.setLayout(hops_layout)
        layout.addWidget(hops_group)
        
        # Scoring weights
        weights_group = QGroupBox("Scoring Weights")
        weights_layout = QVBoxLayout()
        
        self.weight_sliders = {}
        weight_names = ['coverage', 'centrality', 'redundancy', 'hop_reduction', 'critical_path']
        
        for weight_name in weight_names:
            weight_layout = QHBoxLayout()
            label = QLabel(f"{weight_name.replace('_', ' ').title()}:")
            label.setFixedWidth(100)
            weight_layout.addWidget(label)
            
            slider = QDoubleSpinBox()
            slider.setMinimum(0.0)
            slider.setMaximum(2.0)
            slider.setSingleStep(0.1)
            slider.setValue(1.0)
            slider.valueChanged.connect(self._on_weight_changed)
            
            self.weight_sliders[weight_name] = slider
            weight_layout.addWidget(slider)
            weights_layout.addLayout(weight_layout)
            
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # Action buttons
        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self._load_data_from_url)
        buttons_layout.addWidget(self.load_button)
        
        self.optimize_button = QPushButton("Optimize Network")
        self.optimize_button.clicked.connect(self._run_optimization)
        self.optimize_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        buttons_layout.addWidget(self.optimize_button)
        
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self._apply_changes)
        self.apply_button.setEnabled(False)
        buttons_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self._reset_view)
        buttons_layout.addWidget(self.reset_button)
        
        buttons_group.setLayout(buttons_layout)
        layout.addWidget(buttons_group)
        
        panel.setLayout(layout)
        return panel
        
    def _create_health_widget(self):
        """Create widget for displaying network health metrics."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.health_text = QTextEdit()
        self.health_text.setReadOnly(True)
        # Use a monospace font that's available on all platforms
        font = QFont()
        font.setFamily("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.health_text.setFont(font)
        
        layout.addWidget(self.health_text)
        widget.setLayout(layout)
        return widget
        
    def _create_scores_table(self):
        """Create table for displaying node scores."""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Node ID", "Label", "Role", "Coverage", "Centrality", 
            "Redundancy", "Total Score"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSortingEnabled(True)
        table.itemSelectionChanged.connect(self._on_table_selection_changed)
        return table
        
    def _create_recommendations_widget(self):
        """Create widget for displaying optimization recommendations."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        
        layout.addWidget(self.recommendations_text)
        widget.setLayout(layout)
        return widget
        
    def _create_simulation_widget(self):
        """Create widget for displaying simulation results."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.simulation_text = QTextEdit()
        self.simulation_text.setReadOnly(True)
        
        layout.addWidget(self.simulation_text)
        widget.setLayout(layout)
        return widget
        
    @pyqtSlot()
    def _on_redundancy_changed(self):
        """Handle redundancy slider change."""
        value = self.redundancy_slider.value()
        self.redundancy_label.setText(f"Required Redundancy: {value}")
        
    @pyqtSlot()
    def _on_weight_changed(self):
        """Handle weight slider change."""
        # Could trigger live updates if desired
        pass
        
    @pyqtSlot(str)
    def _on_channel_changed(self, channel):
        """Handle channel filter change."""
        self.selected_channel = channel
        self._apply_channel_filter()
        
    @pyqtSlot()
    def _load_sample_data(self):
        """Load sample data for testing."""
        try:
            self.statusBar().showMessage("Loading sample data...")
            
            # Generate sample data
            data = self.data_fetcher.get_sample_data()
            
            # Parse data
            self.current_nodes = self.data_fetcher.parse_node_data(data['nodes'])
            self.current_edges = self.data_fetcher.parse_edge_data(data['edges'], self.current_nodes)
            
            # Update available channels
            self._update_available_channels()
            
            # Update visualization
            self._apply_channel_filter()
            
            # Update health metrics
            self._update_health_display()
            
            self.statusBar().showMessage("Sample data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load sample data: {e}")
            
    @pyqtSlot()
    def _load_data_from_url(self):
        """Load data from the Meshtastic URL."""
        try:
            self.statusBar().showMessage("Fetching data from meshview.bayme.sh...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Fetch data
            data = self.data_fetcher.fetch_network_data(use_cache=False)
            
            # Validate data
            if not DataProcessor.validate_network_data(data):
                raise ValueError("Invalid network data format")
                
            # Clean data
            data = DataProcessor.clean_network_data(data)
            
            # Parse data
            self.current_nodes = self.data_fetcher.parse_node_data(data['nodes'])
            self.current_edges = self.data_fetcher.parse_edge_data(data['edges'], self.current_nodes)
            
            # Update available channels
            self._update_available_channels()
            
            # Update visualization
            self._apply_channel_filter()
            
            # Update health metrics
            self._update_health_display()
            
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(
                f"Loaded {len(self.current_nodes)} nodes and {len(self.current_edges)} edges"
            )
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            self.statusBar().showMessage("Failed to load data")
            
    @pyqtSlot()
    def _load_data_from_file(self):
        """Load data from a JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Network Data", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    
                # Validate and process data
                if not DataProcessor.validate_network_data(data):
                    raise ValueError("Invalid network data format")
                    
                data = DataProcessor.clean_network_data(data)
                
                # Parse data
                self.current_nodes = self.data_fetcher.parse_node_data(data['nodes'])
                self.current_edges = self.data_fetcher.parse_edge_data(data['edges'], self.current_nodes)
                
                # Update available channels
                self._update_available_channels()
                
                # Update visualization
                self._apply_channel_filter()
                
                # Update health metrics
                self._update_health_display()
                
                self.statusBar().showMessage(f"Loaded data from {filename}")
                
            except Exception as e:
                logger.error(f"Error loading file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
                
    def _update_available_channels(self):
        """Update the list of available channels from the current nodes."""
        # Clear existing items except "All Channels"
        self.channel_combo.blockSignals(True)
        while self.channel_combo.count() > 1:
            self.channel_combo.removeItem(1)
        
        # Collect unique channels
        self.available_channels = set()
        for node in self.current_nodes.values():
            channel = node.get('channel', '')
            if channel:
                self.available_channels.add(channel)
        
        # Add channels to combo box
        for channel in sorted(self.available_channels):
            self.channel_combo.addItem(channel)
        
        self.channel_combo.blockSignals(False)
        
    def _apply_channel_filter(self):
        """Apply the channel filter to the visualization."""
        if self.selected_channel == "All Channels":
            # Show all nodes
            filtered_nodes = self.current_nodes
            filtered_edges = self.current_edges
        else:
            # Filter nodes by channel
            filtered_nodes = {
                node_id: node_data 
                for node_id, node_data in self.current_nodes.items()
                if node_data.get('channel', '') == self.selected_channel
            }
            
            # Filter edges to only include those between filtered nodes
            filtered_node_ids = set(filtered_nodes.keys())
            filtered_edges = [
                edge for edge in self.current_edges
                if edge['from'] in filtered_node_ids and edge['to'] in filtered_node_ids
            ]
        
        # Update visualization
        self.network_viz.update_network(filtered_nodes, filtered_edges)
        
        # Update status
        self.statusBar().showMessage(
            f"Showing {len(filtered_nodes)} nodes and {len(filtered_edges)} edges"
            f" (Channel: {self.selected_channel})"
        )
    
    def _update_health_display(self):
        """Update the network health display."""
        if not self.current_nodes:
            return
            
        # Get filtered data for analysis
        if self.selected_channel == "All Channels":
            nodes_for_analysis = self.current_nodes
            edges_for_analysis = self.current_edges
        else:
            # Filter nodes by channel
            nodes_for_analysis = {
                node_id: node_data 
                for node_id, node_data in self.current_nodes.items()
                if node_data.get('channel', '') == self.selected_channel
            }
            
            # Filter edges to only include those between filtered nodes
            filtered_node_ids = set(nodes_for_analysis.keys())
            edges_for_analysis = [
                edge for edge in self.current_edges
                if edge['from'] in filtered_node_ids and edge['to'] in filtered_node_ids
            ]
            
        # Use optimized analyzer for better performance
        analyzer = OptimizedNetworkAnalyzer(nodes_for_analysis, edges_for_analysis)
        health = analyzer.analyze_network_health()
        
        # Format health information
        health_text = f"""
Network Health Report
====================
Total Nodes: {health['total_nodes']}
Total Edges: {health['total_edges']}
Connected Components: {health['connected_components']}
Network Connected: {'Yes' if health['is_connected'] else 'No'}

Router Count: {health['router_count']} ({health['router_count']/health['total_nodes']*100 if health['total_nodes'] > 0 else 0:.1f}%)
Client Count: {health['client_count']} ({health['client_count']/health['total_nodes']*100 if health['total_nodes'] > 0 else 0:.1f}%)

Average Degree: {health['average_degree']:.2f}
Network Diameter: {health['network_diameter']}
Average Path Length: {health['average_path_length']:.2f}

Isolated Nodes: {len(health['isolated_nodes'])}
Vulnerable Nodes (1 connection): {len(health['vulnerable_nodes'])}
"""
        
        if health['isolated_nodes']:
            health_text += f"\nIsolated: {', '.join(health['isolated_nodes'][:5])}"
            if len(health['isolated_nodes']) > 5:
                health_text += f" and {len(health['isolated_nodes']) - 5} more..."
                
        self.health_text.setText(health_text)
        
    @pyqtSlot()
    def _run_optimization(self):
        """Run the network optimization."""
        if not self.current_nodes:
            QMessageBox.warning(self, "Warning", "Please load network data first")
            return
            
        # Get filtered data for optimization
        if self.selected_channel == "All Channels":
            nodes_for_optimization = self.current_nodes
            edges_for_optimization = self.current_edges
        else:
            # Filter nodes by channel
            nodes_for_optimization = {
                node_id: node_data 
                for node_id, node_data in self.current_nodes.items()
                if node_data.get('channel', '') == self.selected_channel
            }
            
            # Filter edges to only include those between filtered nodes
            filtered_node_ids = set(nodes_for_optimization.keys())
            edges_for_optimization = [
                edge for edge in self.current_edges
                if edge['from'] in filtered_node_ids and edge['to'] in filtered_node_ids
            ]
            
        # Prepare parameters
        params = {
            'redundancy_threshold': self.redundancy_slider.value(),
            'max_hops': self.hops_spin.value(),
            'weights': {
                name: slider.value() 
                for name, slider in self.weight_sliders.items()
            },
            'max_demotions': 10,
            'max_promotions': 5
        }
        
        # Create and start worker
        self.optimization_worker = OptimizationWorker(
            nodes_for_optimization, edges_for_optimization, params
        )
        
        # Connect signals
        self.optimization_worker.progress.connect(self._on_optimization_progress)
        self.optimization_worker.status.connect(self._on_optimization_status)
        self.optimization_worker.finished.connect(self._on_optimization_finished)
        self.optimization_worker.error.connect(self._on_optimization_error)
        
        # Disable controls
        self.optimize_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # Start optimization
        self.optimization_worker.start()
        
    @pyqtSlot(int)
    def _on_optimization_progress(self, value):
        """Handle optimization progress updates."""
        self.progress_bar.setValue(value)
        
    @pyqtSlot(str)
    def _on_optimization_status(self, message):
        """Handle optimization status updates."""
        self.statusBar().showMessage(message)
        
    @pyqtSlot(dict)
    def _on_optimization_finished(self, results):
        """Handle optimization completion."""
        self.current_results = results
        
        # Update scores table
        self._update_scores_table(results['node_scores'])
        
        # Update recommendations
        self._update_recommendations(results['recommendations'])
        
        # Update simulation
        self._update_simulation(results['simulation'])
        
        # Highlight proposed changes - need to apply filter
        highlight_nodes = set(results['proposed_demotions'] + results['proposed_promotions'])
        
        # Get filtered data for visualization update
        if self.selected_channel == "All Channels":
            nodes_for_viz = self.current_nodes
            edges_for_viz = self.current_edges
        else:
            # Filter nodes by channel
            nodes_for_viz = {
                node_id: node_data 
                for node_id, node_data in self.current_nodes.items()
                if node_data.get('channel', '') == self.selected_channel
            }
            
            # Filter edges to only include those between filtered nodes
            filtered_node_ids = set(nodes_for_viz.keys())
            edges_for_viz = [
                edge for edge in self.current_edges
                if edge['from'] in filtered_node_ids and edge['to'] in filtered_node_ids
            ]
        
        self.network_viz.update_network(
            nodes_for_viz, edges_for_viz,
            highlight_nodes=highlight_nodes
        )
        
        # Enable controls
        self.optimize_button.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage("Optimization complete!")
        
    @pyqtSlot(str)
    def _on_optimization_error(self, error_msg):
        """Handle optimization errors."""
        logger.error(f"Optimization error: {error_msg}")
        QMessageBox.critical(self, "Optimization Error", error_msg)
        
        self.optimize_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Optimization failed")
        
    def _update_scores_table(self, node_scores):
        """Update the node scores table."""
        self.scores_table.setRowCount(len(node_scores))
        
        for row, (node_id, score) in enumerate(node_scores.items()):
            node_data = self.current_nodes.get(node_id, {})
            
            self.scores_table.setItem(row, 0, QTableWidgetItem(node_id))
            self.scores_table.setItem(row, 1, QTableWidgetItem(node_data.get('label', '')))
            self.scores_table.setItem(row, 2, QTableWidgetItem(node_data.get('role', '')))
            self.scores_table.setItem(row, 3, QTableWidgetItem(f"{score.coverage_score:.3f}"))
            self.scores_table.setItem(row, 4, QTableWidgetItem(f"{score.centrality_score:.3f}"))
            self.scores_table.setItem(row, 5, QTableWidgetItem(f"{score.redundancy_score:.3f}"))
            self.scores_table.setItem(row, 6, QTableWidgetItem(f"{score.total_score:.3f}"))
            
    def _update_recommendations(self, recommendations):
        """Update the recommendations display."""
        text = "Optimization Recommendations\n" + "=" * 30 + "\n\n"
        
        # Demotions
        text += f"Routers to Demote ({len(recommendations['demotions'])}):\n"
        for node_id in recommendations['demotions'][:5]:
            node = self.current_nodes.get(node_id, {})
            text += f"  • {node.get('label', node_id)}\n"
        if len(recommendations['demotions']) > 5:
            text += f"  ... and {len(recommendations['demotions']) - 5} more\n"
            
        text += "\n"
        
        # Promotions
        text += f"Clients to Promote ({len(recommendations['promotions'])}):\n"
        for node_id, benefit in recommendations['promotions'][:5]:
            node = self.current_nodes.get(node_id, {})
            text += f"  • {node.get('label', node_id)} (benefit: {benefit:.2f})\n"
            
        text += "\n"
        
        # Warnings
        if recommendations['warnings']:
            text += "Warnings:\n"
            for warning in recommendations['warnings']:
                text += f"  ⚠ {warning}\n"
            text += "\n"
            
        # Improvements
        if recommendations['improvements']:
            text += "Suggested Improvements:\n"
            for improvement in recommendations['improvements']:
                text += f"  → {improvement}\n"
                
        self.recommendations_text.setText(text)
        
    def _update_simulation(self, simulation):
        """Update the simulation results display."""
        text = "Simulation Results\n" + "=" * 30 + "\n\n"
        
        current = simulation['current']
        proposed = simulation['proposed']
        changes = simulation['changes']
        
        text += "Current Network:\n"
        text += f"  • Routers: {current['router_count']}\n"
        text += f"  • Average Path Length: {current['average_path_length']:.2f}\n"
        text += f"  • Isolated Nodes: {len(current['isolated_nodes'])}\n"
        text += f"  • Vulnerable Nodes: {len(current['vulnerable_nodes'])}\n"
        
        text += "\nProposed Network:\n"
        text += f"  • Routers: {proposed['router_count']} ({changes['router_count_change']:+d})\n"
        text += f"  • Average Path Length: {proposed['average_path_length']:.2f} ({changes['average_path_length_change']:+.2f})\n"
        text += f"  • Isolated Nodes: {len(proposed['isolated_nodes'])} ({changes['isolated_nodes_change']:+d})\n"
        text += f"  • Vulnerable Nodes: {len(proposed['vulnerable_nodes'])} ({changes['vulnerable_nodes_change']:+d})\n"
        
        text += "\nSummary:\n"
        if changes['connectivity_maintained']:
            text += "  ✓ Network connectivity maintained\n"
        else:
            text += "  ✗ WARNING: Network connectivity compromised!\n"
            
        if changes['router_count_change'] < 0:
            text += f"  ✓ Router count reduced by {-changes['router_count_change']}\n"
            
        if changes['average_path_length_change'] <= 0:
            text += "  ✓ Average path length maintained or improved\n"
        else:
            text += f"  ⚠ Average path length increased by {changes['average_path_length_change']:.2f}\n"
            
        self.simulation_text.setText(text)
        
    @pyqtSlot()
    def _apply_changes(self):
        """Apply the proposed changes to the network."""
        if not self.current_results:
            return
            
        reply = QMessageBox.question(
            self, "Apply Changes",
            "Are you sure you want to apply the proposed changes?\n"
            "This will update the node roles in the visualization.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Apply changes to nodes
            for node_id in self.current_results['proposed_demotions']:
                if node_id in self.current_nodes:
                    self.current_nodes[node_id]['is_router'] = False
                    self.current_nodes[node_id]['role'] = 'CLIENT'
                    
            for node_id in self.current_results['proposed_promotions']:
                if node_id in self.current_nodes:
                    self.current_nodes[node_id]['is_router'] = True
                    self.current_nodes[node_id]['role'] = 'ROUTER'
                    
            # Update visualization
            self.network_viz.update_network(self.current_nodes, self.current_edges)
            
            # Update health display
            self._update_health_display()
            
            # Clear results
            self.current_results = None
            self.apply_button.setEnabled(False)
            
            self.statusBar().showMessage("Changes applied successfully")
            
    @pyqtSlot()
    def _reset_view(self):
        """Reset the visualization view."""
        self.network_viz.clear_highlights()
        # Reset channel filter
        self.channel_combo.setCurrentIndex(0)  # Set to "All Channels"
        self.selected_channel = "All Channels"
        self._apply_channel_filter()
        self.statusBar().showMessage("View reset")
        
    @pyqtSlot(str)
    def _on_node_clicked(self, node_id):
        """Handle node click events."""
        if node_id in self.current_nodes:
            node = self.current_nodes[node_id]
            from visualization import sanitize_label
            label = sanitize_label(node.get('label', node_id))
            self.statusBar().showMessage(
                f"Selected: {label} - "
                f"Role: {node.get('role', 'Unknown')} - "
                f"Connections: {len(node.get('neighbors', []))}"
            )
            
    @pyqtSlot()
    def _on_table_selection_changed(self):
        """Handle table selection changes."""
        selected_items = self.scores_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            node_id = self.scores_table.item(row, 0).text()
            
            # Highlight node in visualization
            self.network_viz.highlight_nodes = {node_id}
            self.network_viz.update_network(self.current_nodes, self.current_edges, 
                                          highlight_nodes={node_id})
            
    @pyqtSlot()
    def _export_results(self):
        """Export optimization results to file."""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No optimization results to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                # Prepare export data
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'parameters': self.current_results.get('params', {}),
                    'proposed_demotions': self.current_results['proposed_demotions'],
                    'proposed_promotions': self.current_results['proposed_promotions'],
                    'network_health': self.current_results['current_health'],
                    'simulation': self.current_results['simulation']
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                self.statusBar().showMessage(f"Results exported to {filename}")
                
            except Exception as e:
                logger.error(f"Error exporting results: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export results: {e}")
                
    @pyqtSlot()
    def _export_visualization(self):
        """Export the current visualization."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        
        if filename:
            try:
                self.network_viz.export_to_file(filename)
                self.statusBar().showMessage(f"Visualization exported to {filename}")
            except Exception as e:
                logger.error(f"Error exporting visualization: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export visualization: {e}")
                
    @pyqtSlot(bool)
    def _toggle_network_monitor(self, checked):
        """Toggle network monitoring."""
        if checked:
            # Start monitoring
            self.network_monitor = NetworkMonitor(self.data_fetcher, check_interval=60)
            self.network_monitor.network_changed.connect(self._on_network_changed)
            self.network_monitor.status.connect(lambda msg: self.statusBar().showMessage(msg))
            self.network_monitor.start()
            self.statusBar().showMessage("Network monitoring started")
        else:
            # Stop monitoring
            if self.network_monitor:
                self.network_monitor.stop()
                self.network_monitor = None
            self.statusBar().showMessage("Network monitoring stopped")
            
    @pyqtSlot(dict)
    def _on_network_changed(self, data):
        """Handle network change notifications."""
        reply = QMessageBox.question(
            self, "Network Changed",
            "The network topology has changed. Reload data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.current_nodes = data['nodes']
            self.current_edges = data['edges']
            self.network_viz.update_network(self.current_nodes, self.current_edges)
            self._update_health_display()
            
    @pyqtSlot()
    def _open_interactive_view(self):
        """Open an interactive Plotly visualization."""
        if not self.current_nodes:
            QMessageBox.warning(self, "Warning", "Please load network data first")
            return
            
        try:
            # Create interactive plot
            fig = PlotlyNetworkVisualizer.create_interactive_plot(
                self.current_nodes, self.current_edges,
                node_scores=self.current_results.get('node_scores') if self.current_results else None
            )
            
            # Save to file
            filename = "network_interactive.html"
            PlotlyNetworkVisualizer.save_interactive_plot(fig, filename)
            
            self.statusBar().showMessage(f"Interactive view saved to {filename}")
            
            # Try to open in browser
            import webbrowser
            webbrowser.open(filename)
            
        except Exception as e:
            logger.error(f"Error creating interactive view: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create interactive view: {e}")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Meshtastic Network Optimizer")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MeshOptimizerGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 