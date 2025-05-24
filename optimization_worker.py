"""
Optimization Worker Module for Meshtastic Network Optimizer
Handles background optimization tasks with threading.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from typing import Dict, List, Tuple, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from network_analyzer_optimized import OptimizedNetworkAnalyzer
import copy

logger = logging.getLogger(__name__)


class OptimizationWorker(QThread):
    """Background worker for network optimization tasks."""
    
    # Signals
    progress = pyqtSignal(int)  # Progress percentage
    status = pyqtSignal(str)  # Status message
    finished = pyqtSignal(dict)  # Results when complete
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, nodes: Dict, edges: List[Dict], params: Dict):
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self.params = params
        self.is_cancelled = False
        
    def run(self):
        """Run the optimization process."""
        try:
            self.status.emit("Starting network analysis...")
            self.progress.emit(10)
            
            # Create analyzer - use optimized version if available
            analyzer = OptimizedNetworkAnalyzer(self.nodes, self.edges, max_workers=4)
            
            # Analyze current network health
            self.status.emit("Analyzing network health...")
            current_health = analyzer.analyze_network_health()
            self.progress.emit(20)
            
            if self.is_cancelled:
                return
                
            # Calculate node scores
            self.status.emit("Calculating node scores...")
            weights = self.params.get('weights', {
                'coverage': 1.0,
                'centrality': 1.0,
                'redundancy': 1.0,
                'hop_reduction': 1.0,
                'critical_path': 1.0
            })
            node_scores = analyzer.calculate_node_scores(weights)
            self.progress.emit(40)
            
            if self.is_cancelled:
                return
                
            # Find redundant routers
            self.status.emit("Identifying redundant routers...")
            redundancy_threshold = self.params.get('redundancy_threshold', 2)
            redundant_routers = analyzer.find_redundant_routers(redundancy_threshold)
            self.progress.emit(60)
            
            if self.is_cancelled:
                return
                
            # Find promotion candidates
            self.status.emit("Finding promotion candidates...")
            promotion_candidates = analyzer.find_promotion_candidates(top_n=10)
            self.progress.emit(80)
            
            if self.is_cancelled:
                return
                
            # Generate optimization recommendations
            self.status.emit("Generating recommendations...")
            recommendations = self._generate_recommendations(
                analyzer, redundant_routers, promotion_candidates,
                current_health, node_scores
            )
            self.progress.emit(90)
            
            # Simulate proposed changes
            self.status.emit("Simulating proposed changes...")
            proposed_demotions = recommendations['demotions'][:self.params.get('max_demotions', 5)]
            proposed_promotions = [p[0] for p in recommendations['promotions'][:self.params.get('max_promotions', 3)]]
            
            simulation = analyzer.simulate_changes(proposed_demotions, proposed_promotions)
            
            # Compile results
            results = {
                'current_health': current_health,
                'node_scores': node_scores,
                'redundant_routers': redundant_routers,
                'promotion_candidates': promotion_candidates,
                'recommendations': recommendations,
                'simulation': simulation,
                'proposed_demotions': proposed_demotions,
                'proposed_promotions': proposed_promotions
            }
            
            self.progress.emit(100)
            self.status.emit("Optimization complete!")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            self.error.emit(str(e))
            
    def cancel(self):
        """Cancel the optimization process."""
        self.is_cancelled = True
        
    def _generate_recommendations(self, analyzer: OptimizedNetworkAnalyzer, 
                                redundant_routers: List[str],
                                promotion_candidates: List[Tuple[str, float]],
                                current_health: Dict,
                                node_scores: Dict) -> Dict:
        """Generate optimization recommendations."""
        recommendations = {
            'demotions': [],
            'promotions': [],
            'warnings': [],
            'improvements': []
        }
        
        # Sort redundant routers by their scores (demote lowest scoring first)
        router_scores = [(r, node_scores[r].total_score) for r in redundant_routers]
        router_scores.sort(key=lambda x: x[1])
        recommendations['demotions'] = [r[0] for r in router_scores]
        
        # Add top promotion candidates
        recommendations['promotions'] = promotion_candidates
        
        # Generate warnings
        if current_health['isolated_nodes']:
            recommendations['warnings'].append(
                f"Found {len(current_health['isolated_nodes'])} isolated nodes"
            )
            
        if current_health['vulnerable_nodes']:
            recommendations['warnings'].append(
                f"Found {len(current_health['vulnerable_nodes'])} vulnerable nodes with only one connection"
            )
            
        if not current_health['is_connected']:
            recommendations['warnings'].append(
                "Network is not fully connected!"
            )
            
        # Generate improvement suggestions
        if current_health['average_path_length'] > 3:
            recommendations['improvements'].append(
                "Consider adding routers to reduce average path length"
            )
            
        router_ratio = current_health['router_count'] / current_health['total_nodes']
        if router_ratio > 0.4:
            recommendations['improvements'].append(
                f"Router ratio is {router_ratio:.1%} - consider reducing routers"
            )
        elif router_ratio < 0.2:
            recommendations['improvements'].append(
                f"Router ratio is {router_ratio:.1%} - consider adding more routers"
            )
            
        return recommendations


class BatchOptimizationWorker(QThread):
    """Worker for batch optimization with multiple parameter sets."""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(list)  # List of results
    error = pyqtSignal(str)
    
    def __init__(self, nodes: Dict, edges: List[Dict], param_sets: List[Dict]):
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self.param_sets = param_sets
        self.is_cancelled = False
        
    def run(self):
        """Run batch optimization with multiple parameter sets."""
        try:
            results = []
            total_sets = len(self.param_sets)
            
            for i, params in enumerate(self.param_sets):
                if self.is_cancelled:
                    break
                    
                self.status.emit(f"Running optimization {i+1}/{total_sets}...")
                progress = int((i / total_sets) * 100)
                self.progress.emit(progress)
                
                # Run single optimization
                worker = OptimizationWorker(self.nodes, self.edges, params)
                worker.run()
                
                # Collect results
                # Note: In a real implementation, we'd connect to the worker's signals
                # For now, we'll simulate the results
                result = {
                    'params': params,
                    'timestamp': time.time()
                }
                results.append(result)
                
            self.progress.emit(100)
            self.status.emit("Batch optimization complete!")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Batch optimization error: {e}")
            self.error.emit(str(e))
            
    def cancel(self):
        """Cancel the batch optimization."""
        self.is_cancelled = True


class NetworkMonitor(QThread):
    """Monitor network changes and trigger re-optimization when needed."""
    
    network_changed = pyqtSignal(dict)  # Emitted when network changes significantly
    status = pyqtSignal(str)
    
    def __init__(self, data_fetcher, check_interval: int = 300):
        super().__init__()
        self.data_fetcher = data_fetcher
        self.check_interval = check_interval  # seconds
        self.is_running = True
        self.last_state = None
        
    def run(self):
        """Monitor network for changes."""
        while self.is_running:
            try:
                # Fetch current network state
                current_data = self.data_fetcher.fetch_network_data(use_cache=False)
                
                # Parse data
                nodes = self.data_fetcher.parse_node_data(current_data['nodes'])
                edges = self.data_fetcher.parse_edge_data(current_data['edges'], nodes)
                
                # Check for significant changes
                if self._has_significant_changes(nodes, edges):
                    self.status.emit("Network changes detected!")
                    self.network_changed.emit({
                        'nodes': nodes,
                        'edges': edges,
                        'timestamp': current_data.get('timestamp')
                    })
                    
                # Update last state
                self.last_state = {
                    'nodes': copy.deepcopy(nodes),
                    'edges': copy.deepcopy(edges)
                }
                
                # Wait for next check
                for _ in range(self.check_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                self.status.emit(f"Monitoring error: {e}")
                time.sleep(30)  # Wait before retry
                
    def stop(self):
        """Stop monitoring."""
        self.is_running = False
        
    def _has_significant_changes(self, nodes: Dict, edges: List[Dict]) -> bool:
        """Check if network has changed significantly."""
        if self.last_state is None:
            return True
            
        # Check node count changes
        if len(nodes) != len(self.last_state['nodes']):
            return True
            
        # Check edge count changes
        if len(edges) != len(self.last_state['edges']):
            return True
            
        # Check for role changes
        for node_id, node_data in nodes.items():
            if node_id in self.last_state['nodes']:
                old_role = self.last_state['nodes'][node_id].get('role')
                new_role = node_data.get('role')
                if old_role != new_role:
                    return True
                    
        return False 