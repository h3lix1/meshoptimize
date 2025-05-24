"""
Network Analyzer Module for Meshtastic Network Optimizer
Core algorithms for analyzing network topology and optimization.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class NodeScore:
    """Scoring data for a node."""
    node_id: str
    coverage_score: float = 0.0
    centrality_score: float = 0.0
    redundancy_score: float = 0.0
    hop_reduction_score: float = 0.0
    critical_path_score: float = 0.0
    total_score: float = 0.0
    
    def calculate_total(self, weights: Dict[str, float]):
        """Calculate total score based on weights."""
        self.total_score = (
            self.coverage_score * weights.get('coverage', 1.0) +
            self.centrality_score * weights.get('centrality', 1.0) +
            self.redundancy_score * weights.get('redundancy', 1.0) +
            self.hop_reduction_score * weights.get('hop_reduction', 1.0) +
            self.critical_path_score * weights.get('critical_path', 1.0)
        )


class NetworkAnalyzer:
    """Analyzes mesh network topology and provides optimization recommendations."""
    
    def __init__(self, nodes: Dict, edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self.graph = self._build_graph()
        self.node_scores = {}
        self._cache = {}
        
    def _build_graph(self) -> nx.Graph:
        """Build NetworkX graph from nodes and edges."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, node_data in self.nodes.items():
            G.add_node(node_id, **node_data)
            
        # Add edges
        for edge in self.edges:
            G.add_edge(edge['from'], edge['to'], 
                      weight=edge.get('weight', 1.0),
                      snr=edge.get('snr'))
                      
        return G
        
    def analyze_network_health(self) -> Dict:
        """Analyze overall network health metrics."""
        health = {
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'connected_components': nx.number_connected_components(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'average_degree': np.mean([d for _, d in self.graph.degree()]),
            'network_diameter': 0,
            'average_path_length': 0,
            'isolated_nodes': [],
            'vulnerable_nodes': [],  # Nodes with only one connection
            'router_count': sum(1 for n in self.nodes.values() if n['is_router']),
            'client_count': sum(1 for n in self.nodes.values() if not n['is_router'])
        }
        
        # Get largest connected component for diameter calculation
        if health['is_connected']:
            health['network_diameter'] = nx.diameter(self.graph)
            health['average_path_length'] = nx.average_shortest_path_length(self.graph)
        else:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            if len(subgraph) > 1:
                health['network_diameter'] = nx.diameter(subgraph)
                health['average_path_length'] = nx.average_shortest_path_length(subgraph)
                
        # Find isolated and vulnerable nodes
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree == 0:
                health['isolated_nodes'].append(node)
            elif degree == 1:
                health['vulnerable_nodes'].append(node)
                
        return health
        
    def calculate_node_scores(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, NodeScore]:
        """Calculate comprehensive scores for all nodes."""
        if weights is None:
            weights = {
                'coverage': 1.0,
                'centrality': 1.0,
                'redundancy': 1.0,
                'hop_reduction': 1.0,
                'critical_path': 1.0
            }
            
        # Use ThreadPoolExecutor for parallel computation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for node_id in self.graph.nodes():
                future = executor.submit(self._calculate_single_node_score, node_id, weights)
                futures.append((node_id, future))
                
            for node_id, future in futures:
                try:
                    self.node_scores[node_id] = future.result()
                except Exception as e:
                    logger.error(f"Error calculating score for node {node_id}: {e}")
                    self.node_scores[node_id] = NodeScore(node_id)
                    
        return self.node_scores
        
    def _calculate_single_node_score(self, node_id: str, weights: Dict[str, float]) -> NodeScore:
        """Calculate score for a single node."""
        score = NodeScore(node_id)
        
        # Coverage score - how many unique nodes this node can reach
        coverage = self._calculate_coverage_score(node_id)
        score.coverage_score = coverage / len(self.graph.nodes()) if self.graph.nodes() else 0
        
        # Centrality score - betweenness centrality
        if 'centrality' not in self._cache:
            self._cache['centrality'] = nx.betweenness_centrality(self.graph)
        score.centrality_score = self._cache['centrality'].get(node_id, 0)
        
        # Redundancy score - how much redundancy this node provides
        score.redundancy_score = self._calculate_redundancy_score(node_id)
        
        # Hop reduction score - impact on average hop count
        score.hop_reduction_score = self._calculate_hop_reduction_score(node_id)
        
        # Critical path score - is this node on critical paths?
        score.critical_path_score = self._calculate_critical_path_score(node_id)
        
        # Calculate total score
        score.calculate_total(weights)
        
        return score
        
    def _calculate_coverage_score(self, node_id: str) -> float:
        """Calculate how many nodes this node can reach."""
        if not self.graph.has_node(node_id):
            return 0.0
            
        # Get all reachable nodes
        reachable = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=5)
        return len(reachable)
        
    def _calculate_redundancy_score(self, node_id: str) -> float:
        """Calculate redundancy score based on alternative paths."""
        if not self.graph.has_node(node_id):
            return 0.0
            
        redundancy_sum = 0
        neighbors = list(self.graph.neighbors(node_id))
        
        # For each pair of neighbors, check if they're connected without this node
        temp_graph = self.graph.copy()
        temp_graph.remove_node(node_id)
        
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if temp_graph.has_node(neighbors[i]) and temp_graph.has_node(neighbors[j]):
                    if nx.has_path(temp_graph, neighbors[i], neighbors[j]):
                        redundancy_sum += 1
                        
        # Normalize by maximum possible redundancy
        max_redundancy = len(neighbors) * (len(neighbors) - 1) / 2
        return redundancy_sum / max_redundancy if max_redundancy > 0 else 0
        
    def _calculate_hop_reduction_score(self, node_id: str) -> float:
        """Calculate impact on average hop count if this node is a router."""
        if not self.graph.has_node(node_id):
            return 0.0
            
        # This is computationally expensive, so we sample
        sample_size = min(20, len(self.graph.nodes()) // 2)
        nodes_sample = list(self.graph.nodes())[:sample_size]
        
        total_reduction = 0
        for source in nodes_sample:
            if source == node_id:
                continue
            for target in nodes_sample:
                if target == node_id or target == source:
                    continue
                    
                try:
                    # Current shortest path
                    current_path = nx.shortest_path_length(self.graph, source, target)
                    
                    # Path through this node
                    path_through = (
                        nx.shortest_path_length(self.graph, source, node_id) +
                        nx.shortest_path_length(self.graph, node_id, target)
                    )
                    
                    if path_through < current_path:
                        total_reduction += current_path - path_through
                except nx.NetworkXNoPath:
                    pass
                    
        return total_reduction / (sample_size * sample_size) if sample_size > 0 else 0
        
    def _calculate_critical_path_score(self, node_id: str) -> float:
        """Calculate if this node is on critical paths."""
        if not self.graph.has_node(node_id):
            return 0.0
            
        # Check articulation points (nodes whose removal disconnects the graph)
        articulation_points = list(nx.articulation_points(self.graph))
        if node_id in articulation_points:
            return 1.0
            
        # Check how many shortest paths go through this node
        paths_through = 0
        total_paths = 0
        
        # Sample paths for efficiency
        sample_size = min(10, len(self.graph.nodes()) // 3)
        nodes_sample = list(self.graph.nodes())[:sample_size]
        
        for source in nodes_sample:
            if source == node_id:
                continue
            for target in nodes_sample:
                if target == node_id or target == source:
                    continue
                    
                try:
                    path = nx.shortest_path(self.graph, source, target)
                    total_paths += 1
                    if node_id in path:
                        paths_through += 1
                except nx.NetworkXNoPath:
                    pass
                    
        return paths_through / total_paths if total_paths > 0 else 0
        
    def find_redundant_routers(self, redundancy_threshold: int = 2) -> List[str]:
        """Find routers that can be safely demoted to clients."""
        redundant_routers = []
        
        # Only consider current routers
        routers = [n for n, data in self.nodes.items() if data['is_router']]
        
        for router in routers:
            # Check if removing this router maintains connectivity
            if self._can_safely_remove_router(router, redundancy_threshold):
                redundant_routers.append(router)
                
        return redundant_routers
        
    def _can_safely_remove_router(self, router_id: str, redundancy_threshold: int) -> bool:
        """Check if a router can be safely demoted."""
        # Create a test graph with this node as a client
        test_graph = self.graph.copy()
        
        # Check connectivity impact
        neighbors = list(self.graph.neighbors(router_id))
        
        # For each neighbor, ensure it has enough alternative connections
        for neighbor in neighbors:
            other_routers = 0
            for n in self.graph.neighbors(neighbor):
                if n != router_id and self.nodes[n]['is_router']:
                    other_routers += 1
                    
            if other_routers < redundancy_threshold:
                return False
                
        # Check if removal increases hop count significantly
        sample_nodes = list(self.graph.nodes())[:10]
        for source in sample_nodes:
            for target in sample_nodes:
                if source != target:
                    try:
                        current_hops = nx.shortest_path_length(self.graph, source, target)
                        # Simulate removal
                        test_graph.remove_node(router_id)
                        new_hops = nx.shortest_path_length(test_graph, source, target)
                        test_graph.add_node(router_id)
                        
                        if new_hops > current_hops + 2:  # Significant increase
                            return False
                    except nx.NetworkXNoPath:
                        return False
                        
        return True
        
    def find_promotion_candidates(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find clients that would benefit the network if promoted to routers."""
        candidates = []
        
        # Only consider current clients
        clients = [n for n, data in self.nodes.items() if not data['is_router']]
        
        for client in clients:
            score = self._calculate_promotion_benefit(client)
            candidates.append((client, score))
            
        # Sort by benefit score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]
        
    def _calculate_promotion_benefit(self, client_id: str) -> float:
        """Calculate the benefit of promoting a client to router."""
        # Factors to consider:
        # 1. Number of clients that would benefit
        # 2. Reduction in average hop count
        # 3. Improvement in network redundancy
        
        benefit = 0.0
        
        # Count neighboring clients
        client_neighbors = 0
        for neighbor in self.graph.neighbors(client_id):
            if not self.nodes[neighbor]['is_router']:
                client_neighbors += 1
                
        benefit += client_neighbors * 0.3
        
        # Check centrality
        if 'centrality' in self._cache:
            benefit += self._cache['centrality'].get(client_id, 0) * 0.4
            
        # Check if it connects disconnected components
        temp_nodes = {client_id: {'is_router': True}}
        temp_nodes.update(self.nodes)
        temp_analyzer = NetworkAnalyzer(temp_nodes, self.edges)
        new_health = temp_analyzer.analyze_network_health()
        current_health = self.analyze_network_health()
        
        if new_health['connected_components'] < current_health['connected_components']:
            benefit += 1.0
            
        return benefit
        
    def simulate_changes(self, demotions: List[str], promotions: List[str]) -> Dict:
        """Simulate the network with proposed changes."""
        # Create modified nodes
        modified_nodes = self.nodes.copy()
        
        for node_id in demotions:
            if node_id in modified_nodes:
                modified_nodes[node_id] = modified_nodes[node_id].copy()
                modified_nodes[node_id]['is_router'] = False
                
        for node_id in promotions:
            if node_id in modified_nodes:
                modified_nodes[node_id] = modified_nodes[node_id].copy()
                modified_nodes[node_id]['is_router'] = True
                
        # Create new analyzer with modified network
        new_analyzer = NetworkAnalyzer(modified_nodes, self.edges)
        
        # Compare metrics
        current_health = self.analyze_network_health()
        new_health = new_analyzer.analyze_network_health()
        
        comparison = {
            'current': current_health,
            'proposed': new_health,
            'changes': {
                'router_count_change': new_health['router_count'] - current_health['router_count'],
                'connectivity_maintained': new_health['is_connected'],
                'isolated_nodes_change': len(new_health['isolated_nodes']) - len(current_health['isolated_nodes']),
                'vulnerable_nodes_change': len(new_health['vulnerable_nodes']) - len(current_health['vulnerable_nodes']),
                'average_path_length_change': (new_health['average_path_length'] - current_health['average_path_length']) if (new_health['average_path_length'] is not None and new_health['average_path_length'] > 0 and current_health['average_path_length'] is not None) else 0
            }
        }
        
        return comparison
        
    def get_node_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get all nodes ranked by importance."""
        if not self.node_scores:
            self.calculate_node_scores()
            
        ranking = [(node_id, score.total_score) 
                   for node_id, score in self.node_scores.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking 