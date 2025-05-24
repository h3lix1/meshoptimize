"""
Optimized Network Analyzer Module with Numba acceleration
For handling large networks efficiently
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
from functools import lru_cache
import warnings

# Try to import numba, fall back gracefully if not available
try:
    from numba import jit, prange, njit
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available, falling back to pure Python implementation")
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    prange = range

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


class OptimizedNetworkAnalyzer:
    """Optimized network analyzer with parallel processing and caching."""
    
    def __init__(self, nodes: Dict, edges: List[Dict], max_workers: int = 4):
        self.nodes = nodes
        self.edges = edges
        self.max_workers = max_workers
        self.graph = self._build_graph()
        self.node_scores = {}
        self._cache = {}
        
        # Precompute adjacency matrix for faster operations
        self._precompute_structures()
        
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
        
    def _precompute_structures(self):
        """Precompute data structures for faster access."""
        # Create node index mapping
        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.n_nodes = len(self.node_list)
        
        # Create adjacency matrix (sparse representation)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        
        # Precompute degrees
        self.degrees = np.array([self.graph.degree(node) for node in self.node_list])
        
    @lru_cache(maxsize=1024)
    def _get_node_distance(self, source_idx: int, target_idx: int) -> int:
        """Get shortest path distance between two nodes (cached)."""
        try:
            return nx.shortest_path_length(
                self.graph, 
                self.node_list[source_idx], 
                self.node_list[target_idx]
            )
        except nx.NetworkXNoPath:
            return float('inf')
            
    def analyze_network_health(self) -> Dict:
        """Analyze overall network health metrics with optimizations."""
        health = {
            'total_nodes': self.n_nodes,
            'total_edges': len(self.edges),
            'connected_components': nx.number_connected_components(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'average_degree': self.degrees.mean() if self.n_nodes > 0 else 0,
            'network_diameter': 0,
            'average_path_length': 0,
            'isolated_nodes': [],
            'vulnerable_nodes': [],
            'router_count': sum(1 for n in self.nodes.values() if n.get('is_router', False)),
            'client_count': sum(1 for n in self.nodes.values() if not n.get('is_router', False))
        }
        
        # Use vectorized operations for finding isolated and vulnerable nodes
        isolated_mask = self.degrees == 0
        vulnerable_mask = self.degrees == 1
        
        health['isolated_nodes'] = [self.node_list[i] for i in np.where(isolated_mask)[0]]
        health['vulnerable_nodes'] = [self.node_list[i] for i in np.where(vulnerable_mask)[0]]
        
        # Calculate diameter and average path length for connected components
        if health['is_connected']:
            health['network_diameter'] = nx.diameter(self.graph)
            health['average_path_length'] = nx.average_shortest_path_length(self.graph)
        else:
            # Handle disconnected graph
            largest_cc = max(nx.connected_components(self.graph), key=len)
            if len(largest_cc) > 1:
                subgraph = self.graph.subgraph(largest_cc)
                health['network_diameter'] = nx.diameter(subgraph)
                health['average_path_length'] = nx.average_shortest_path_length(subgraph)
                
        return health
        
    def calculate_node_scores(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, NodeScore]:
        """Calculate scores for all nodes using parallel processing."""
        if weights is None:
            weights = {
                'coverage': 1.0,
                'centrality': 1.0,
                'redundancy': 1.0,
                'hop_reduction': 1.0,
                'critical_path': 1.0
            }
            
        # Precompute centrality once for all nodes
        logger.info("Computing betweenness centrality...")
        self._cache['centrality'] = nx.betweenness_centrality(self.graph)
        
        # Find articulation points once
        logger.info("Finding articulation points...")
        self._cache['articulation_points'] = set(nx.articulation_points(self.graph))
        
        # Use process pool for CPU-intensive calculations
        logger.info(f"Calculating node scores using {self.max_workers} workers...")
        
        # Split nodes into chunks for parallel processing
        chunk_size = max(1, self.n_nodes // (self.max_workers * 4))
        node_chunks = [
            self.node_list[i:i+chunk_size] 
            for i in range(0, self.n_nodes, chunk_size)
        ]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for chunk in node_chunks:
                future = executor.submit(
                    self._calculate_chunk_scores, 
                    chunk, 
                    weights,
                    self._cache.get('centrality', {}),
                    self._cache.get('articulation_points', set())
                )
                futures.append(future)
                
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_scores = future.result()
                    self.node_scores.update(chunk_scores)
                except Exception as e:
                    logger.error(f"Error calculating scores: {e}")
                    
        return self.node_scores
        
    def _calculate_chunk_scores(self, node_chunk: List[str], weights: Dict[str, float],
                               centrality_cache: Dict, articulation_points: Set) -> Dict[str, NodeScore]:
        """Calculate scores for a chunk of nodes."""
        chunk_scores = {}
        
        for node_id in node_chunk:
            score = NodeScore(node_id)
            
            # Coverage score
            coverage = self._calculate_coverage_score_fast(node_id)
            score.coverage_score = coverage / self.n_nodes if self.n_nodes > 0 else 0
            
            # Centrality score (from cache)
            score.centrality_score = centrality_cache.get(node_id, 0)
            
            # Redundancy score
            score.redundancy_score = self._calculate_redundancy_score_fast(node_id)
            
            # Hop reduction score (sampled)
            score.hop_reduction_score = self._calculate_hop_reduction_score_sampled(node_id)
            
            # Critical path score
            if node_id in articulation_points:
                score.critical_path_score = 1.0
            else:
                score.critical_path_score = self._calculate_critical_path_score_fast(node_id)
                
            # Calculate total
            score.calculate_total(weights)
            chunk_scores[node_id] = score
            
        return chunk_scores
        
    def _calculate_coverage_score_fast(self, node_id: str) -> float:
        """Fast coverage calculation using BFS with early termination."""
        if node_id not in self.node_to_idx:
            return 0.0
            
        # Use BFS with max depth
        visited = set()
        queue = deque([(node_id, 0)])
        visited.add(node_id)
        
        while queue:
            current, depth = queue.popleft()
            if depth >= 5:  # Max hop count
                continue
                
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    
        return len(visited)
        
    def _calculate_redundancy_score_fast(self, node_id: str) -> float:
        """Fast redundancy calculation."""
        if node_id not in self.graph:
            return 0.0
            
        neighbors = list(self.graph.neighbors(node_id))
        if len(neighbors) < 2:
            return 0.0
            
        # Sample pairs for large neighbor sets
        if len(neighbors) > 20:
            import random
            sampled_neighbors = random.sample(neighbors, 20)
        else:
            sampled_neighbors = neighbors
            
        redundancy_count = 0
        total_pairs = 0
        
        # Check connectivity between neighbor pairs
        for i in range(len(sampled_neighbors)):
            for j in range(i + 1, len(sampled_neighbors)):
                total_pairs += 1
                # Check if neighbors are connected without going through node_id
                try:
                    path = nx.shortest_path(
                        self.graph, 
                        sampled_neighbors[i], 
                        sampled_neighbors[j]
                    )
                    if node_id not in path[1:-1]:  # Exclude source and target
                        redundancy_count += 1
                except nx.NetworkXNoPath:
                    pass
                    
        return redundancy_count / total_pairs if total_pairs > 0 else 0
        
    def _calculate_hop_reduction_score_sampled(self, node_id: str) -> float:
        """Calculate hop reduction with sampling for performance."""
        if node_id not in self.graph:
            return 0.0
            
        # Sample a subset of nodes for large networks
        sample_size = min(30, self.n_nodes // 10)
        if sample_size < 5:
            return 0.0
            
        import random
        sampled_nodes = random.sample(self.node_list, sample_size)
        
        improvement_sum = 0
        comparisons = 0
        
        for source in sampled_nodes:
            if source == node_id:
                continue
            for target in sampled_nodes:
                if target == node_id or target == source:
                    continue
                    
                try:
                    # Current shortest path
                    current_dist = nx.shortest_path_length(self.graph, source, target)
                    
                    # Distance through this node
                    dist_to_node = nx.shortest_path_length(self.graph, source, node_id)
                    dist_from_node = nx.shortest_path_length(self.graph, node_id, target)
                    
                    if dist_to_node + dist_from_node < current_dist:
                        improvement_sum += current_dist - (dist_to_node + dist_from_node)
                        
                    comparisons += 1
                except nx.NetworkXNoPath:
                    pass
                    
        return improvement_sum / comparisons if comparisons > 0 else 0
        
    def _calculate_critical_path_score_fast(self, node_id: str) -> float:
        """Fast critical path score using sampling."""
        if node_id not in self.graph:
            return 0.0
            
        # Sample paths
        sample_size = min(20, self.n_nodes // 5)
        import random
        sampled_nodes = random.sample(self.node_list, sample_size)
        
        paths_through = 0
        total_paths = 0
        
        for source in sampled_nodes[:sample_size//2]:
            if source == node_id:
                continue
            for target in sampled_nodes[sample_size//2:]:
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
        """Find routers that can be safely demoted."""
        redundant_routers = []
        routers = [n for n, data in self.nodes.items() if data.get('is_router', False)]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._can_safely_remove_router, router, redundancy_threshold): router
                for router in routers
            }
            
            for future in as_completed(futures):
                router = futures[future]
                try:
                    if future.result():
                        redundant_routers.append(router)
                except Exception as e:
                    logger.error(f"Error checking router {router}: {e}")
                    
        return redundant_routers
        
    def _can_safely_remove_router(self, router_id: str, redundancy_threshold: int) -> bool:
        """Check if a router can be safely demoted (optimized)."""
        neighbors = list(self.graph.neighbors(router_id))
        
        # Quick check: ensure each neighbor has enough router connections
        for neighbor in neighbors:
            router_neighbors = sum(
                1 for n in self.graph.neighbors(neighbor) 
                if n != router_id and self.nodes[n].get('is_router', False)
            )
            if router_neighbors < redundancy_threshold:
                return False
                
        # Sample-based connectivity check
        sample_size = min(10, len(self.node_list) // 10)
        import random
        sampled_nodes = random.sample(self.node_list, sample_size)
        
        # Create test graph without this router
        test_graph = self.graph.copy()
        test_graph.remove_node(router_id)
        
        for source in sampled_nodes[:sample_size//2]:
            for target in sampled_nodes[sample_size//2:]:
                if source != target:
                    try:
                        current_dist = nx.shortest_path_length(self.graph, source, target)
                        new_dist = nx.shortest_path_length(test_graph, source, target)
                        
                        if new_dist > current_dist + 2:  # Significant increase
                            return False
                    except nx.NetworkXNoPath:
                        # Path was broken
                        return False
                        
        return True
        
    def find_promotion_candidates(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find clients that would benefit the network if promoted (optimized)."""
        clients = [n for n, data in self.nodes.items() if not data.get('is_router', False)]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._calculate_promotion_benefit_fast, client): client
                for client in clients
            }
            
            candidates = []
            for future in as_completed(futures):
                client = futures[future]
                try:
                    benefit = future.result()
                    candidates.append((client, benefit))
                except Exception as e:
                    logger.error(f"Error calculating benefit for {client}: {e}")
                    
        # Sort and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]
        
    def _calculate_promotion_benefit_fast(self, client_id: str) -> float:
        """Fast promotion benefit calculation."""
        benefit = 0.0
        
        # Count neighboring clients
        client_neighbors = sum(
            1 for n in self.graph.neighbors(client_id) 
            if not self.nodes[n].get('is_router', False)
        )
        benefit += client_neighbors * 0.3
        
        # Use cached centrality
        if 'centrality' in self._cache:
            benefit += self._cache['centrality'].get(client_id, 0) * 0.4
            
        # Degree-based benefit
        degree = self.graph.degree(client_id)
        benefit += degree * 0.1
        
        # Check if it's a bridge node
        if client_id in self._cache.get('articulation_points', set()):
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
        new_analyzer = OptimizedNetworkAnalyzer(modified_nodes, self.edges)
        
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