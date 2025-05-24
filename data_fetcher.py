"""
Data Fetcher Module for Meshtastic Network Optimizer
Handles fetching and parsing network data from meshview.bayme.sh
PERFORMANCE OPTIMIZED with caching, threading, and faster parsing
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)


class MeshDataFetcher:
    """Fetches and parses Meshtastic network data from various sources - PERFORMANCE OPTIMIZED."""
    
    def __init__(self, url: str = "https://meshview.bayme.sh/nodegraph"):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; MeshOptimizer/1.0)'
        })
        self._cache = {}
        self._cache_time = None
        self.cache_duration = 300  # 5 minutes cache
        self._cache_lock = threading.RLock()
        
        # Pre-compile regexes for performance
        self._node_regex = re.compile(r'const\s+nodes\s*=\s*(\[[\s\S]*?\]);')
        self._edge_regex = re.compile(r'const\s+edges\s*=\s*(\[[\s\S]*?\]);')
        self._node_block_regex = re.compile(r'(?=\s*\{\s*name:)')
        self._field_regexes = {
            'name': re.compile(r'name:\s*`([^`]+)`'),
            'value': re.compile(r'value:\s*`([^`]+)`'),
            'role': re.compile(r'role:\s*`([^`]+)`'),
            'hw_model': re.compile(r'hw_model:\s*`([^`]+)`'),
            'short_name': re.compile(r'short_name:\s*`([^`]+)`'),
            'long_name': re.compile(r'long_name:\s*`([^`]+)`'),
            'channel': re.compile(r'channel:\s*`([^`]+)`')
        }
        
    def fetch_network_data(self, use_cache: bool = True) -> Dict:
        """
        Fetch network data from meshview.bayme.sh with performance optimizations.
        Returns a dictionary with nodes and edges data.
        """
        # Check cache with thread safety
        with self._cache_lock:
            if use_cache and self._cache_time:
                if (datetime.now() - self._cache_time).seconds < self.cache_duration:
                    logger.info("Using cached data")
                    return self._cache.copy()  # Return copy to avoid modification
                    
        try:
            logger.info(f"Fetching data from {self.url}")
            start_time = time.time()
            
            response = self.session.get(self.url, timeout=30)
            response.raise_for_status()
            
            fetch_time = time.time() - start_time
            logger.info(f"Data fetched in {fetch_time:.2f}s")
            
            # Parse HTML with performance optimizations
            parse_start = time.time()
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract JavaScript data more efficiently
            script_content = self._extract_script_content(soup)
            
            # Parse nodes and edges in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                nodes_future = executor.submit(self._parse_nodes_fast, script_content)
                edges_future = executor.submit(self._parse_edges_fast, script_content)
                
                nodes_data = nodes_future.result()
                edges_data = edges_future.result()
            
            parse_time = time.time() - parse_start
            logger.info(f"Data parsed in {parse_time:.2f}s")
            
            # If we still don't have data, use sample data
            if not nodes_data or not edges_data:
                logger.warning("Could not parse real data, using sample data")
                return self.get_sample_data()
            
            result = {
                'nodes': nodes_data,
                'edges': edges_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache with thread safety
            with self._cache_lock:
                self._cache = result.copy()
                self._cache_time = datetime.now()
            
            total_time = time.time() - start_time
            logger.info(f"Total fetch and parse time: {total_time:.2f}s")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {e}")
            # Return cached data if available
            with self._cache_lock:
                if self._cache:
                    logger.warning("Using stale cache due to fetch error")
                    return self._cache.copy()
            # Otherwise return sample data
            logger.warning("No cache available, using sample data")
            return self.get_sample_data()
    
    def _extract_script_content(self, soup) -> str:
        """Extract relevant script content more efficiently."""
        script_tags = soup.find_all('script')
        
        # Find the script containing both nodes and edges
        for script in script_tags:
            if script.string and 'const nodes = [' in script.string and 'const edges = [' in script.string:
                return script.string
        
        # Fallback: concatenate all scripts
        return '\n'.join(script.string or '' for script in script_tags)
    
    def _parse_nodes_fast(self, script_content: str) -> List[Dict]:
        """Fast node parsing with optimized regex and parallel processing."""
        nodes_match = self._node_regex.search(script_content)
        if not nodes_match:
            return []
            
        nodes_str = nodes_match.group(1)
        
        # Split into node blocks more efficiently
        node_blocks = self._node_block_regex.split(nodes_str)
        
        # Process blocks in parallel batches
        nodes = []
        batch_size = 50
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, len(node_blocks), batch_size):
                batch = node_blocks[i:i + batch_size]
                future = executor.submit(self._process_node_batch, batch)
                futures.append(future)
            
            for future in as_completed(futures):
                nodes.extend(future.result())
        
        logger.info(f"Parsed {len(nodes)} nodes from raw data")
        return nodes
    
    def _process_node_batch(self, node_blocks: List[str]) -> List[Dict]:
        """Process a batch of node blocks."""
        batch_nodes = []
        
        for block in node_blocks:
            if 'name:' not in block:
                continue
                
            try:
                node_data = self._extract_node_fields(block)
                if node_data:
                    batch_nodes.append(node_data)
            except Exception as e:
                logger.debug(f"Failed to parse node: {e}")
                continue
                
        return batch_nodes
    
    @lru_cache(maxsize=1000)
    def _extract_node_fields(self, block: str) -> Optional[Dict]:
        """Extract node fields using cached regex matches."""
        node_data = {}
        
        # Extract name (node ID) and convert to hex format
        name_match = self._field_regexes['name'].search(block)
        if not name_match:
            return None
            
        name_str = name_match.group(1)
        # Optimized hex conversion
        node_data.update(self._convert_to_hex_fast(name_str))
        
        # Extract other fields efficiently
        for field, regex in self._field_regexes.items():
            if field == 'name':
                continue
                
            match = regex.search(block)
            if match:
                value = match.group(1)
                if field == 'value':
                    node_data['label'] = self._clean_label_fast(value)
                elif field == 'long_name':
                    node_data['long_name'] = value.replace('&lt;', '<').replace('&gt;', '>')
                else:
                    node_data[field] = value
        
        # Set default role if not found
        if 'role' not in node_data:
            node_data['role'] = 'CLIENT'
            
        return node_data
    
    @lru_cache(maxsize=10000)
    def _convert_to_hex_fast(self, name_str: str) -> Dict:
        """Fast hex conversion with caching."""
        try:
            if name_str.startswith('0x') or name_str.startswith('!'):
                return {'id': name_str, 'hex_name': name_str}
            else:
                name_int = int(name_str)
                hex_name = f"!{name_int:08x}"
                return {
                    'id': hex_name,
                    'hex_name': hex_name,
                    'original_name': name_str
                }
        except ValueError:
            return {'id': name_str, 'hex_name': name_str}
    
    @lru_cache(maxsize=1000)
    def _clean_label_fast(self, label: str) -> str:
        """Fast label cleaning with caching."""
        label = label.strip('"')
        try:
            return label.encode('utf-8').decode('unicode_escape')
        except:
            return label
    
    def _parse_edges_fast(self, script_content: str) -> List[Dict]:
        """Fast edge parsing with optimized regex."""
        edges_match = self._edge_regex.search(script_content)
        if not edges_match:
            return []
            
        edges_str = edges_match.group(1)
        
        # Use more efficient regex for edge objects
        edge_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
        edge_matches = edge_pattern.finditer(edges_str)
        
        edges = []
        edge_regexes = {
            'source': re.compile(r'source:\s*`([^`]+)`'),
            'target': re.compile(r'target:\s*`([^`]+)`'),
            'originalColor': re.compile(r'originalColor:\s*`([^`]+)`'),
            'width': re.compile(r'width:\s*(\d+(?:\.\d+)?)')
        }
        
        for match in edge_matches:
            edge_str = match.group(0)
            
            try:
                edge_data = {}
                
                # Extract source and target with hex conversion
                source_match = edge_regexes['source'].search(edge_str)
                target_match = edge_regexes['target'].search(edge_str)
                
                if source_match and target_match:
                    edge_data['from'] = self._convert_edge_id_fast(source_match.group(1))
                    edge_data['to'] = self._convert_edge_id_fast(target_match.group(1))
                    
                    # Extract other fields
                    color_match = edge_regexes['originalColor'].search(edge_str)
                    if color_match:
                        edge_data['originalColor'] = color_match.group(1)
                    
                    width_match = edge_regexes['width'].search(edge_str)
                    if width_match:
                        edge_data['weight'] = float(width_match.group(1)) / 2.0
                    else:
                        edge_data['weight'] = 1.0
                    
                    edges.append(edge_data)
                    
            except Exception as e:
                logger.debug(f"Failed to parse edge: {e}")
                continue
        
        return edges
    
    @lru_cache(maxsize=10000)
    def _convert_edge_id_fast(self, id_str: str) -> str:
        """Fast edge ID conversion with caching."""
        try:
            if id_str.startswith('0x') or id_str.startswith('!'):
                return id_str
            else:
                id_int = int(id_str)
                return f"!{id_int:08x}"
        except ValueError:
            return id_str
        
    def parse_node_data(self, nodes_data: List[Dict]) -> Dict:
        """
        Parse and normalize node data.
        Extract important fields like id, label, role, etc.
        """
        parsed_nodes = {}
        
        for node in nodes_data:
            node_id = node.get('id')
            if not node_id:
                continue
                
            parsed_node = {
                'id': node_id,
                'label': node.get('label', str(node_id)),
                'role': node.get('role', 'CLIENT'),
                'x': node.get('x', 0),
                'y': node.get('y', 0),
                'battery': node.get('battery'),
                'last_heard': node.get('last_heard'),
                'snr': node.get('snr'),
                'hops_away': node.get('hops_away', 0),
                'distance': node.get('distance', 0),
                'neighbors': set(),  # Will be populated from edges
                'is_router': node.get('role') in ['ROUTER', 'ROUTER_LATE'],
                'hardware': node.get('hardware', ''),
                'short_name': node.get('short_name', ''),
                'long_name': node.get('long_name', ''),
                'channel': node.get('channel', ''),
                'description': node.get('description', ''),
                'raw_data': node  # Keep original data
            }
            
            parsed_nodes[node_id] = parsed_node
            
        return parsed_nodes
        
    def parse_edge_data(self, edges_data: List[Dict], nodes: Dict) -> List[Dict]:
        """
        Parse and normalize edge data.
        Also update node neighbor information.
        """
        parsed_edges = []
        
        for edge in edges_data:
            from_id = edge.get('from')
            to_id = edge.get('to')
            
            if not from_id or not to_id:
                continue
                
            # Skip edges where one node doesn't exist
            if from_id not in nodes or to_id not in nodes:
                continue
                
            parsed_edge = {
                'from': from_id,
                'to': to_id,
                'weight': edge.get('weight', 1),
                'snr': edge.get('snr'),
                'hop_limit': edge.get('hop_limit'),
                'originalColor': edge.get('originalColor'),
                'description': edge.get('description', ''),
                'raw_data': edge
            }
            
            parsed_edges.append(parsed_edge)
            
            # Update neighbor information
            if from_id in nodes:
                nodes[from_id]['neighbors'].add(to_id)
            if to_id in nodes:
                nodes[to_id]['neighbors'].add(from_id)
                
        return parsed_edges
        
    def get_sample_data(self) -> Dict:
        """
        Generate sample data for testing when real data is unavailable.
        """
        import random
        
        num_nodes = 50
        num_routers = 15
        
        # Define available channels
        channels = ['MediumFast', 'MediumSlow', 'LongFast', 'LongSlow', 'VeryLongSlow']
        
        nodes = []
        for i in range(num_nodes):
            is_router = i < num_routers
            # Assign channels to create groups
            if i < 20:
                channel = 'MediumSlow'
            elif i < 35:
                channel = 'MediumFast'
            elif i < 45:
                channel = 'LongFast'
            else:
                channel = 'LongSlow'
                
            node = {
                'id': f'node_{i}',
                'label': f'Node {i}',
                'role': 'ROUTER' if is_router else 'CLIENT',
                'x': random.randint(0, 800),
                'y': random.randint(0, 600),
                'battery': random.randint(20, 100),
                'snr': random.uniform(-20, 10),
                'hops_away': random.randint(0, 5),
                'distance': random.randint(1, 10),
                'neighbors': random.randint(1, 6),
                'hardware': random.choice(['T-Beam', 'Heltec V3', 'RAK4631', 'Station G1']),
                'short_name': f'N{i}',
                'long_name': f'Sample Node {i} ({channel})',
                'channel': channel
            }
            nodes.append(node)
            
        edges = []
        # Create a somewhat realistic mesh network
        for i in range(num_nodes):
            # Each node connects to 2-5 nearby nodes
            num_connections = random.randint(2, 5)
            for _ in range(num_connections):
                j = random.randint(0, num_nodes - 1)
                if i != j:
                    edge = {
                        'from': f'node_{i}',
                        'to': f'node_{j}',
                        'weight': random.uniform(0.5, 1.0),
                        'snr': random.uniform(-20, 10)
                    }
                    edges.append(edge)
                    
        return {
            'nodes': nodes,
            'edges': edges,
            'timestamp': datetime.now().isoformat()
        }


class DataProcessor:
    """Process and prepare data for analysis."""
    
    @staticmethod
    def validate_network_data(data: Dict) -> bool:
        """Validate that the network data has required fields."""
        if not isinstance(data, dict):
            return False
            
        if 'nodes' not in data or 'edges' not in data:
            return False
            
        if not isinstance(data['nodes'], list) or not isinstance(data['edges'], list):
            return False
            
        # Check if nodes have required fields
        if data['nodes']:
            required_fields = {'id'}
            sample_node = data['nodes'][0]
            if not all(field in sample_node for field in required_fields):
                return False
                
        return True
        
    @staticmethod
    def clean_network_data(data: Dict) -> Dict:
        """Clean and standardize network data."""
        cleaned_nodes = []
        node_ids = set()
        
        # Clean nodes
        for node in data['nodes']:
            if 'id' in node and node['id'] not in node_ids:
                node_ids.add(node['id'])
                cleaned_nodes.append(node)
                
        # Clean edges - remove edges with non-existent nodes
        cleaned_edges = []
        for edge in data['edges']:
            if edge.get('from') in node_ids and edge.get('to') in node_ids:
                cleaned_edges.append(edge)
                
        return {
            'nodes': cleaned_nodes,
            'edges': cleaned_edges,
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        }


if __name__ == "__main__":
    # Test the data fetcher
    logging.basicConfig(level=logging.INFO)
    
    fetcher = MeshDataFetcher()
    
    # Try to fetch real data
    try:
        data = fetcher.fetch_network_data()
        print(f"Successfully fetched {len(data['nodes'])} nodes and {len(data['edges'])} edges")
        
        # Parse the data
        nodes = fetcher.parse_node_data(data['nodes'])
        edges = fetcher.parse_edge_data(data['edges'], nodes)
        
        print(f"Parsed {len(nodes)} nodes")
        print(f"Found {sum(1 for n in nodes.values() if n['is_router'])} routers")
        
        # Show some sample data
        if nodes:
            sample_node = list(nodes.values())[0]
            print(f"\nSample node: {sample_node['id']}")
            print(f"  Label: {sample_node['label']}")
            print(f"  Role: {sample_node['role']}")
            print(f"  Is Router: {sample_node['is_router']}")
            print(f"  Hardware: {sample_node.get('hardware', 'N/A')}")
            print(f"  Channel: {sample_node.get('channel', 'N/A')}")
        
    except Exception as e:
        print(f"Failed to fetch real data: {e}")
        import traceback
        traceback.print_exc()
        print("\nUsing sample data instead...")
        
        data = fetcher.get_sample_data()
        print(f"Generated {len(data['nodes'])} sample nodes and {len(data['edges'])} edges") 