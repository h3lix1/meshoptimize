"""
Data Fetcher Module for Meshtastic Network Optimizer
Handles fetching and parsing network data from meshview.bayme.sh
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class MeshDataFetcher:
    """Fetches and parses Meshtastic network data from various sources."""
    
    def __init__(self, url: str = "https://meshview.bayme.sh/nodegraph"):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; MeshOptimizer/1.0)'
        })
        self._cache = {}
        self._cache_time = None
        self.cache_duration = 300  # 5 minutes cache
        
    def fetch_network_data(self, use_cache: bool = True) -> Dict:
        """
        Fetch network data from meshview.bayme.sh
        Returns a dictionary with nodes and edges data.
        """
        # Check cache
        if use_cache and self._cache_time:
            if (datetime.now() - self._cache_time).seconds < self.cache_duration:
                logger.info("Using cached data")
                return self._cache
                
        try:
            logger.info(f"Fetching data from {self.url}")
            response = self.session.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract JavaScript data
            script_tags = soup.find_all('script')
            
            nodes_data = None
            edges_data = None
            
            for script in script_tags:
                if script.string:
                    content = script.string
                    
                    # Look for nodes in the meshview.bayme.sh format
                    if nodes_data is None and 'const nodes = [' in content:
                        nodes_match = re.search(r'const\s+nodes\s*=\s*(\[[\s\S]*?\]);', content)
                        if nodes_match:
                            try:
                                nodes_str = nodes_match.group(1)
                                # Parse the custom format
                                nodes_data = self._parse_meshview_nodes(nodes_str)
                                logger.info(f"Found {len(nodes_data)} nodes in meshview format")
                            except Exception as e:
                                logger.error(f"Failed to parse meshview nodes: {e}")
                    
                    # Look for edges in the meshview.bayme.sh format
                    if edges_data is None and 'const edges = [' in content:
                        edges_match = re.search(r'const\s+edges\s*=\s*(\[[\s\S]*?\]);', content)
                        if edges_match:
                            try:
                                edges_str = edges_match.group(1)
                                # Parse the custom format
                                edges_data = self._parse_meshview_edges(edges_str)
                                logger.info(f"Found {len(edges_data)} edges in meshview format")
                            except Exception as e:
                                logger.error(f"Failed to parse meshview edges: {e}")
            
            # If we still don't have data, use sample data
            if nodes_data is None or edges_data is None:
                logger.warning("Could not parse real data, using sample data")
                return self.get_sample_data()
            
            result = {
                'nodes': nodes_data,
                'edges': edges_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self._cache = result
            self._cache_time = datetime.now()
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {e}")
            # Return cached data if available
            if self._cache:
                logger.warning("Using stale cache due to fetch error")
                return self._cache
            # Otherwise return sample data
            logger.warning("No cache available, using sample data")
            return self.get_sample_data()
    
    def _parse_meshview_nodes(self, js_str: str) -> List[Dict]:
        """Parse nodes in the meshview.bayme.sh format."""
        nodes = []
        
        # Split by node objects more carefully
        # Look for patterns starting with { and containing name:
        node_blocks = re.split(r'(?=\s*\{\s*name:)', js_str)
        
        for block in node_blocks:
            if 'name:' not in block:
                continue
                
            try:
                node_data = {}
                
                # Extract name (node ID)
                name_match = re.search(r'name:\s*`([^`]+)`', block)
                if name_match:
                    node_data['id'] = name_match.group(1)
                else:
                    continue  # Skip if no ID
                
                # Extract value (label)
                value_match = re.search(r'value:\s*`([^`]+)`', block)
                if value_match:
                    # Remove quotes and escape sequences from the value
                    label = value_match.group(1).strip('"')
                    # Decode unicode escapes
                    try:
                        label = label.encode('utf-8').decode('unicode_escape')
                    except:
                        pass
                    node_data['label'] = label
                
                # Extract role
                role_match = re.search(r'role:\s*`([^`]+)`', block)
                if role_match:
                    node_data['role'] = role_match.group(1)
                else:
                    node_data['role'] = 'CLIENT'
                
                # Extract hardware model
                hw_match = re.search(r'hw_model:\s*`([^`]+)`', block)
                if hw_match:
                    node_data['hardware'] = hw_match.group(1)
                
                # Extract short name
                short_match = re.search(r'short_name:\s*`([^`]+)`', block)
                if short_match:
                    node_data['short_name'] = short_match.group(1)
                
                # Extract long name
                long_match = re.search(r'long_name:\s*`([^`]+)`', block)
                if long_match:
                    long_name = long_match.group(1)
                    # Decode HTML entities
                    long_name = long_name.replace('&lt;', '<').replace('&gt;', '>')
                    node_data['long_name'] = long_name
                
                # Extract channel
                channel_match = re.search(r'channel:\s*`([^`]+)`', block)
                if channel_match:
                    node_data['channel'] = channel_match.group(1)
                
                nodes.append(node_data)
                    
            except Exception as e:
                logger.debug(f"Failed to parse node: {e}")
                continue
        
        logger.info(f"Parsed {len(nodes)} nodes from raw data")
        return nodes
    
    def _parse_meshview_edges(self, js_str: str) -> List[Dict]:
        """Parse edges in the meshview.bayme.sh format."""
        edges = []
        
        # Use regex to extract each edge object
        edge_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        
        # Find all edge objects
        edge_matches = re.finditer(edge_pattern, js_str)
        
        for match in edge_matches:
            edge_str = match.group(0)
            
            try:
                edge_data = {}
                
                # Extract source
                source_match = re.search(r'source:\s*`([^`]+)`', edge_str)
                if source_match:
                    edge_data['from'] = source_match.group(1)
                
                # Extract target
                target_match = re.search(r'target:\s*`([^`]+)`', edge_str)
                if target_match:
                    edge_data['to'] = target_match.group(1)
                
                # Extract color (for weight estimation)
                color_match = re.search(r'originalColor:\s*`([^`]+)`', edge_str)
                if color_match:
                    edge_data['originalColor'] = color_match.group(1)
                
                # Extract line width if available
                width_match = re.search(r'width:\s*(\d+(?:\.\d+)?)', edge_str)
                if width_match:
                    edge_data['weight'] = float(width_match.group(1)) / 2.0
                else:
                    edge_data['weight'] = 1.0
                
                # Only add if we have both source and target
                if 'from' in edge_data and 'to' in edge_data:
                    edges.append(edge_data)
                    
            except Exception as e:
                logger.debug(f"Failed to parse edge: {e}")
                continue
        
        return edges
            
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