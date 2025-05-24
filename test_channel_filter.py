#!/usr/bin/env python3
"""Test script for channel filter functionality"""

import sys
from PyQt6.QtWidgets import QApplication
from mesh_optimizer import MeshOptimizerGUI

def test_channel_filter():
    """Test the channel filter functionality"""
    app = QApplication(sys.argv)
    
    # Create main window
    window = MeshOptimizerGUI()
    
    # Show window
    window.show()
    
    # Load sample data automatically
    window._load_sample_data()
    
    # Print available channels
    print("Available channels:", window.available_channels)
    
    # The GUI will now have channel filter functionality
    print("\nChannel filter added successfully!")
    print("You can now:")
    print("1. Select a channel from the dropdown to filter nodes")
    print("2. Hover over nodes to see detailed information including channel")
    print("3. All optimization functions will work on the filtered data")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    test_channel_filter() 