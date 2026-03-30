import networkx as nx
import logging
import pickle
import os
from typing import Dict, List, Optional, Tuple

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CricketKnowledgeGraph:
    """GraphRAG-style knowledge base for deep matchup historical memory."""
    def __init__(self, storage_path: str = "models/matchup_graph.pkl"):
        self.graph = nx.MultiDiGraph()
        self.storage_path = storage_path
        
        # Load existing graph if available
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"✅ Loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.warning(f"❌ Failed to load Knowledge Graph from {self.storage_path}: {e}. Creating new.")

    def add_matchup_event(self, batsman: str, bowler: str, runs: int, wicket: bool, context: Dict):
        """Adds a single ball outcome between batsman and bowler as an edge."""
        # Clean labels
        batsman, bowler = batsman.strip(), bowler.strip()
        
        # Add edge with attributes
        self.graph.add_edge(
            bowler, 
            batsman, 
            runs=runs, 
            wicket=wicket, 
            venue=context.get('venue', 'N/A'),
            match_id=context.get('match_id', 'N/A')
        )

    def query_matchup_stats(self, batsman: str, bowler: str) -> Dict[str, float]:
        """Expressed as [Bowler -> Batsman] edges. Returns statistical summary."""
        batsman, bowler = batsman.strip(), bowler.strip()
        
        if not self.graph.has_edge(bowler, batsman):
            return {
                'matchup_count': 0,
                'avg_runs_per_ball': 0.0,
                'wicket_rate': 0.0,
                'dominance_index': 0.5  # Neutral index
            }
        
        # Extract edge data
        edge_data = self.graph.get_edge_data(bowler, batsman)
        balls = len(edge_data)
        runs = sum(e['runs'] for e in edge_data.values())
        wickets = sum(1 for e in edge_data.values() if e['wicket'])
        
        # Dominance Heuristic: High if batsman dominates bowler
        # (Average runs per ball vs Wicket frequency)
        avg_runs = runs / balls
        wicket_freq = wickets / balls
        
        # Dominance Index: 1.0 (Batsman dominates), 0.0 (Bowler dominates)
        dominance = (avg_runs / 2.0) - (wicket_freq * 10.0)
        dominance = max(0.0, min(1.0, 0.5 + dominance)) # Clamped between 0 and 1
        
        return {
            'matchup_count': balls,
            'avg_runs_per_ball': avg_runs,
            'wicket_rate': wicket_freq,
            'dominance_index': dominance
        }

    def save(self):
        """Persist the graph to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"✅ Knowledge Graph saved to {self.storage_path}.")
