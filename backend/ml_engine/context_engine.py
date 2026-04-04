"""
Titan v4.8 — Sovereign DNA Engine (Contextual Intelligence)
===========================================================
Background service for H2H, Venue Edge, and Achievement Analysis.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 🏛️ Sovereign Achievement Registry (Historical Data)
IPL_TEAM_ACHIEVEMENTS = {
    "GT": {"titles": 1, "record_last_5": "W-L-W-W-L", "h2h_vs_rr": 0.60},
    "RR": {"titles": 1, "record_last_5": "W-W-L-W-W", "h2h_vs_gt": 0.40},
    "CSK": {"titles": 5, "record_last_5": "W-W-W-L-W", "h2h_default": 0.55},
    "MI": {"titles": 5, "record_last_5": "L-W-L-L-W", "h2h_default": 0.52},
}

class ContextualAuditor:
    """Computes DNA factors (H2H, Venue, Achievements) for live matches."""

    def __init__(self):
        self.registry = IPL_TEAM_ACHIEVEMENTS

    def get_match_dna(self, team1: str, team2: str, venue: str) -> Dict:
        """Analyze the DNA of a specific matchup."""
        t1_short = self._get_short(team1)
        t2_short = self._get_short(team2)

        # 1. H2H Edge
        h2h_edge = 0.5
        if t1_short in self.registry and f"h2h_vs_{t2_short.lower()}" in self.registry[t1_short]:
            h2h_edge = self.registry[t1_short][f"h2h_vs_{t2_short.lower()}"]
        
        # 2. Performance Achievement Factor
        t1_perf = self.registry.get(t1_short, {}).get("record_last_5", "L-L-L-L-L").count("W") / 5.0
        t2_perf = self.registry.get(t2_short, {}).get("record_last_5", "L-L-L-L-L").count("W") / 5.0
        
        # 3. Venue Sovereignty
        venue_edge = 0.0
        if "Ahmedabad" in venue and t1_short == "GT": venue_edge = 0.15 # GT home ground advantage
        if "Jaipur" in venue and t2_short == "RR": venue_edge = -0.12 # RR home ground advantage

        # 4. Total DNA Shift
        total_shift = (h2h_edge - 0.5) + (t1_perf - t2_perf) * 0.2 + venue_edge
        
        return {
            "h2h_edge": float(h2h_edge),
            "form_delta": float(t1_perf - t2_perf),
            "venue_edge": float(venue_edge),
            "total_dna_shift": float(total_shift),
            "insight": self._generate_insight(t1_short, t2_short, venue, h2h_edge)
        }

    def _get_short(self, team: str) -> str:
        # Simplified mapper
        mapping = {
            "Gujarat Titans": "GT", "Rajasthan Royals": "RR", 
            "Mumbai Indians": "MI", "Delhi Capitals": "DC",
            "Chennai Super Kings": "CSK", "Royal Challengers Bengaluru": "RCB"
        }
        for full, short in mapping.items():
            if full in team or short in team: return short
        return team[:3].upper()

    def _generate_insight(self, t1: str, t2: str, venue: str, h2h: float) -> str:
        if h2h > 0.55:
            return f"{t1} DNA dominates this matchup with a {h2h*100:.0f}% H2H record."
        if "Ahmedabad" in venue and t1 == "GT":
            return f"Sovereign Edge: GT holds a massive 15% local gradient in Ahmedabad."
        return f"Equilibrium detected in {t1} vs {t2} DNA profile."

# Singleton instance
_context_engine = ContextualAuditor()
def get_context_engine():
    return _context_engine
