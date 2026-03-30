import os
import random
import logging
import numpy as np
from openai import OpenAI
from typing import Dict, List, Optional

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CricketAnalystAgent:
    """An AI agent representing a specific cricket analyst archetype."""
    def __init__(self, bias_type: str, client: Optional[OpenAI] = None):
        self.bias_type = bias_type  # e.g., 'pitch_focused', 'form_focused', 'tactical'
        self.client = client

    def analyze(self, match_state: Dict) -> float:
        """Return a sentiment/confidence score [0-1] based on the agent's bias."""
        if self.client:
            try:
                # LLM-based analysis for high-fidelity qualitative signal
                prompt = self._build_prompt(match_state)
                response = self.client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=60,
                    temperature=0.7
                )
                text = response.choices[0].text.strip().lower()
                
                # Simple sentiment heuristic
                if any(word in text for word in ["high", "confident", "easy", "dominant"]):
                    return 0.85
                elif any(word in text for word in ["low", "worried", "struggle", "collapse"]):
                    return 0.25
                else:
                    return 0.5
            except Exception as e:
                logger.warning(f"LLM Agent ({self.bias_type}) failed: {e}. Falling back to simulation.")
        
        # Fallback deterministic simulation based on match state
        return self._simulate_score(match_state)

    def _build_prompt(self, match_state: Dict) -> str:
        return (f"As a cricket analyst specializing in {self.bias_type}, analyze this IPL match state: "
                f"{match_state}. Does the batting team look dominant or under pressure? "
                f"Give a short 1-sentence analysis and a confidence score.")

    def _simulate_score(self, match_state: Dict) -> float:
        """Heuristic-based fallback logic to ensure speed when API is unavailable."""
        if self.bias_type == 'pitch_focused':
            # Pitch focus: Confidence high if runs are being scored
            crr = match_state.get('crr', 6.0)
            return 0.8 if crr > 9.0 else 0.4
        elif self.bias_type == 'form_focused':
            # Form focus: Confidence high if few wickets lost
            wickets = match_state.get('total_wickets', 0)
            return 0.9 if wickets < 2 else 0.4
        elif self.bias_type == 'momentum_analyst':
            # Momentum focus: Recent runs
            recent_runs = match_state.get('runs_last_6', 6)
            return 0.85 if recent_runs > 12 else 0.35
        return 0.5


class MultiAgentSimulator:
    """Manages a swarm of analysts to extract emergent confidence features."""
    def __init__(self, num_agents: int = 10, api_key: Optional[str] = None):
        self.num_agents = num_agents
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.biases = ['pitch_focused', 'form_focused', 'momentum_analyst', 'tactical_observer', 'collapse_historian']
        self.agents = [CricketAnalystAgent(random.choice(self.biases), self.client) for _ in range(num_agents)]

    def simulate(self, match_state: Dict) -> Dict[str, float]:
        """Aggregate analysis scores into aggregate ensemble features."""
        scores = [agent.analyze(match_state) for agent in self.agents]
        
        # We extract two key metrics for the XGBoost model
        # 1. Agent Confidence: The average sentiment across the swarm
        # 2. Agent Disagreement (Entropy): The standard deviation (uncertainty)
        
        return {
            'agent_confidence': float(np.mean(scores)),
            'agent_disagreement': float(np.std(scores))
        }
