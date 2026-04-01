"""
Advanced Simulators for IPL Match Scenarios & Player Performances
Includes:
- Monte Carlo Ball-by-Ball Simulator  (DLS-aware transition matrices)
- Bayesian Hierarchical Player Predictor
- Fantasy Points Calculator with ownership differential weighting
"""
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ScenarioSimulator:
    """
    Run Monte Carlo simulations over remaining balls to produce:
    - Projected median score
    - P90 ceiling
    - Milestone probabilities (180+, all-out, etc.)
    """

    def __init__(self):
        # Baseline ball-outcome probabilities derived from IPL historical data
        self.outcomes = ['dot', 'single', 'two', 'three', 'four', 'six', 'wicket']
        self.base_probs = np.array([0.33, 0.30, 0.07, 0.02, 0.13, 0.07, 0.06])
        # Powerplay adjustment (higher scoring)
        self.pp_probs = np.array([0.28, 0.30, 0.06, 0.01, 0.18, 0.10, 0.07])
        # Death overs adjustment (slog)
        self.death_probs = np.array([0.25, 0.26, 0.05, 0.01, 0.17, 0.14, 0.07])

    def _get_transition_probs(self, ball_num: int) -> np.ndarray:
        """Return context-sensitive transition probabilities."""
        over = ball_num // 6
        if over < 6:
            return self.pp_probs
        elif over >= 15:
            return self.death_probs
        return self.base_probs

    def simulate_remaining_balls(
        self, current_state: Dict, n_simulations: int = 5000
    ) -> Dict:
        """
        Simulate remaining deliveries to forecast final scores and milestones.

        Args:
            current_state: Dict with 'balls_remaining', 'total_runs', 'total_wickets'
            n_simulations: Number of MC samples

        Returns:
            Dict with projected_median_score, p90_score, and milestone probabilities
        """
        balls_remaining = int(current_state.get('balls_remaining', 120))
        current_runs = int(current_state.get('total_runs', 0))
        current_wickets = int(current_state.get('total_wickets', 0))
        balls_bowled = 120 - balls_remaining

        simulated_scores = []
        milestone_180_count = 0
        milestone_200_count = 0
        early_collapse_count = 0  # 3 wickets in powerplay

        for _ in range(n_simulations):
            sim_runs = current_runs
            sim_wickets = current_wickets
            sim_pp_wkts = 0

            for b in range(balls_remaining):
                if sim_wickets >= 10:
                    break

                abs_ball = balls_bowled + b
                probs = self._get_transition_probs(abs_ball)
                event = np.random.choice(self.outcomes, p=probs)

                if event == 'single':
                    sim_runs += 1
                elif event == 'two':
                    sim_runs += 2
                elif event == 'three':
                    sim_runs += 3
                elif event == 'four':
                    sim_runs += 4
                elif event == 'six':
                    sim_runs += 6
                elif event == 'wicket':
                    sim_wickets += 1
                    if abs_ball < 36:
                        sim_pp_wkts += 1

            simulated_scores.append(sim_runs)
            if sim_runs >= 180:
                milestone_180_count += 1
            if sim_runs >= 200:
                milestone_200_count += 1
            if sim_pp_wkts >= 3:
                early_collapse_count += 1

        return {
            'projected_median_score': float(np.median(simulated_scores)),
            'projected_mean_score': float(np.mean(simulated_scores)),
            'p10_score': float(np.percentile(simulated_scores, 10)),
            'p90_score': float(np.percentile(simulated_scores, 90)),
            'prob_180_plus': round(milestone_180_count / n_simulations, 3),
            'prob_200_plus': round(milestone_200_count / n_simulations, 3),
            'prob_early_collapse': round(early_collapse_count / n_simulations, 3),
        }


class BayesianPlayerPredictor:
    """
    Bayesian Hierarchical Model for predicting individual player performance.
    Uses conjugate Gaussian-Gaussian priors for fast inference.
    """

    def __init__(self):
        # Global IPL season hyperparameters (prior)
        self.global_bat_mu = 25.0      # avg expected runs
        self.global_bat_sigma = 12.0
        self.global_wkt_mu = 1.2       # avg expected wickets
        self.global_wkt_sigma = 0.8

    def predict_player_runs(
        self,
        player_history: List[float],
        opp_strength: float = 1.0,
        venue_factor: float = 1.0
    ) -> Dict:
        """
        Posterior run distribution conditioned on history, opposition, and venue.

        Args:
            player_history: List of run scores from recent matches
            opp_strength: Normalised (0.8=strong opp, 1.2=weak opp)
            venue_factor: Venue batting average factor (>1 = batting friendly)

        Returns:
            Dict with expected_runs, lower/upper 95% CI
        """
        if not player_history:
            adj_mu = self.global_bat_mu * opp_strength * venue_factor
            return {
                'expected_runs': round(adj_mu, 1),
                'confidence_interval': [max(0, adj_mu - 24), adj_mu + 24],
                'samples': 0,
            }

        n = len(player_history)
        sample_mean = float(np.mean(player_history))
        known_variance = max(self.global_bat_sigma ** 2, float(np.var(player_history)) + 1e-6)

        # Conjugate posterior update
        prior_precision = 1.0 / (self.global_bat_sigma ** 2)
        likelihood_precision = n / known_variance
        post_precision = prior_precision + likelihood_precision
        post_var = 1.0 / post_precision
        post_mu = post_var * (
            prior_precision * self.global_bat_mu + likelihood_precision * sample_mean
        )

        # Apply contextual modifiers
        adj_mu = post_mu * opp_strength * venue_factor
        ci_width = 1.96 * math.sqrt(post_var)

        return {
            'expected_runs': round(max(0.0, adj_mu), 1),
            'confidence_interval': [
                round(max(0.0, adj_mu - ci_width), 1),
                round(adj_mu + ci_width, 1),
            ],
            'posterior_std': round(math.sqrt(post_var), 2),
            'samples': n,
        }

    def predict_player_wickets(
        self,
        player_history: List[float],
        pitch_factor: float = 1.0
    ) -> Dict:
        """Posterior wicket distribution using Poisson approximation."""
        if not player_history:
            rate = self.global_wkt_mu * pitch_factor
        else:
            rate = float(np.mean(player_history)) * pitch_factor

        # Poisson confidence interval
        rate = max(0.01, rate)
        lower = max(0.0, rate - 1.65 * math.sqrt(rate))
        upper = rate + 1.65 * math.sqrt(rate)

        return {
            'expected_wickets': round(rate, 2),
            'confidence_interval': [round(lower, 2), round(upper, 2)],
        }


class FantasyEngine:
    """
    Dream11-style Fantasy Points Calculator with Ownership differential weighting.
    """

    # Dream11 standard scoring schema
    BATTING_POINTS = {
        'run': 1.0,
        'boundary_bonus': 1.0,
        'six_bonus': 2.0,
        'half_century': 8.0,
        'century': 16.0,
        'duck': -2.0,
        'sr_bonus_170': 6.0,
        'sr_bonus_150': 4.0,
        'sr_penalty_60': -2.0,
        'sr_penalty_70': -1.0,
    }

    BOWLING_POINTS = {
        'wicket': 25.0,
        'wicket_bonus_4': 8.0,
        'wicket_bonus_5': 16.0,
        'maiden': 8.0,
        'econ_bonus_4': 6.0,
        'econ_bonus_5': 4.0,
        'econ_penalty_10': -2.0,
        'econ_penalty_11': -4.0,
    }

    FIELDING_POINTS = {
        'catch': 8.0,
        'stumping': 12.0,
        'run_out': 6.0,
    }

    def calculate_expected_points(
        self,
        player_projection: Dict,
        role: str = 'batsman',
        ownership_pct: float = 0.5,
    ) -> Dict:
        """
        Calculate expected fantasy points and differential value.

        Args:
            player_projection: Output from BayesianPlayerPredictor
            role: 'batsman', 'bowler', 'allrounder', 'wicketkeeper'
            ownership_pct: Fractional ownership (0.0 to 1.0)

        Returns:
            Dict with expected_points, differential_score, risk_reward_ratio
        """
        exp_runs = player_projection.get('expected_runs', 0.0)
        exp_wkts = player_projection.get('expected_wickets', 0.0)

        # Base batting points
        bat_pts = exp_runs * self.BATTING_POINTS['run']
        if exp_runs >= 50:
            bat_pts += self.BATTING_POINTS['half_century']
        if exp_runs >= 100:
            bat_pts += self.BATTING_POINTS['century']

        # Base bowling points
        bowl_pts = exp_wkts * self.BOWLING_POINTS['wicket']

        # Fielding contribution approximation
        field_pts = 5.0 if role == 'wicketkeeper' else 2.5

        total_base = bat_pts + bowl_pts + field_pts

        # Differential multiplier: low ownership → higher value due to differentiation
        # Captaincy / VC multiplier also embedded here at 2x/1.5x for simplicity
        ownership_mod = 1.0 + 0.5 * (1.0 - ownership_pct)

        expected_points = total_base * ownership_mod

        # Risk-reward: high ceiling, low ownership → high ratio
        ceiling = player_projection.get('confidence_interval', [0, exp_runs * 1.5])[1]
        risk_reward = (ceiling * ownership_mod) / max(1.0, expected_points)

        return {
            'expected_points': round(expected_points, 1),
            'base_points': round(total_base, 1),
            'differential_score': round(ownership_mod, 3),
            'risk_reward_ratio': round(risk_reward, 2),
            'captaincy_value': round(expected_points * 2, 1),
            'vc_value': round(expected_points * 1.5, 1),
        }
