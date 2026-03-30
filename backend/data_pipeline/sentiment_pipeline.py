import os
import logging
import numpy as np
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentPipeline:
    """Real-time sentiment analyzer for social media match reports."""
    def __init__(self, 
                 consumer_key: str = None, 
                 consumer_secret: str = None, 
                 access_token: str = None, 
                 access_token_secret: str = None):
        
        self.analyzer = SentimentIntensityAnalyzer()
        self.api = None
        
        # Initialize Twitter API if keys are provided
        consumer_key = consumer_key or os.getenv("TWITTER_CONSUMER_KEY")
        consumer_secret = consumer_secret or os.getenv("TWITTER_CONSUMER_SECRET")
        access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN")
        access_token_secret = access_token_secret or os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        
        if all([consumer_key, consumer_secret, access_token, access_token_secret]):
            try:
                auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
                auth.set_access_token(access_token, access_token_secret)
                self.api = tweepy.API(auth)
                logger.info("✅ Twitter API connection established.")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Twitter API: {e}")
        else:
            logger.warning("⚠️ Twitter API keys missing. Sentiment will use mock random data.")

    def get_match_sentiment(self, hashtag: str = "#IPL2024") -> Dict[str, float]:
        """Scrape latest 100 tweets and return aggregate sentiment."""
        if not self.api:
            # Mock data for demonstration when offline/unauthenticated
            return {
                'sentiment_score': float(np.random.uniform(0.3, 0.8)),
                'sentiment_volatility': float(np.random.uniform(0.05, 0.2))
            }
            
        try:
            # Search tweets with the given hashtag
            tweets = self.api.search_tweets(q=hashtag, count=100, lang='en', result_type='recent')
            scores = []
            
            for tweet in tweets:
                # VADER: compound score (-1.0 to 1.0)
                sentiment = self.analyzer.polarity_scores(tweet.text)['compound']
                scores.append(sentiment)
            
            if not scores:
                return {'sentiment_score': 0.0, 'sentiment_volatility': 0.0}
                
            return {
                'sentiment_score': float(np.mean(scores)),
                'sentiment_volatility': float(np.std(scores))
            }
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {'sentiment_score': 0.0, 'sentiment_volatility': 0.0}

class MomentumMonitor:
    """Combines sentiment and simulation scores into a unified momentum feature."""
    def __init__(self):
        # We integrate the MiroFish sentiment signal into our RealTimePredictor
        pass
