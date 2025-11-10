"""
Configuration file for stock scanner
"""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class BTSTCriteria:
    """BTST scanning criteria"""
    min_gain_percent: float = 2.0
    max_gain_percent: float = 3.5
    min_volume_multiplier: float = 1.5
    min_delivery_percent: float = 40.0
    scan_time: str = "15:15"  # 3:15 PM IST
    max_results: int = 10

@dataclass
class SwingCriteria:
    """Swing trading criteria"""
    min_market_cap: float = 5000  # crores
    max_debt_to_equity: float = 0.5
    min_roe: float = 15.0
    rsi_min: int = 40
    rsi_max: int = 60
    min_volume_multiplier: float = 1.2
    max_results: int = 15

@dataclass
class TechnicalParams:
    """Technical analysis parameters"""
    ema_short: int = 20
    ema_medium: int = 50
    sma_long: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_lookback: int = 20

class Config:
    """Main configuration class"""
    
    # Environment
    ENV = os.getenv("ENV", "production")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Data source
    DATA_SOURCE = os.getenv("DATA_SOURCE", "yfinance")  # yfinance or nsepy
    
    # Stock universe
    NIFTY_500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    
    # Notification settings
    SEND_EMAIL = os.getenv("SEND_EMAIL", "True").lower() == "true"
    SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "False").lower() == "true"
    
    # Email configuration
    EMAIL_FROM = os.getenv("EMAIL_FROM", "")
    EMAIL_TO = os.getenv("EMAIL_TO", "")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    
    # Telegram configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # GitHub configuration
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO = os.getenv("GITHUB_REPO", "")
    
    # Scanning criteria
    BTST = BTSTCriteria()
    SWING = SwingCriteria()
    TECHNICAL = TechnicalParams()
    
    # File paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    NIFTY500_FILE = os.path.join(DATA_DIR, "nifty500.csv")
    
    # Market hours (IST)
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    
    # Rate limiting
    RATE_LIMIT_DELAY = 0.5  # seconds between API calls
    BATCH_SIZE = 50  # stocks per batch
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if cls.SEND_EMAIL:
            if not cls.EMAIL_FROM:
                errors.append("EMAIL_FROM not set")
            if not cls.EMAIL_TO:
                errors.append("EMAIL_TO not set")
            if not cls.EMAIL_PASSWORD:
                errors.append("EMAIL_PASSWORD not set")
        
        if cls.SEND_TELEGRAM:
            if not cls.TELEGRAM_BOT_TOKEN:
                errors.append("TELEGRAM_BOT_TOKEN not set")
            if not cls.TELEGRAM_CHAT_ID:
                errors.append("TELEGRAM_CHAT_ID not set")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True

# Exclude sectors for BTST (too volatile or news-driven)
BTST_EXCLUDED_SECTORS = [
    "IT",
    "PHARMA"
]

# Preferred sectors for swing trading (Nov 2025)
SWING_PREFERRED_SECTORS = [
    "DEFENCE",
    "CAPITAL GOODS",
    "INFRASTRUCTURE",
    "PSU BANKS",
    "RENEWABLE ENERGY"
]
