"""
Main runner - Orchestrates daily stock scanning
"""
import os
import sys
import logging
from datetime import datetime
import argparse

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from btst_scanner import BTSTScanner, run_btst_scan
from swing_scanner import SwingScanner, run_swing_scan
from notifier import Notifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(scan_type: str = "both", test_mode: bool = False):
    """
    Main function to run stock scans
    
    Args:
        scan_type: "btst", "swing", or "both"
        test_mode: If True, scan only a small subset
    """
    try:
        # Validate config
        Config.validate()
        
        notifier = Notifier()
        
        logger.info(f"Starting {scan_type} scan (test_mode={test_mode})")
        
        # Run BTST scan
        if scan_type in ["btst", "both"]:
            logger.info("=" * 50)
            logger.info("RUNNING BTST SCAN")
            logger.info("=" * 50)
            
            try:
                candidates, report = run_btst_scan(test_mode=test_mode)
                
                logger.info(f"Found {len(candidates)} BTST candidates")
                
                # Send notification
                if candidates:
                    notifier.notify_btst_results(report, len(candidates))
                    logger.info("BTST notification sent")
                else:
                    logger.info("No BTST candidates - no notification sent")
            
            except Exception as e:
                logger.error(f"BTST scan failed: {e}", exc_info=True)
                notifier.notify_error(f"BTST scan failed: {str(e)}")
        
        # Run Swing scan
        if scan_type in ["swing", "both"]:
            logger.info("=" * 50)
            logger.info("RUNNING SWING SCAN")
            logger.info("=" * 50)
            
            try:
                candidates, report = run_swing_scan(test_mode=test_mode)
                
                logger.info(f"Found {len(candidates)} swing candidates")
                
                # Send notification
                if candidates:
                    notifier.notify_swing_results(report, len(candidates))
                    logger.info("Swing notification sent")
                else:
                    logger.info("No swing candidates - no notification sent")
            
            except Exception as e:
                logger.error(f"Swing scan failed: {e}", exc_info=True)
                notifier.notify_error(f"Swing scan failed: {str(e)}")
        
        logger.info("=" * 50)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            notifier = Notifier()
            notifier.notify_error(f"Fatal error in scanner: {str(e)}")
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Scanner")
    parser.add_argument(
        "--type",
        choices=["btst", "swing", "both"],
        default="both",
        help="Type of scan to run"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (scan only 10 stocks)"
    )
    
    args = parser.parse_args()
    
    main(scan_type=args.type, test_mode=args.test)
