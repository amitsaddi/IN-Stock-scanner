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

        # Track results for summary
        btst_count = 0
        swing_count = 0
        btst_success = True
        swing_success = True

        # Run BTST scan
        if scan_type in ["btst", "both"]:
            logger.info("=" * 50)
            logger.info("RUNNING BTST SCAN")
            logger.info("=" * 50)

            try:
                candidates, report = run_btst_scan(test_mode=test_mode)
                btst_count = len(candidates)

                logger.info(f"Found {btst_count} BTST candidates")

                # Send notification if candidates found
                if candidates:
                    notifier.notify_btst_results(report, btst_count)
                    logger.info("BTST notification sent")
                else:
                    logger.info("No BTST candidates found")

            except Exception as e:
                btst_success = False
                logger.error(f"BTST scan failed: {e}", exc_info=True)
                notifier.notify_error(f"BTST scan failed: {str(e)}")

        # Run Swing scan
        if scan_type in ["swing", "both"]:
            logger.info("=" * 50)
            logger.info("RUNNING SWING SCAN")
            logger.info("=" * 50)

            try:
                candidates, report = run_swing_scan(test_mode=test_mode)
                swing_count = len(candidates)

                logger.info(f"Found {swing_count} swing candidates")

                # Send notification if candidates found
                if candidates:
                    notifier.notify_swing_results(report, swing_count)
                    logger.info("Swing notification sent")
                else:
                    logger.info("No swing candidates found")

            except Exception as e:
                swing_success = False
                logger.error(f"Swing scan failed: {e}", exc_info=True)
                notifier.notify_error(f"Swing scan failed: {str(e)}")

        logger.info("=" * 50)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 50)

        # Always send completion summary to Telegram
        send_completion_summary(notifier, scan_type, btst_count, swing_count, btst_success, swing_success)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            notifier = Notifier()
            notifier.notify_error(f"Fatal error in scanner: {str(e)}")
        except:
            pass
        sys.exit(1)


def send_completion_summary(notifier, scan_type, btst_count, swing_count, btst_success, swing_success):
    """Send scan completion summary to Telegram"""
    from datetime import datetime

    timestamp = datetime.now().strftime('%d %b %Y, %I:%M %p IST')

    summary = f"<b>📊 Stock Scanner - Scan Complete</b>\n"
    summary += f"<i>{timestamp}</i>\n\n"

    if scan_type in ["btst", "both"]:
        if btst_success:
            summary += f"🟢 <b>BTST Scan:</b> {btst_count} opportunities found\n"
        else:
            summary += f"❌ <b>BTST Scan:</b> Failed\n"

    if scan_type in ["swing", "both"]:
        if swing_success:
            summary += f"🟠 <b>Swing Scan:</b> {swing_count} stocks found\n"
        else:
            summary += f"❌ <b>Swing Scan:</b> Failed\n"

    summary += "\n"

    if btst_count == 0 and swing_count == 0 and btst_success and swing_success:
        summary += "ℹ️ <i>No trading opportunities found today.</i>\n"
        summary += "The scanner ran successfully but market conditions didn't meet our criteria."
    elif btst_count > 0 or swing_count > 0:
        summary += "✅ <i>Detailed reports sent via email.</i>"

    # Always send to Telegram for completion status
    if notifier.config.SEND_TELEGRAM:
        notifier.send_telegram(summary)
        logger.info("Completion summary sent to Telegram")
    else:
        logger.info("Telegram disabled - completion summary not sent")


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
