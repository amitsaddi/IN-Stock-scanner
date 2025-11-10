"""
Notification module - Send scan results via email and Telegram
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Notifier:
    """Handle notifications via email and Telegram"""
    
    def __init__(self):
        self.config = Config()
    
    def send_email(self, subject: str, body: str) -> bool:
        """
        Send email notification
        
        Args:
            subject: Email subject
            body: Email body (plain text)
        
        Returns:
            True if sent successfully
        """
        if not self.config.SEND_EMAIL:
            logger.info("Email notifications disabled")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.EMAIL_FROM
            msg['To'] = self.config.EMAIL_TO
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT)
            server.starttls()
            server.login(self.config.EMAIL_FROM, self.config.EMAIL_PASSWORD)
            
            text = msg.as_string()
            server.sendmail(self.config.EMAIL_FROM, self.config.EMAIL_TO, text)
            server.quit()
            
            logger.info(f"Email sent to {self.config.EMAIL_TO}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_telegram(self, message: str) -> bool:
        """
        Send Telegram notification
        
        Args:
            message: Message text
        
        Returns:
            True if sent successfully
        """
        if not self.config.SEND_TELEGRAM:
            logger.info("Telegram notifications disabled")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            # Split long messages (Telegram limit is 4096 chars)
            max_length = 4000
            if len(message) > max_length:
                parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                for part in parts:
                    data = {
                        "chat_id": self.config.TELEGRAM_CHAT_ID,
                        "text": part,
                        "parse_mode": "HTML"
                    }
                    response = requests.post(url, data=data)
                    response.raise_for_status()
            else:
                data = {
                    "chat_id": self.config.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML"
                }
                response = requests.post(url, data=data)
                response.raise_for_status()
            
            logger.info(f"Telegram message sent to chat {self.config.TELEGRAM_CHAT_ID}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_for_telegram(self, text: str) -> str:
        """
        Format plain text for Telegram HTML
        
        Args:
            text: Plain text
        
        Returns:
            HTML formatted text
        """
        # Convert markdown-like symbols to HTML
        text = text.replace('✓', '✅')
        text = text.replace('✗', '❌')
        text = text.replace('⚠️', '⚠️')
        
        # Bold headers (lines with ==== under them)
        lines = text.split('\n')
        formatted_lines = []
        for i, line in enumerate(lines):
            if i > 0 and lines[i-1].strip().startswith('='):
                formatted_lines.append(f"<b>{line}</b>")
            elif line.strip().startswith('='):
                continue
            elif line.strip().endswith(':') and len(line) < 50:
                formatted_lines.append(f"<b>{line}</b>")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def notify_btst_results(self, report: str, count: int) -> bool:
        """
        Send BTST scan results notification
        
        Args:
            report: Report text
            count: Number of candidates found
        
        Returns:
            True if notification sent
        """
        subject = f"🟢 BTST Scan: {count} Opportunities Found"
        
        success = False
        
        if self.config.SEND_EMAIL:
            success = self.send_email(subject, report) or success
        
        if self.config.SEND_TELEGRAM:
            telegram_msg = self.format_for_telegram(report)
            success = self.send_telegram(telegram_msg) or success
        
        return success
    
    def notify_swing_results(self, report: str, count: int) -> bool:
        """
        Send swing scan results notification
        
        Args:
            report: Report text
            count: Number of candidates found
        
        Returns:
            True if notification sent
        """
        subject = f"🟠 Swing Trading Watchlist: {count} Stocks"
        
        success = False
        
        if self.config.SEND_EMAIL:
            success = self.send_email(subject, report) or success
        
        if self.config.SEND_TELEGRAM:
            telegram_msg = self.format_for_telegram(report)
            success = self.send_telegram(telegram_msg) or success
        
        return success
    
    def notify_error(self, error_msg: str) -> bool:
        """
        Send error notification
        
        Args:
            error_msg: Error message
        
        Returns:
            True if notification sent
        """
        subject = "❌ Stock Scanner Error"
        
        success = False
        
        if self.config.SEND_EMAIL:
            success = self.send_email(subject, error_msg) or success
        
        if self.config.SEND_TELEGRAM:
            success = self.send_telegram(f"<b>ERROR</b>\n{error_msg}") or success
        
        return success


def test_notifications():
    """Test notification system"""
    notifier = Notifier()
    
    test_message = """
🧪 TEST NOTIFICATION

This is a test message from your stock scanner.

If you're seeing this, notifications are working correctly!

✅ Email system: OK
✅ Telegram system: OK
    """
    
    print("Testing email...")
    notifier.send_email("Test: Stock Scanner", test_message)
    
    print("Testing Telegram...")
    notifier.send_telegram(notifier.format_for_telegram(test_message))
    
    print("Test complete!")


if __name__ == "__main__":
    test_notifications()
