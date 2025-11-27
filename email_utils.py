import os
import smtplib
from email.message import EmailMessage
from typing import List, Optional


class EmailConfigError(Exception):
    """Raised when SMTP configuration is missing or invalid."""


def _get_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is not None:
        return value
    return fallback


def load_smtp_settings() -> dict:
    """
    Load SMTP configuration from environment variables.
    Expected vars:
      - SMTP_HOST (required)
      - SMTP_PORT (default 587)
      - SMTP_USER (required for authenticated servers)
      - SMTP_PASS (required for authenticated servers)
      - SMTP_SENDER or EMAIL_FROM (fallback)
      - SMTP_USE_TLS (default True)
    """
    host = _get_env("SMTP_HOST")
    port = int(_get_env("SMTP_PORT", "587"))
    user = _get_env("SMTP_USER")
    password = _get_env("SMTP_PASS")
    sender = _get_env("SMTP_SENDER") or _get_env("EMAIL_FROM") or user
    use_tls = (_get_env("SMTP_USE_TLS", "true").lower() != "false")

    if not host or not sender:
        raise EmailConfigError("SMTP_HOST and SMTP_SENDER (or EMAIL_FROM) must be set.")
    if user and not password:
        raise EmailConfigError("SMTP_PASS must be set when SMTP_USER is provided.")

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "sender": sender,
        "use_tls": use_tls,
    }


def send_summary_email(subject: str, body: str, recipients: List[str]) -> None:
    """
    Send the meeting summary via SMTP.
    Raises EmailConfigError for misconfiguration and smtplib.SMTPException for runtime errors.
    """
    if not recipients:
        raise ValueError("At least one recipient email is required.")

    settings = load_smtp_settings()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["sender"]
    msg["To"] = ", ".join(recipients)
    msg.set_content(body or "")

    with smtplib.SMTP(settings["host"], settings["port"]) as server:
        if settings["use_tls"]:
            server.starttls()
        if settings["user"]:
            server.login(settings["user"], settings["password"])
        server.send_message(msg)

