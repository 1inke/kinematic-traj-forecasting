"""Optional notifications: credentials are environment-only, never embedded."""
import os,smtplib
from email.message import EmailMessage

def send_email(subject,body):
    keys=['F16_SMTP_HOST','F16_SMTP_USERNAME','F16_SMTP_PASSWORD','F16_EMAIL_TO']
    if not all(os.environ.get(k) for k in keys):raise RuntimeError('Notification credentials are not configured')
    msg=EmailMessage();msg['Subject']=subject;msg['From']=os.environ['F16_SMTP_USERNAME'];msg['To']=os.environ['F16_EMAIL_TO'];msg.set_content(body)
    with smtplib.SMTP_SSL(os.environ['F16_SMTP_HOST'],int(os.environ.get('F16_SMTP_PORT','465')),timeout=20) as smtp:
        smtp.login(os.environ['F16_SMTP_USERNAME'],os.environ['F16_SMTP_PASSWORD']);smtp.send_message(msg)
