# -*- coding: utf-8 -*-

import os
import glob
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta


def get_today_cn_str() -> str:
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d")


def build_email_subject() -> str:
    prefix = os.getenv("MAIL_SUBJECT_PREFIX", "A股 3 Bar Play 扫描结果").strip()
    return f"{prefix} {get_today_cn_str()}"


def build_email_body(files):
    lines = []
    lines.append(f"您好，附件是 {get_today_cn_str()} 的 3 Bar Play 扫描结果。")
    lines.append("")
    if files:
        lines.append("本次附带文件：")
        for f in files:
            lines.append(f"- {os.path.basename(f)}")
    else:
        lines.append("本次 output 目录下没有找到可发送的 csv 文件。")
    lines.append("")
    lines.append("此邮件由 GitHub Actions 自动发送。")
    return "\n".join(lines)


def main():
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    mail_to = os.getenv("MAIL_TO", "").strip()
    mail_from = os.getenv("MAIL_FROM", smtp_user).strip()

    if not smtp_host:
        raise ValueError("缺少 SMTP_HOST")
    if not smtp_user:
        raise ValueError("缺少 SMTP_USER")
    if not smtp_pass:
        raise ValueError("缺少 SMTP_PASS")
    if not mail_to:
        raise ValueError("缺少 MAIL_TO")

    files = sorted(glob.glob("output/*.csv"))

    msg = EmailMessage()
    msg["Subject"] = build_email_subject()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(build_email_body(files))

    for file_path in files:
        with open(file_path, "rb") as f:
            data = f.read()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=os.path.basename(file_path),
        )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

    print(f"邮件发送成功，收件人: {mail_to}")
    print(f"附件数量: {len(files)}")


if __name__ == "__main__":
    main()