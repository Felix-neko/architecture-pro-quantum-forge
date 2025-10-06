#!/usr/bin/env python3
"""
set_webhook.py

Примеры использования:

  # Установить webhook:
  python set_webhook.py https://a1b2c3d4.ngrok.io

  # Установить webhook с кастомным токеном и путём:
  python set_webhook.py https://a1b2c3d4.ngrok.io --token 12345:ABCDEF --path /webhook

  # Удалить webhook:
  python set_webhook.py --delete
"""

import os
import sys
import argparse
import httpx
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Установка или удаление Telegram webhook через ngrok URL.")
    p.add_argument(
        "ngrok_url", nargs="?", help="ngrok public URL (например, https://a1b2c3d4.ngrok.io). Не нужен при --delete."
    )
    p.add_argument("--token", "-t", help="Telegram BOT токен. Если не указан, берётся из .env (BOT_TOKEN).")
    p.add_argument(
        "--path",
        "-p",
        default=os.getenv("WEBHOOK_PATH", "/webhook"),
        help="Путь webhook (по умолчанию /webhook или значение из .env).",
    )
    p.add_argument("--delete", "-d", action="store_true", help="Удалить существующий webhook.")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout (сек).")
    return p.parse_args()


def ensure_https(url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL должен начинаться с http:// или https://")
    if url.startswith("http://"):
        raise ValueError("Telegram требует HTTPS для webhook URL.")
    return url.rstrip("/")


def main():
    args = parse_args()
    token = args.token or os.getenv("BOT_TOKEN")

    if not token:
        print("❌ Ошибка: не задан BOT_TOKEN. Передайте --token или добавьте в .env", file=sys.stderr)
        sys.exit(2)

    api_base = f"https://api.telegram.org/bot{token}"

    if args.delete:
        # Удаляем webhook
        print("🧹 Удаляем webhook...")
        try:
            r = httpx.post(f"{api_base}/deleteWebhook", timeout=args.timeout)
            r.raise_for_status()
            data = r.json()
            if data.get("ok"):
                print("✅ Webhook успешно удалён.")
            else:
                print("❌ Ошибка при удалении webhook:", data)
        except Exception as e:
            print("Ошибка при обращении к Telegram API:", e, file=sys.stderr)
            sys.exit(1)
        return

    # Устанавливаем webhook
    if not args.ngrok_url:
        print("❌ Не указан ngrok_url. Пример: python set_webhook.py https://a1b2c3d4.ngrok.io", file=sys.stderr)
        sys.exit(2)

    try:
        base = ensure_https(args.ngrok_url)
    except ValueError as e:
        print("Ошибка:", e, file=sys.stderr)
        sys.exit(2)

    path = args.path if args.path.startswith("/") else f"/{args.path}"
    webhook_url = f"{base}{path}"

    print(f"🔗 Устанавливаем webhook: {webhook_url}")
    try:
        r = httpx.post(f"{api_base}/setWebhook", params={"url": webhook_url}, timeout=args.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("ok"):
            print("✅ Webhook успешно установлен.")
        else:
            print("❌ Ошибка при установке webhook:", data)
    except Exception as e:
        print("Ошибка при установке webhook:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
