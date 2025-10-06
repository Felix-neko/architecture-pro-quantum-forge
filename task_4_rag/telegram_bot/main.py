# Generated With ChatGPT
# Слегка доработано руками
import os
from typing import Dict, Any

from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Update, Message
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import uvicorn


# Загружаем переменные окружения из .env файла
# Это позволяет хранить чувствительные данные (токены, ключи) вне кода
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 18000))

# === Создаём бота и диспетчер ===
# Bot - объект для взаимодействия с Telegram API (отправка сообщений, получение данных)
bot = Bot(token=BOT_TOKEN)

# Dispatcher - маршрутизатор событий от Telegram (сообщения, колбэки, команды)
# MemoryStorage - хранилище состояний пользователей в оперативной памяти (нам сгодится: история нам не нужна)
dp = Dispatcher(storage=MemoryStorage())

# === FastAPI-приложение ===
app = FastAPI()

# @dp.message(F.text)
# async def echo_handler(message: types.Message):
#     """Обрабатывает любое текстовое сообщение"""
#     await message.answer(f"<b>Ты написал:</b> {message.text}", parse_mode="HTML")


@dp.message(F.text)
async def echo_handler(message: types.Message) -> None:
    """Обрабатывает любое текстовое сообщение"""
    await message.answer(f"<b>Ты написал:</b> {message.text}", parse_mode="HTML")


@app.on_event("startup")
async def on_startup():
    print("✅ FastAPI и aiogram запущены.")


# === Webhook endpoint ===
@app.post(WEBHOOK_PATH)
async def webhook_handler(update: Update) -> Dict[str, Any]:
    """Принимает обновления от Telegram через webhook и передаёт их в диспетчер"""

    # Передаём обновление в диспетчер aiogram для обработки зарегистрированными хендлерами
    await dp.feed_update(bot, update)

    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
