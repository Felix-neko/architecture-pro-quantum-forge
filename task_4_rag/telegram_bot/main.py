# Generated With ChatGPT
# Слегка доработано руками
import os
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Update, Message
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import uvicorn

# Импортируем RAG функции из rag_example.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_example import create_rag_chain, answer_question

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

# === Инициализация RAG chain при старте ===
rag_chain = None


@dp.message(F.text)
async def rag_handler(message: types.Message) -> None:
    """Обрабатывает текстовые сообщения и отвечает через RAG"""
    global rag_chain

    if not rag_chain:
        await message.answer("⚠️ RAG chain не инициализирован. Попробуйте позже.")
        return

    user_query = message.text

    # Отправить статус "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # Получить ответ через RAG
    result = answer_question(rag_chain, user_query)

    # Форматировать ответ
    response_parts = [f"<b>Запрос:</b> {user_query}"]

    if result.get("think"):
        response_parts.append(f"\n<b>Thinking:</b>\n{result['think']}")

    final_answer = result.get("final_answer", "Нет ответа")
    response_parts.append(f"\n<b>Ответ:</b> {final_answer}")

    response_text = "\n".join(response_parts)

    # Отправить ответ пользователю
    await message.answer(response_text, parse_mode="HTML")


@app.on_event("startup")
async def on_startup():
    global rag_chain
    print("🔧 Инициализация RAG chain...")
    rag_chain = create_rag_chain()
    print("RAG chain ON!")


# === Webhook endpoint ===
@app.post(WEBHOOK_PATH)
async def webhook_handler(update: Update) -> Dict[str, Any]:
    """Принимает обновления от Telegram через webhook и передаёт их в диспетчер"""

    # Передаём обновление в диспетчер aiogram для обработки зарегистрированными хендлерами
    await dp.feed_update(bot, update)

    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
