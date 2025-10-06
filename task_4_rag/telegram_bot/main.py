import os


from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Update
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler
from dotenv import load_dotenv
import uvicorn


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 18000))

# === Создаём бота и диспетчер ===
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# === FastAPI-приложение ===
app = FastAPI()


@dp.message(F.text)
async def echo_handler(message: types.Message):
    """Обрабатывает любое текстовое сообщение"""
    await message.answer(f"Ты написал: {message.text}")


@app.on_event("startup")
async def on_startup():
    print("✅ FastAPI и aiogram запущены.")


# === Webhook endpoint ===
@app.post(WEBHOOK_PATH)
async def webhook_handler(request: Request):
    data = await request.json()
    update = Update.model_validate(data)
    await dp.feed_update(bot, update)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
