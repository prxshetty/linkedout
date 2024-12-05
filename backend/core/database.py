from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings

class Database:
    client : AsyncIOMotorClient = None

async def get_database() -> AsyncIOMotorClient:
    return Database.client[settings.MONGODB_DB_NAME]

async def connect_to_mongodb():
    Database.client = AsyncIOMotorClient(settings.MONGODB_URL)
    print("Connected to MongoDB")

async def close_mongodb_connection():
    if Database.client is not None:
        Database.client.close()
        print("Closed MongoDB connection")