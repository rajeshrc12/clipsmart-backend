from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get the frontend URL from environment variables
frontend_url = os.getenv("FRONTEND_URL")

# Allow CORS from frontend URL (and development URLs)
origins = [
    frontend_url,  # Production URL from environment variable
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Hello World!"}
