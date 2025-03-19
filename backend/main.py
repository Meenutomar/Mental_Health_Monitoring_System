from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chatbot
import videostream

# Initialize FastAPI app
app = FastAPI()

# Allow requests from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chatbot.router)
app.include_router(videostream.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
