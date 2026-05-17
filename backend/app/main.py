from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, datasets, exports, health, inference, models, pretrain, training
from app.config.settings import settings
from app.database.session import Base, engine


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Create all tables on startup (migration-safe: only adds missing tables)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(datasets.router, prefix=settings.api_prefix)
app.include_router(training.router, prefix=settings.api_prefix)
app.include_router(pretrain.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
app.include_router(exports.router, prefix=settings.api_prefix)
app.include_router(inference.router, prefix=settings.api_prefix)
