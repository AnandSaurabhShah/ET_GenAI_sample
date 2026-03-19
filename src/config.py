"""
Configuration settings for ET GenAI Hackathon
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "indicbert": {
        "model_name": "ai4bharat/indic-bert",
        "max_length": 512,
        "batch_size": 16
    },
    "llama3_8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    },
    "indictrans2": {
        "model_name": "ai4bharat/indictrans2-en-indic-1.0",
        "src_lang": "en",
        "tgt_lang": "hi"
    }
}

# Database configurations
DATABASE_CONFIGS = {
    "chroma": {
        "host": "localhost",
        "port": 8000,
        "collection_name": "enterprise_docs"
    },
    "postgresql": {
        "host": "localhost",
        "port": 5432,
        "database": "et_hackathon",
        "username": "postgres",
        "password": "password"
    },
    "immudb": {
        "host": "localhost",
        "port": 3322,
        "database": "audit_trail"
    }
}

# Security configurations
JWT_CONFIG = {
    "algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "secret_key": os.getenv("JWT_SECRET", "your-secret-key-here")
}

# Feature flags
FEATURES = {
    "code_mixed_interface": True,
    "gstin_reconciliation": True,
    "kyc_ner_extractor": True,
    "sla_breach_predictor": True,
    "ondc_router": True,
    "invoice_parser": True,
    "cryptographic_audit": True,
    "self_healing_engine": True,
    "meeting_intelligence": True,
    "enterprise_rag": True,
    "contract_analyzer": True,
    "workflow_observability": True,
    "invoice_validator": True,
    "access_control": True,
    "vendor_scorer": True,
    "state_checkpointing": True,
    "tax_planning_agent": True,
    "sentiment_analyzer": True,
    "temporal_triggers": True,
    "merkle_trees": True
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# API configurations
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "reload": True
}

# ML Model paths
MODEL_PATHS = {
    "indicbert": MODELS_DIR / "indicbert",
    "llama3": MODELS_DIR / "llama3_8b",
    "indictrans2": MODELS_DIR / "indictrans2",
    "ocr": MODELS_DIR / "ocr_models",
    "speech": MODELS_DIR / "speech_models"
}
