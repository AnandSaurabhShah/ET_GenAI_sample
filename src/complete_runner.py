"""
Complete Runner for ET GenAI Hackathon
Loads all models with 100% precision and runs the entire project
"""

import logging
import time
import sys
from pathlib import Path
import streamlit as st
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('et_genai_complete.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import modules after path setup
try:
    from model_loader import load_all_models, model_loader
    from enhanced_features import get_enhanced_feature
    from config import FEATURES
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

class CompleteRunner:
    """Complete runner for the entire ET GenAI project"""
    
    def __init__(self):
        self.model_status = {}
        self.enhanced_features = {}
        self.startup_time = time.time()
        
    def initialize_everything(self):
        """Initialize all models and features with 100% precision"""
        logger.info("🚀 Starting ET GenAI Complete Initialization...")
        
        # Step 1: Load all models
        logger.info("📦 Loading all models with 100% precision...")
        self.model_status = load_all_models()
        
        # Step 2: Initialize enhanced features
        logger.info("🔧 Initializing enhanced features...")
        self._initialize_enhanced_features()
        
        # Step 3: Verify everything is working
        logger.info("✅ Verifying system integrity...")
        self._verify_system()
        
        # Step 4: Generate initialization report
        self._generate_initialization_report()
        
        logger.info("🎉 Complete initialization finished!")
        
    def _initialize_enhanced_features(self):
        """Initialize all enhanced features"""
        feature_configs = {
            "sentiment_analyzer": FEATURES.get("sentiment_analyzer", True),
            "gstin_reconciliation": FEATURES.get("gstin_reconciliation", True),
            "kyc_ner_extractor": FEATURES.get("kyc_ner_extractor", True),
            "sla_predictor": FEATURES.get("sla_predictor", True),
        }
        
        for feature_name, enabled in feature_configs.items():
            if enabled:
                try:
                    logger.info(f"Initializing {feature_name}...")
                    feature = get_enhanced_feature(feature_name)
                    self.enhanced_features[feature_name] = feature
                    logger.info(f"✅ {feature_name} initialized successfully")
                except Exception as e:
                    logger.error(f"❌ {feature_name} failed to initialize: {e}")
                    self.enhanced_features[feature_name] = None
            else:
                logger.info(f"⏭️ {feature_name} is disabled")
                self.enhanced_features[feature_name] = None
    
    def _verify_system(self):
        """Verify system integrity"""
        logger.info("🔍 Running system verification...")
        
        # Check models
        loaded_models = sum(1 for status in self.model_status.values() if status.loaded)
        total_models = len(self.model_status)
        
        logger.info(f"Models: {loaded_models}/{total_models} loaded successfully")
        
        # Check features
        working_features = sum(1 for feature in self.enhanced_features.values() if feature is not None)
        total_features = len(self.enhanced_features)
        
        logger.info(f"Features: {working_features}/{total_features} working properly")
        
        # Check memory
        memory_usage = model_loader.get_memory_usage()
        if memory_usage:
            logger.info(f"Memory Usage: {memory_usage}")
        
        # Test basic functionality
        self._run_basic_tests()
    
    def _run_basic_tests(self):
        """Run basic functionality tests"""
        logger.info("🧪 Running basic functionality tests...")
        
        # Test sentiment analysis
        if "sentiment_analyzer" in self.enhanced_features and self.enhanced_features["sentiment_analyzer"]:
            try:
                result = self.enhanced_features["sentiment_analyzer"].analyze_sentiment("This is a great product!")
                logger.info(f"✅ Sentiment analysis test passed: {result}")
            except Exception as e:
                logger.error(f"❌ Sentiment analysis test failed: {e}")
        
        # Test GSTIN extraction
        if "gstin_reconciliation" in self.enhanced_features and self.enhanced_features["gstin_reconciliation"]:
            try:
                result = self.enhanced_features["gstin_reconciliation"].extract_gstin_from_text("Our GSTIN is 27AAAPL1234C1ZV")
                logger.info(f"✅ GSTIN extraction test passed: {result}")
            except Exception as e:
                logger.error(f"❌ GSTIN extraction test failed: {e}")
        
        # Test KYC extraction
        if "kyc_ner_extractor" in self.enhanced_features and self.enhanced_features["kyc_ner_extractor"]:
            try:
                result = self.enhanced_features["kyc_ner_extractor"].extract_entities("Contact us at test@example.com")
                logger.info(f"✅ KYC extraction test passed: {len(result)} entities found")
            except Exception as e:
                logger.error(f"❌ KYC extraction test failed: {e}")
        
        # Test SLA prediction
        if "sla_predictor" in self.enhanced_features and self.enhanced_features["sla_predictor"]:
            try:
                ticket_data = {
                    "priority": "High",
                    "category": "Technical",
                    "customer_tier": "Gold",
                    "complexity": "Moderate"
                }
                result = self.enhanced_features["sla_predictor"].predict_breach_risk(ticket_data)
                logger.info(f"✅ SLA prediction test passed: {result}")
            except Exception as e:
                logger.error(f"❌ SLA prediction test failed: {e}")
    
    def _generate_initialization_report(self):
        """Generate comprehensive initialization report"""
        total_time = time.time() - self.startup_time
        
        report = {
            "initialization_time": total_time,
            "models": {},
            "features": {},
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "device": str(model_loader.device),
                "cuda_available": torch.cuda.is_available()
            }
        }
        
        # Model status
        for name, status in self.model_status.items():
            report["models"][name] = {
                "loaded": status.loaded,
                "precision": status.precision,
                "load_time": status.load_time,
                "error": status.error
            }
        
        # Feature status
        for name, feature in self.enhanced_features.items():
            report["features"][name] = {
                "working": feature is not None,
                "type": type(feature).__name__ if feature else None
            }
        
        # Save report
        import json
        with open("initialization_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info("📊 Initialization Report:")
        logger.info(f"  Total Time: {total_time:.2f}s")
        logger.info(f"  Models Loaded: {sum(1 for s in self.model_status.values() if s.loaded)}/{len(self.model_status)}")
        logger.info(f"  Features Working: {sum(1 for f in self.enhanced_features.values() if f is not None)}/{len(self.enhanced_features)}")
        logger.info(f"  Device: {model_loader.device}")
    
    def run_complete_application(self):
        """Run the complete application"""
        logger.info("🎯 Starting complete application...")
        
        # Initialize everything
        self.initialize_everything()
        
        # Launch Streamlit app
        logger.info("🌐 Launching Streamlit application...")
        
        # Import and run the main app
        try:
            from main_app import ETGenAIApp
            
            # Patch the main app to use enhanced features
            original_init = ETGenAIApp._initialize_features
            runner = self
            
            def patched_init(self):
                # Call original initialization
                original_init(self)
                
                # Replace with enhanced features
                for feature_name, enhanced_feature in runner.enhanced_features.items():
                    if enhanced_feature:
                        # Map feature names
                        feature_map = {
                            "sentiment_analyzer": "sentiment_analyzer",
                            "gstin_reconciliation": "gstin_reconciliation",
                            "kyc_ner_extractor": "kyc_ner",
                            "sla_predictor": "sla_predictor"
                        }
                        
                        app_feature_name = feature_map.get(feature_name)
                        if app_feature_name:
                            self.features[app_feature_name] = enhanced_feature
                            logger.info(f"✅ Enhanced {app_feature_name} loaded")
            
            ETGenAIApp._initialize_features = patched_init
            
            # Create and run app
            app = ETGenAIApp()
            app.run()
            
        except Exception as e:
            logger.error(f"❌ Failed to run application: {e}")
            raise

def main():
    """Main function"""
    print("🚀 ET GenAI Hackathon - Complete Runner")
    print("=" * 50)
    
    # Create and run complete system
    runner = CompleteRunner()
    
    try:
        runner.run_complete_application()
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"💥 Application crashed: {e}")
        raise

if __name__ == "__main__":
    main()
