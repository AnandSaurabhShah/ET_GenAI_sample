"""
Features module for ET GenAI Hackathon
Contains all 20 SLM-based features
"""

# Try to import features, handle missing dependencies gracefully
try:
    from .code_mixed_interface import CodeMixedConversationalInterface
    CODE_MIXED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Code-mixed interface not available: {e}")
    CodeMixedConversationalInterface = None
    CODE_MIXED_AVAILABLE = False

try:
    from .gstin_reconciliation import GSTINReconciliationAgent
    GSTIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GSTIN reconciliation not available: {e}")
    GSTINReconciliationAgent = None
    GSTIN_AVAILABLE = False

try:
    from .kyc_ner_extractor import CorporateKYCNERExtractor
    KYC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: KYC NER extractor not available: {e}")
    CorporateKYCNERExtractor = None
    KYC_AVAILABLE = False

try:
    from .sla_breach_predictor import SLABreachPredictor
    SLA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SLA breach predictor not available: {e}")
    SLABreachPredictor = None
    SLA_AVAILABLE = False

try:
    from .ondc_router import ONDCRouter
    ONDC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ONDC router not available: {e}")
    ONDCRouter = None
    ONDC_AVAILABLE = False

try:
    from .invoice_parser import InvoiceParser
    INVOICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Invoice parser not available: {e}")
    InvoiceParser = None
    INVOICE_AVAILABLE = False

try:
    from .cryptographic_audit import CryptographicAuditTrail
    AUDIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cryptographic audit not available: {e}")
    CryptographicAuditTrail = None
    AUDIT_AVAILABLE = False

try:
    from .self_healing_engine import SelfHealingEngine
    HEALING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Self-healing engine not available: {e}")
    SelfHealingEngine = None
    HEALING_AVAILABLE = False

try:
    from .meeting_intelligence import MeetingIntelligence
    MEETING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Meeting intelligence not available: {e}")
    MeetingIntelligence = None
    MEETING_AVAILABLE = False

try:
    from .enterprise_rag import EnterpriseRAG
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enterprise RAG not available: {e}")
    EnterpriseRAG = None
    RAG_AVAILABLE = False

try:
    from .sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sentiment analyzer not available: {e}")
    SENTIMENT_AVAILABLE = False

    class SentimentAnalyzer:
        def __init__(self):
            pass

        def analyze_sentiment(self, text, threshold=0.7):
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "threshold": threshold,
                "above_threshold": False,
                "language": "auto",
                "model": "unavailable",
                "mode": "unavailable",
                "distribution": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
            }

# Placeholder classes for remaining features
class ContractAnalyzer:
    def __init__(self):
        pass

class WorkflowObservability:
    def __init__(self):
        pass

class InvoiceValidator:
    def __init__(self):
        pass

class AccessControl:
    def __init__(self):
        pass

class VendorPerformanceScorer:
    def __init__(self):
        pass

class StateCheckpointing:
    def __init__(self):
        pass

class TaxPlanningAgent:
    def __init__(self):
        pass

class TemporalTriggers:
    def __init__(self):
        pass

class MerkleTrees:
    def __init__(self):
        pass

__all__ = [
    "CodeMixedConversationalInterface",
    "GSTINReconciliationAgent", 
    "CorporateKYCNERExtractor",
    "SLABreachPredictor",
    "ONDCRouter",
    "InvoiceParser",
    "CryptographicAuditTrail",
    "SelfHealingEngine",
    "MeetingIntelligence",
    "EnterpriseRAG",
    "ContractAnalyzer",
    "WorkflowObservability",
    "InvoiceValidator",
    "AccessControl",
    "VendorPerformanceScorer",
    "StateCheckpointing",
    "TaxPlanningAgent",
    "SentimentAnalyzer",
    "TemporalTriggers",
    "MerkleTrees"
]
