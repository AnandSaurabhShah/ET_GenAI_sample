"""
Main Application for ET GenAI Hackathon
Comprehensive SLM-based Enterprise Automation Suite
"""

import logging
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from features import *
from config import FEATURES, API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETGenAIApp:
    """Main application class for ET GenAI Hackathon"""
    
    def __init__(self):
        self.title = "🚀 ET GenAI Hackathon - SLM Enterprise Suite"
        self.subtitle = "20 Small Language Models for Complete Enterprise Automation"
        
        # Initialize features
        self.features = {}
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize all features"""
        feature_classes = {
            "code_mixed": CodeMixedConversationalInterface,
            "gstin_reconciliation": GSTINReconciliationAgent,
            "kyc_ner": CorporateKYCNERExtractor,
            "sla_predictor": SLABreachPredictor,
            "ondc_router": ONDCRouter,
            "invoice_parser": InvoiceParser,
            "cryptographic_audit": CryptographicAuditTrail,
            "self_healing": SelfHealingEngine,
            "meeting_intelligence": MeetingIntelligence,
            "enterprise_rag": EnterpriseRAG,
            "contract_analyzer": ContractAnalyzer,
            "workflow_observability": WorkflowObservability,
            "invoice_validator": InvoiceValidator,
            "access_control": AccessControl,
            "vendor_scorer": VendorPerformanceScorer,
            "state_checkpointing": StateCheckpointing,
            "tax_planning": TaxPlanningAgent,
            "sentiment_analyzer": SentimentAnalyzer,
            "temporal_triggers": TemporalTriggers,
            "merkle_trees": MerkleTrees
        }

        feature_flag_map = {
            "code_mixed": "code_mixed_interface",
            "gstin_reconciliation": "gstin_reconciliation",
            "kyc_ner": "kyc_ner_extractor",
            "sla_predictor": "sla_breach_predictor",
            "ondc_router": "ondc_router",
            "invoice_parser": "invoice_parser",
            "cryptographic_audit": "cryptographic_audit",
            "self_healing": "self_healing_engine",
            "meeting_intelligence": "meeting_intelligence",
            "enterprise_rag": "enterprise_rag",
            "contract_analyzer": "contract_analyzer",
            "workflow_observability": "workflow_observability",
            "invoice_validator": "invoice_validator",
            "access_control": "access_control",
            "vendor_scorer": "vendor_scorer",
            "state_checkpointing": "state_checkpointing",
            "tax_planning": "tax_planning_agent",
            "sentiment_analyzer": "sentiment_analyzer",
            "temporal_triggers": "temporal_triggers",
            "merkle_trees": "merkle_trees",
        }
        
        for feature_name, feature_class in feature_classes.items():
            config_key = feature_flag_map.get(feature_name, feature_name)
            if FEATURES.get(config_key, False):
                try:
                    self.features[feature_name] = feature_class()
                    logger.info(f"Initialized feature: {feature_name}")
                except Exception as e:
                    logger.error(f"Error initializing {feature_name}: {e}")
                    self.features[feature_name] = None
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="ET GenAI Hackathon",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🚀 ET GenAI Hackathon</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">20 Small Language Models for Complete Enterprise Automation</p>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_sidebar(self):
        """Render sidebar"""
        st.sidebar.title("🎯 Features")
        
        # Feature categories
        feature_categories = {
            "🗣️ Language & Speech": ["code_mixed", "meeting_intelligence", "sentiment_analyzer"],
            "📄 Document Processing": ["kyc_ner", "invoice_parser", "contract_analyzer", "invoice_validator"],
            "🤖 AI & ML Models": ["sla_predictor", "vendor_scorer", "tax_planning"],
            "🔐 Security & Audit": ["cryptographic_audit", "access_control", "merkle_trees"],
            "🔄 Workflow & Automation": ["self_healing", "workflow_observability", "state_checkpointing", "temporal_triggers"],
            "🌐 Integration & Routing": ["gstin_reconciliation", "ondc_router", "enterprise_rag"]
        }
        
        selected_category = st.sidebar.selectbox("Select Category", list(feature_categories.keys()))
        
        # Features in selected category
        category_features = feature_categories[selected_category]
        feature_names = []
        feature_descriptions = {
            "code_mixed": "Code-Mixed Conversational Interface",
            "meeting_intelligence": "Autonomous Meeting Intelligence",
            "sentiment_analyzer": "Cross-Lingual Sentiment Analyzer",
            "kyc_ner": "Corporate KYC NER Extractor",
            "invoice_parser": "Indic Vision-Language Invoice Parser",
            "contract_analyzer": "Multilingual Contract Analyzer",
            "invoice_validator": "Smart e-Invoice Predictive Validator",
            "sla_predictor": "Agentic SLA Breach Predictor",
            "vendor_scorer": "Vendor Performance Scorer",
            "tax_planning": "Tax-Context Action Planning Agent",
            "cryptographic_audit": "Zero-Cost Cryptographic Audit Trail",
            "access_control": "Zero-Trust Autonomous Access Control",
            "merkle_trees": "Cryptographic Python Merkle Trees",
            "self_healing": "Self-Healing Execution Engine",
            "workflow_observability": "Local Agentic Workflow Observability",
            "state_checkpointing": "Local System State Checkpointing",
            "temporal_triggers": "Temporal Logic Trigger",
            "gstin_reconciliation": "Autonomous GSTIN Reconciliation Agent",
            "ondc_router": "Multi-Agent ONDC Router",
            "enterprise_rag": "Local Enterprise RAG Memory"
        }
        
        for feature in category_features:
            if feature in feature_descriptions:
                status = "✅" if self.features.get(feature) else "❌"
                feature_names.append(f"{status} {feature_descriptions[feature]}")
        
        selected_feature_name = st.sidebar.selectbox("Select Feature", feature_names)
        
        # Extract feature key
        selected_feature = None
        for feature in category_features:
            if feature_descriptions.get(feature, "") in selected_feature_name:
                selected_feature = feature
                break
        
        st.session_state.selected_feature = selected_feature
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 System Status")
        
        active_features = sum(1 for f in self.features.values() if f is not None)
        total_features = len(self.features)
        
        st.sidebar.metric("Active Features", f"{active_features}/{total_features}")
        st.sidebar.metric("Success Rate", f"{(active_features/total_features)*100:.1f}%")
    
    def _render_main_content(self):
        """Render main content area"""
        selected_feature = st.session_state.get("selected_feature")
        
        if not selected_feature:
            self._render_dashboard()
        else:
            self._render_feature_interface(selected_feature)
    
    def _render_dashboard(self):
        """Render main dashboard"""
        st.header("🎯 Welcome to ET GenAI Hackathon Suite")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>20</h3>
                <p>SLM Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            active_features = sum(1 for f in self.features.values() if f is not None)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{active_features}</h3>
                <p>Active Models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>100%</h3>
                <p>Local Processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>0</h3>
                <p>API Dependencies</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature overview
        st.markdown("---")
        st.subheader("🚀 Feature Categories")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🗣️ Language & Speech</h4>
                <ul>
                    <li>Code-Mixed Conversational Interface</li>
                    <li>Autonomous Meeting Intelligence</li>
                    <li>Cross-Lingual Sentiment Analyzer</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📄 Document Processing</h4>
                <ul>
                    <li>Corporate KYC NER Extractor</li>
                    <li>Indic Vision-Language Invoice Parser</li>
                    <li>Multilingual Contract Analyzer</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>🤖 AI & ML Models</h4>
                <ul>
                    <li>Agentic SLA Breach Predictor</li>
                    <li>Vendor Performance Scorer</li>
                    <li>Tax-Context Action Planning Agent</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick demo
        st.markdown("---")
        st.subheader("⚡ Quick Demo")
        
        demo_text = st.text_area("Try our features instantly:", 
                               placeholder="Enter text for sentiment analysis, GSTIN reconciliation, or document processing...")
        
        if st.button("🚀 Process"):
            if demo_text:
                with st.spinner("Processing..."):
                    # Demo sentiment analysis
                    if "sentiment_analyzer" in self.features and self.features["sentiment_analyzer"]:
                        try:
                            result = self.features["sentiment_analyzer"].analyze_sentiment(demo_text)
                            st.success(f"Sentiment: {result.get('sentiment', 'Unknown')} (Confidence: {result.get('confidence', 0):.2f})")
                        except:
                            st.warning("Sentiment analysis temporarily unavailable")
                    
                    # Demo GSTIN extraction
                    if "gstin_reconciliation" in self.features and self.features["gstin_reconciliation"]:
                        try:
                            gstins = self.features["gstin_reconciliation"].extract_gstin_from_text(demo_text)
                            if gstins:
                                st.info(f"GSTINs found: {', '.join(gstins)}")
                            else:
                                st.info("No GSTINs found in text")
                        except:
                            st.warning("GSTIN reconciliation temporarily unavailable")
    
    def _render_feature_interface(self, feature_name: str):
        """Render specific feature interface"""
        feature = self.features.get(feature_name)
        
        if not feature:
            st.error(f"Feature {feature_name} is not available")
            return
        
        # Feature header
        feature_titles = {
            "code_mixed": "🗣️ Code-Mixed Conversational Interface",
            "meeting_intelligence": "🎙️ Autonomous Meeting Intelligence",
            "sentiment_analyzer": "📊 Cross-Lingual Sentiment Analyzer",
            "kyc_ner": "📋 Corporate KYC NER Extractor",
            "invoice_parser": "🧾 Indic Vision-Language Invoice Parser",
            "contract_analyzer": "📜 Multilingual Contract Analyzer",
            "invoice_validator": "✅ Smart e-Invoice Predictive Validator",
            "sla_predictor": "📈 Agentic SLA Breach Predictor",
            "vendor_scorer": "⭐ Vendor Performance Scorer",
            "tax_planning": "💰 Tax-Context Action Planning Agent",
            "cryptographic_audit": "🔒 Zero-Cost Cryptographic Audit Trail",
            "access_control": "🔐 Zero-Trust Autonomous Access Control",
            "merkle_trees": "🌳 Cryptographic Python Merkle Trees",
            "self_healing": "🔄 Self-Healing Execution Engine",
            "workflow_observability": "👁️ Local Agentic Workflow Observability",
            "state_checkpointing": "💾 Local System State Checkpointing",
            "temporal_triggers": "⏰ Temporal Logic Trigger",
            "gstin_reconciliation": "🏷️ Autonomous GSTIN Reconciliation Agent",
            "ondc_router": "🌐 Multi-Agent ONDC Router",
            "enterprise_rag": "📚 Local Enterprise RAG Memory"
        }
        
        st.header(feature_titles.get(feature_name, feature_name))
        
        # Render feature-specific interface
        if feature_name == "sentiment_analyzer":
            self._render_sentiment_analyzer_interface(feature)
        elif feature_name == "gstin_reconciliation":
            self._render_gstin_reconciliation_interface(feature)
        elif feature_name == "kyc_ner":
            self._render_kyc_ner_interface(feature)
        elif feature_name == "sla_predictor":
            self._render_sla_predictor_interface(feature)
        elif feature_name == "enterprise_rag":
            self._render_enterprise_rag_interface(feature)
        else:
            self._render_generic_feature_interface(feature, feature_name)
    
    def _render_sentiment_analyzer_interface(self, feature):
        """Render sentiment analyzer interface"""
        st.markdown("Analyze sentiment in English and Indian languages")
        
        text_input = st.text_area("Enter text for sentiment analysis:", 
                                 placeholder="Enter text in English, Hindi, Hinglish, or other Indian languages...")
        
        if st.button("Analyze Sentiment"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        result = feature.analyze_sentiment(text_input)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment", result.get('sentiment', 'Unknown'))
                        
                        with col2:
                            st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                        
                        with col3:
                            st.metric("Language", result.get('language', 'Unknown'))
                        
                        if result.get('emotions'):
                            st.subheader("Emotion Analysis")
                            for emotion, score in result['emotions'].items():
                                st.progress(score, text=f"{emotion}: {score:.2f}")
                    
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {e}")
    
    def _render_gstin_reconciliation_interface(self, feature):
        """Render GSTIN reconciliation interface"""
        st.markdown("Extract and validate GSTIN numbers from unstructured text")
        
        text_input = st.text_area("Enter supplier or business text:", 
                                 placeholder="Paste supplier information, invoice details, or business descriptions...")
        
        if st.button("Extract GSTINs"):
            if text_input:
                with st.spinner("Extracting GSTINs..."):
                    try:
                        result = feature.intelligent_gstin_extraction(text_input)
                        
                        st.subheader("Extracted Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Validated GSTINs:**")
                            for gstin in result.get('validated_gstins', []):
                                st.success(gstin)
                        
                        with col2:
                            st.write("**Extracted Data:**")
                            st.json(result.get('extracted_data', {}))
                        
                        if result.get('validated_gstins'):
                            if st.button("Reconcile with Database"):
                                reconciliation_result = feature.reconcile_supplier_data({
                                    'company_name': result.get('extracted_data', {}).get('company_names', [''])[0],
                                    'description': text_input
                                })
                                
                                st.subheader("Reconciliation Results")
                                for rec in reconciliation_result.get('reconciliation_results', []):
                                    st.write(f"✅ {rec}")
                    
                    except Exception as e:
                        st.error(f"Error extracting GSTINs: {e}")
    
    def _render_kyc_ner_interface(self, feature):
        """Render KYC NER extractor interface"""
        st.markdown("Extract corporate identity data from documents")
        
        text_input = st.text_area("Enter KYC document text:", 
                                 placeholder="Paste company registration documents, PAN details, or corporate information...")
        
        if st.button("Extract Entities"):
            if text_input:
                with st.spinner("Extracting entities..."):
                    try:
                        result = feature.process_document(text_input)
                        
                        st.subheader("Extracted Entities")
                        
                        # Group by entity type
                        grouped = result.get('grouped_entities', {})
                        
                        for entity_type, entities in grouped.items():
                            if entities:
                                st.write(f"**{entity_type}:**")
                                for entity in entities:
                                    st.write(f"• {entity['text']} (Confidence: {entity['confidence']:.2f})")
                        
                        # Summary
                        st.subheader("Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Entities", result.get('total_entities', 0))
                        
                        with col2:
                            summary = result.get('summary', {})
                            st.metric("Company Identified", "✅" if summary.get('company_identified') else "❌")
                        
                        with col3:
                            tax_ids = summary.get('tax_ids', {})
                            total_tax_ids = sum(tax_ids.values())
                            st.metric("Tax IDs Found", total_tax_ids)
                    
                    except Exception as e:
                        st.error(f"Error extracting entities: {e}")
    
    def _render_sla_predictor_interface(self, feature):
        """Render SLA breach predictor interface"""
        st.markdown("Predict SLA breaches and get recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            category = st.selectbox("Category", ["Technical", "Billing", "Account", "Network", "Security"])
        
        with col2:
            customer_tier = st.selectbox("Customer Tier", ["Bronze", "Silver", "Gold", "Platinum"])
            complexity = st.selectbox("Complexity", ["Simple", "Moderate", "Complex", "Very Complex"])
        
        ticket_data = {
            "priority": priority,
            "category": category,
            "customer_tier": customer_tier,
            "complexity": complexity,
            "created_hour": 10,
            "is_weekend": 0,
            "agent_experience": 2.5,
            "customer_history": 25
        }
        
        if st.button("Predict SLA Breach Risk"):
            with st.spinner("Analyzing risk..."):
                try:
                    result = feature.predict_breach_risk(ticket_data)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Breach Probability", f"{result.get('breach_probability', 0):.2%}")
                    
                    with col2:
                        st.metric("Risk Level", result.get('risk_level', 'Unknown'))
                    
                    with col3:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                    
                    if result.get('recommendations'):
                        st.subheader("Recommendations")
                        for rec in result['recommendations']:
                            st.write(f"• {rec}")
                
                except Exception as e:
                    st.error(f"Error predicting risk: {e}")
    
    def _render_enterprise_rag_interface(self, feature):
        """Render enterprise RAG interface"""
        st.markdown("Search enterprise knowledge base")
        
        # Add document section
        with st.expander("Add Document"):
            title = st.text_input("Document Title")
            content = st.text_area("Document Content")
            category = st.selectbox("Category", ["Policy", "Procedure", "Technical", "Legal", "General"])
            
            if st.button("Add Document"):
                if title and content:
                    doc_id = feature.add_document(title, content, category)
                    st.success(f"Document added: {doc_id[:8]}...")
        
        # Search section
        st.subheader("Search Documents")
        query = st.text_input("Enter your query:")
        search_category = st.selectbox("Category (optional)", ["All"] + feature.get_categories())
        
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    try:
                        category_filter = None if search_category == "All" else search_category
                        results = feature.search(query, top_k=5, category=category_filter)
                        
                        if results:
                            for i, result in enumerate(results, 1):
                                with st.expander(f"{i}. {result.document.title} (Score: {result.similarity_score:.3f})"):
                                    st.write(f"**Category:** {result.document.category}")
                                    st.write(f"**Author:** {result.document.author}")
                                    st.write(f"**Answer:** {result.answer}")
                                    st.write(f"**Relevant Chunks:**")
                                    for chunk in result.relevant_chunks:
                                        st.write(f"• {chunk[:200]}...")
                        else:
                            st.info("No results found")
                    
                    except Exception as e:
                        st.error(f"Error searching: {e}")
    
    def _render_generic_feature_interface(self, feature, feature_name):
        """Render generic interface for other features"""
        st.markdown(f"Interface for {feature_name}")
        st.info("This feature is available but requires specific implementation details.")
        
        # Show feature info
        if hasattr(feature, '__class__'):
            st.write(f"**Class:** {feature.__class__.__name__}")
            st.write(f"**Module:** {feature.__class__.__module__}")
        
        # Show available methods
        methods = [method for method in dir(feature) if not method.startswith('_') and callable(getattr(feature, method))]
        if methods:
            st.write("**Available Methods:**")
            for method in methods[:10]:  # Show first 10 methods
                st.write(f"• {method}()")

def main():
    """Main function"""
    app = ETGenAIApp()
    app.run()

if __name__ == "__main__":
    main()
