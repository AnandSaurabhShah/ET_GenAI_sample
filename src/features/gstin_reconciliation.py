"""
2. Autonomous GSTIN Reconciliation Agent
Deploy locally hosted SLM like Llama 3 8B for fuzzy-matching on unstructured supplier descriptions
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class GSTINReconciliationAgent:
    """
    Autonomous GSTIN Reconciliation Agent using local SLM
    Performs fuzzy-matching on unstructured supplier descriptions
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", use_llm: bool = False):
        self.model_name = model_name
        self.use_llm = use_llm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            lowercase=True,
            min_df=1
        )
        
        # GSTIN database (mock data)
        self.gstin_database = self._initialize_gstin_database()
        
        # Default to the fast local reconciliation path unless the caller explicitly wants the LLM.
        if self.use_llm:
            self._load_models()
        
        logger.info("GSTIN Reconciliation Agent initialized")
    
    def _load_models(self):
        """Load Llama 3 model for intelligent matching"""
        try:
            # For demo purposes, we'll use a smaller model
            # In production, you would load the actual Llama 3 8B model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium"
            ).to(self.device)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model_loaded = True
            
            logger.info("SLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SLM model: {e}")
            # Fallback to rule-based processing
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback processing"""
        logger.warning("Using fallback processing for GSTIN reconciliation")
        self.fallback_mode = True
    
    def _initialize_gstin_database(self) -> pd.DataFrame:
        """Initialize mock GSTIN database"""
        data = {
            'gstin': [
                '27AAAPL1234C1ZV', '27AAAPL5678B2ZY', '27AAAPL9012C3ZX',
                '27AAACR1234D1ZV', '27AAACR5678E2ZY', '27AAACR9012F3ZX',
                '27AABCS1234G1ZV', '27AABCS5678H2ZY', '27AABCS9012I3ZX'
            ],
            'business_name': [
                'ABC Technologies Pvt Ltd', 'XYZ Solutions Ltd', 'PQR Industries',
                'Global Services Inc', 'Local Traders Pvt Ltd', 'National Distributors',
                'Tech Innovations Ltd', 'Digital Solutions Pvt Ltd', 'Smart Systems Inc'
            ],
            'address': [
                'Mumbai, Maharashtra', 'Delhi, Delhi', 'Bangalore, Karnataka',
                'Chennai, Tamil Nadu', 'Kolkata, West Bengal', 'Hyderabad, Telangana',
                'Pune, Maharashtra', 'Gurgaon, Haryana', 'Noida, Uttar Pradesh'
            ],
            'status': ['Active'] * 9
        }
        return pd.DataFrame(data)
    
    def validate_gstin_format(self, gstin: str) -> bool:
        """
        Validate GSTIN format
        """
        # Remove spaces and convert to uppercase
        gstin = gstin.replace(" ", "").upper()
        
        # Check length
        if len(gstin) != 15:
            return False
        
        # Check pattern: 2 digits + 10 chars + 1 digit + 1 char + 1 digit
        pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$'
        return bool(re.match(pattern, gstin))
    
    def extract_gstin_from_text(self, text: str) -> List[str]:
        """
        Extract GSTIN numbers from unstructured text
        """
        # GSTIN pattern
        gstin_pattern = r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}\b'
        
        # Find all matches
        matches = re.findall(gstin_pattern, text.upper())
        
        # Validate each match
        valid_gstins = []
        for match in matches:
            if self.validate_gstin_format(match):
                valid_gstins.append(match)
        
        return valid_gstins
    
    def fuzzy_match_company_name(self, company_name: str, threshold: float = 0.7) -> List[Dict]:
        """
        Fuzzy match company name against database
        """
        matches = []
        
        for _, row in self.gstin_database.iterrows():
            db_name = row['business_name'].lower().strip()
            input_name = company_name.lower().strip()
            
            # Calculate similarity scores
            sequence_similarity = SequenceMatcher(None, db_name, input_name).ratio()
            
            # Character-based similarity
            char_similarity = self._calculate_char_similarity(db_name, input_name)
            
            # Combined similarity
            combined_similarity = (sequence_similarity + char_similarity) / 2
            
            if combined_similarity >= threshold:
                matches.append({
                    'gstin': row['gstin'],
                    'business_name': row['business_name'],
                    'address': row['address'],
                    'similarity_score': combined_similarity,
                    'sequence_similarity': sequence_similarity,
                    'char_similarity': char_similarity
                })
        
        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return matches
    
    def _calculate_char_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate character-level similarity using TF-IDF
        """
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([str1, str2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating character similarity: {e}")
            return 0.0
    
    def intelligent_gstin_extraction(self, text: str) -> Dict:
        """
        Use SLM for intelligent GSTIN extraction and validation
        """
        try:
            if hasattr(self, 'fallback_mode') or not self.model_loaded:
                return self._fallback_extraction(text)
            
            # Prepare prompt for SLM
            prompt = f"""
            Extract GSTIN information from the following text:
            
            Text: {text}
            
            Please extract:
            1. Any GSTIN numbers mentioned
            2. Company/business names
            3. Address information
            4. Contact information
            
            Respond in JSON format:
            {{
                "gstins": ["list of gstins"],
                "company_names": ["list of company names"],
                "addresses": ["list of addresses"],
                "contacts": ["list of contacts"]
            }}
            """
            
            # Generate response using SLM
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=500,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response.split('{')[-1].split('}')[0] + '}')
            except:
                # Fallback to regex extraction
                extracted_data = self._regex_extraction(text)
            
            # Validate extracted GSTINs
            validated_gstins = []
            for gstin in extracted_data.get('gstins', []):
                if self.validate_gstin_format(gstin):
                    validated_gstins.append(gstin)
            
            return {
                "extracted_data": extracted_data,
                "validated_gstins": validated_gstins,
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent GSTIN extraction: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict:
        """Fallback extraction using regex"""
        return self._regex_extraction(text)
    
    def _regex_extraction(self, text: str) -> Dict:
        """Extract information using regex patterns"""
        # GSTIN extraction
        gstins = self.extract_gstin_from_text(text)
        
        # Company name patterns
        company_patterns = [
            r'([A-Z][a-z]+\s+(?:Technologies|Solutions|Industries|Services|Traders|Distributors|Innovations|Systems)\s+(?:Pvt\s+Ltd|Ltd|Inc))',
            r'([A-Z][a-z]+\s+(?:Pvt\s+Ltd|Ltd|Inc))'
        ]
        
        company_names = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            company_names.extend(matches)
        
        # Address patterns
        address_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s*[A-Z][a-z]+)'
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = re.findall(pattern, text)
            addresses.extend(matches)
        
        return {
            "extracted_data": {
                "gstins": gstins,
                "company_names": company_names,
                "addresses": addresses,
                "contacts": []
            },
            "validated_gstins": gstins,
            "confidence": 0.6
        }
    
    def reconcile_supplier_data(self, supplier_data: Dict) -> Dict:
        """
        Reconcile supplier data with GSTIN database
        """
        results = {
            "supplier_data": supplier_data,
            "reconciliation_results": [],
            "recommendations": []
        }
        
        # Extract GSTINs from supplier data
        text_data = " ".join([
            supplier_data.get('company_name', ''),
            supplier_data.get('description', ''),
            supplier_data.get('address', ''),
            supplier_data.get('contact_info', '')
        ])
        
        # Intelligent extraction
        extracted_info = self.intelligent_gstin_extraction(text_data)
        
        # Process each extracted GSTIN
        for gstin in extracted_info.get('validated_gstins', []):
            # Check if GSTIN exists in database
            db_match = self.gstin_database[self.gstin_database['gstin'] == gstin]
            
            if not db_match.empty:
                # Exact match found
                match_info = {
                    "gstin": gstin,
                    "match_type": "exact",
                    "database_info": db_match.iloc[0].to_dict(),
                    "confidence": 1.0
                }
                results["reconciliation_results"].append(match_info)
            else:
                # No exact match, try fuzzy matching
                company_name = supplier_data.get('company_name', '')
                if company_name:
                    fuzzy_matches = self.fuzzy_match_company_name(company_name)
                    if fuzzy_matches:
                        match_info = {
                            "gstin": gstin,
                            "match_type": "fuzzy",
                            "fuzzy_matches": fuzzy_matches,
                            "confidence": fuzzy_matches[0]['similarity_score']
                        }
                        results["reconciliation_results"].append(match_info)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations
        
        return results
    
    def _generate_recommendations(self, reconciliation_results: Dict) -> List[str]:
        """Generate reconciliation recommendations"""
        recommendations = []
        
        if not reconciliation_results["reconciliation_results"]:
            recommendations.append("No GSTIN matches found. Please verify the GSTIN number.")
            return recommendations
        
        for result in reconciliation_results["reconciliation_results"]:
            if result["match_type"] == "exact":
                recommendations.append(f"✓ GSTIN {result['gstin']} verified in database")
            elif result["match_type"] == "fuzzy":
                best_match = result["fuzzy_matches"][0]
                if best_match["similarity_score"] > 0.8:
                    recommendations.append(f"⚠ High similarity match found: {best_match['business_name']} ({best_match['gstin']})")
                else:
                    recommendations.append(f"❓ Possible match: {best_match['business_name']} ({best_match['gstin']}) - Please verify")
        
        return recommendations
    
    def batch_reconciliation(self, supplier_list: List[Dict]) -> Dict:
        """
        Perform batch reconciliation for multiple suppliers
        """
        batch_results = {
            "total_suppliers": len(supplier_list),
            "successful_reconciliations": 0,
            "failed_reconciliations": 0,
            "results": []
        }
        
        for i, supplier in enumerate(supplier_list):
            try:
                result = self.reconcile_supplier_data(supplier)
                result["supplier_index"] = i
                batch_results["results"].append(result)
                batch_results["successful_reconciliations"] += 1
                
            except Exception as e:
                logger.error(f"Error reconciling supplier {i}: {e}")
                batch_results["failed_reconciliations"] += 1
                batch_results["results"].append({
                    "supplier_index": i,
                    "error": str(e),
                    "supplier_data": supplier
                })
        
        return batch_results
    
    def generate_reconciliation_report(self, batch_results: Dict) -> str:
        """
        Generate reconciliation report
        """
        report_lines = [
            "GSTIN Reconciliation Report",
            "=" * 50,
            f"Total Suppliers: {batch_results['total_suppliers']}",
            f"Successful Reconciliations: {batch_results['successful_reconciliations']}",
            f"Failed Reconciliations: {batch_results['failed_reconciliations']}",
            "",
            "Detailed Results:",
            "-" * 30
        ]
        
        for result in batch_results["results"]:
            if "error" in result:
                report_lines.append(f"❌ Supplier {result['supplier_index']}: {result['error']}")
            else:
                supplier_name = result["supplier_data"].get("company_name", "Unknown")
                report_lines.append(f"✓ Supplier {result['supplier_index']}: {supplier_name}")
                
                for rec in result["reconciliation_results"]:
                    if rec["match_type"] == "exact":
                        report_lines.append(f"  - Exact match: {rec['gstin']}")
                    else:
                        best_match = rec["fuzzy_matches"][0]
                        report_lines.append(f"  - Fuzzy match: {best_match['gstin']} ({best_match['similarity_score']:.2f})")
        
        return "\n".join(report_lines)
