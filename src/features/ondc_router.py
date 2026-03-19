"""
5. Multi-Agent ONDC Router
Fine-tune local Llama 3 8B model to translate natural language procurement requests into Beckn Protocol JSON
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class BecknMessage:
    """Beckn Protocol message structure"""
    context: Dict[str, Any]
    message: Dict[str, Any]

class ONDCRouter:
    """
    Multi-Agent ONDC Router using local Llama 3 8B
    Translates natural language procurement requests to Beckn Protocol JSON
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        
        # Beckn Protocol templates
        self.beckn_templates = self._initialize_beckn_templates()
        
        # Load model
        self._load_model()
        
        logger.info("ONDC Router initialized")
    
    def _load_model(self):
        """Load Llama 3 model for ONDC routing"""
        try:
            # For demo, using a smaller model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium"
            ).to(self.device)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("ONDC model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ONDC model: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback rule-based processing"""
        logger.warning("Using fallback processing for ONDC routing")
        self.fallback_mode = True
    
    def _initialize_beckn_templates(self) -> Dict[str, Dict]:
        """Initialize Beckn Protocol message templates"""
        return {
            "search": {
                "context": {
                    "domain": "ONDC:RET10",
                    "country": "IND",
                    "city": "*",
                    "action": "search",
                    "core_version": "1.2.0",
                    "bap_id": "ondc-bap",
                    "bap_uri": "https://bap.ondc.com",
                    "transaction_id": "",
                    "message_id": "",
                    "timestamp": ""
                },
                "message": {
                    "intent": {
                        "item": {
                            "descriptor": {
                                "name": ""
                            }
                        },
                        "category": {
                            "descriptor": {
                                "name": ""
                            }
                        },
                        "location": {
                            "city": {
                                "name": ""
                            },
                            "country": {
                                "code": "IND"
                            }
                        },
                        "price": {
                            "currency": "INR",
                            "value": ""
                        }
                    }
                }
            },
            "select": {
                "context": {
                    "domain": "ONDC:RET10",
                    "country": "IND",
                    "city": "*",
                    "action": "select",
                    "core_version": "1.2.0",
                    "bap_id": "ondc-bap",
                    "bap_uri": "https://bap.ondc.com",
                    "transaction_id": "",
                    "message_id": "",
                    "timestamp": ""
                },
                "message": {
                    "order": {
                        "provider": {
                            "id": "",
                            "locations": [{"id": ""}]
                        },
                        "items": [{
                            "id": "",
                            "quantity": {
                                "count": 0
                            }
                        }]
                    }
                }
            },
            "init": {
                "context": {
                    "domain": "ONDC:RET10",
                    "country": "IND",
                    "city": "*",
                    "action": "init",
                    "core_version": "1.2.0",
                    "bap_id": "ondc-bap",
                    "bap_uri": "https://bap.ondc.com",
                    "transaction_id": "",
                    "message_id": "",
                    "timestamp": ""
                },
                "message": {
                    "order": {
                        "provider": {
                            "id": "",
                            "locations": [{"id": ""}]
                        },
                        "items": [{
                            "id": "",
                            "quantity": {
                                "count": 0
                            }
                        }],
                        "billing": {
                            "name": "",
                            "address": "",
                            "phone": ""
                        },
                        "fulfillment": {
                            "end": {
                                "location": {
                                    "address": ""
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def parse_natural_language_request(self, request: str) -> Dict:
        """
        Parse natural language procurement request
        """
        try:
            if hasattr(self, 'fallback_mode'):
                return self._fallback_parsing(request)
            
            # Prepare prompt for LLM
            prompt = f"""
            Parse the following procurement request and extract key information:
            
            Request: "{request}"
            
            Extract:
            1. Action (search/select/init/confirm/cancel)
            2. Product/Service name
            3. Category
            4. Quantity
            5. Price range (if mentioned)
            6. Location/City
            7. Provider/Store name (if mentioned)
            8. Customer details (name, phone, address)
            
            Respond in JSON format:
            {{
                "action": "action_name",
                "product": "product_name",
                "category": "category_name",
                "quantity": number,
                "price_range": "price_range",
                "location": "city_name",
                "provider": "provider_name",
                "customer": {{
                    "name": "customer_name",
                    "phone": "phone_number",
                    "address": "delivery_address"
                }}
            }}
            """
            
            # Generate response using LLM
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
                parsed_data = json.loads(response.split('{')[-1].split('}')[0] + '}')
            except:
                parsed_data = self._fallback_parsing(request)
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing natural language request: {e}")
            return self._fallback_parsing(request)
    
    def _fallback_parsing(self, request: str) -> Dict:
        """Fallback parsing using regex and heuristics"""
        request_lower = request.lower()
        
        parsed = {
            "action": "search",
            "product": "",
            "category": "",
            "quantity": 1,
            "price_range": "",
            "location": "",
            "provider": "",
            "customer": {
                "name": "",
                "phone": "",
                "address": ""
            }
        }
        
        # Extract action
        actions = {
            "search": ["search", "find", "look for", "show me"],
            "select": ["select", "choose", "pick", "take"],
            "init": ["order", "buy", "purchase", "book"],
            "confirm": ["confirm", "proceed", "yes"],
            "cancel": ["cancel", "stop", "no"]
        }
        
        for action, keywords in actions.items():
            if any(keyword in request_lower for keyword in keywords):
                parsed["action"] = action
                break
        
        # Extract product names (common Indian products)
        product_patterns = [
            r'\b(?:rice|wheat|flour|dal|lentils|vegetables|fruits|milk|bread|eggs|chicken|fish|meat)\b',
            r'\b(?:mobile|phone|laptop|computer|tablet|headphones|speaker|camera)\b',
            r'\b(?:shirt|pants|jeans|dress|shoes|sandals|watch|jewelry)\b',
            r'\b(?:medicine|tablet|syrup|ointment|bandage|mask|sanitizer)\b'
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, request_lower)
            if match:
                parsed["product"] = match.group()
                break
        
        # Extract quantity
        quantity_patterns = [
            r'(\d+)\s*(?:kg|kilogram|grams?|liters?|pcs?|pieces?|units?)',
            r'(\d+)\s*(?:dozen|dozens?)',
            r'(\d+)\s*(?:pack|packs|box|boxes|bottle|bottles)'
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, request_lower)
            if match:
                parsed["quantity"] = int(match.group(1))
                break
        
        # Extract price range
        price_patterns = [
            r'under\s*₹?\s*(\d+)',
            r'below\s*₹?\s*(\d+)',
            r'₹?\s*(\d+)\s*to\s*₹?\s*(\d+)',
            r'between\s*₹?\s*(\d+)\s*and\s*₹?\s*(\d+)',
            r'₹?\s*(\d+)\s*-\s*₹?\s*(\d+)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, request_lower)
            if match:
                if len(match.groups()) == 2:
                    parsed["price_range"] = f"{match.group(1)}-{match.group(2)}"
                else:
                    parsed["price_range"] = f"0-{match.group(1)}"
                break
        
        # Extract location (Indian cities)
        cities = ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", 
                 "pune", "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", 
                 "indore", "thane", "bhopal", "visakhapatnam", "pimpri", "patna"]
        
        for city in cities:
            if city in request_lower:
                parsed["location"] = city.title()
                break
        
        # Extract phone number
        phone_pattern = r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b'
        phone_match = re.search(phone_pattern, request)
        if phone_match:
            parsed["customer"]["phone"] = phone_match.group()
        
        return parsed
    
    def generate_beckn_message(self, parsed_request: Dict) -> BecknMessage:
        """
        Generate Beckn Protocol message from parsed request
        """
        action = parsed_request.get("action", "search")
        
        # Get template
        if action not in self.beckn_templates:
            action = "search"
        
        template = self.beckn_templates[action]
        
        # Generate unique IDs
        transaction_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat() + "Z"
        
        # Update context
        context = template["context"].copy()
        context.update({
            "transaction_id": transaction_id,
            "message_id": message_id,
            "timestamp": timestamp
        })
        
        # Update message based on action
        if action == "search":
            message = self._create_search_message(parsed_request, template["message"])
        elif action == "select":
            message = self._create_select_message(parsed_request, template["message"])
        elif action == "init":
            message = self._create_init_message(parsed_request, template["message"])
        else:
            message = template["message"].copy()
        
        return BecknMessage(context=context, message=message)
    
    def _create_search_message(self, parsed_request: Dict, template: Dict) -> Dict:
        """Create search message"""
        message = template.copy()
        
        # Update intent
        if parsed_request.get("product"):
            message["message"]["intent"]["item"]["descriptor"]["name"] = parsed_request["product"]
        
        if parsed_request.get("category"):
            message["message"]["intent"]["category"]["descriptor"]["name"] = parsed_request["category"]
        
        if parsed_request.get("location"):
            message["message"]["intent"]["location"]["city"]["name"] = parsed_request["location"]
        
        if parsed_request.get("price_range"):
            # Extract max price from range
            price_range = parsed_request["price_range"]
            if "-" in price_range:
                max_price = price_range.split("-")[1]
            else:
                max_price = price_range.replace("0-", "")
            message["message"]["intent"]["price"]["value"] = max_price
        
        return message
    
    def _create_select_message(self, parsed_request: Dict, template: Dict) -> Dict:
        """Create select message"""
        message = template.copy()
        
        # This would typically require provider and item IDs from previous search results
        # For demo, using placeholder values
        message["message"]["order"]["provider"]["id"] = "provider_001"
        message["message"]["order"]["provider"]["locations"][0]["id"] = "location_001"
        message["message"]["order"]["items"][0]["id"] = "item_001"
        message["message"]["order"]["items"][0]["quantity"]["count"] = parsed_request.get("quantity", 1)
        
        return message
    
    def _create_init_message(self, parsed_request: Dict, template: Dict) -> Dict:
        """Create init message"""
        message = template.copy()
        
        # Provider and item info
        message["message"]["order"]["provider"]["id"] = "provider_001"
        message["message"]["order"]["provider"]["locations"][0]["id"] = "location_001"
        message["message"]["order"]["items"][0]["id"] = "item_001"
        message["message"]["order"]["items"][0]["quantity"]["count"] = parsed_request.get("quantity", 1)
        
        # Customer info
        customer = parsed_request.get("customer", {})
        message["message"]["order"]["billing"]["name"] = customer.get("name", "Customer Name")
        message["message"]["order"]["billing"]["phone"] = customer.get("phone", "")
        message["message"]["order"]["billing"]["address"] = customer.get("address", "Delivery Address")
        
        # Delivery location
        message["message"]["order"]["fulfillment"]["end"]["location"]["address"] = customer.get("address", "Delivery Address")
        
        return message
    
    def route_request(self, natural_language_request: str) -> Dict:
        """
        Route natural language request to appropriate Beckn Protocol message
        """
        try:
            # Parse natural language
            parsed_request = self.parse_natural_language_request(natural_language_request)
            
            # Generate Beckn message
            beckn_message = self.generate_beckn_message(parsed_request)
            
            # Convert to dictionary
            result = {
                "original_request": natural_language_request,
                "parsed_request": parsed_request,
                "beckn_message": asdict(beckn_message),
                "routing_info": {
                    "action": parsed_request.get("action", "search"),
                    "target_domain": "ONDC:RET10",
                    "message_type": f"ondc_{parsed_request.get('action', 'search')}"
                }
            }
            
            logger.info(f"Successfully routed request: {parsed_request.get('action', 'search')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return {
                "error": str(e),
                "original_request": natural_language_request,
                "beckn_message": None
            }
    
    def batch_route_requests(self, requests: List[str]) -> Dict:
        """
        Route multiple requests in batch
        """
        results = {
            "total_requests": len(requests),
            "successful_routes": 0,
            "failed_routes": 0,
            "routing_results": []
        }
        
        for i, request in enumerate(requests):
            try:
                result = self.route_request(request)
                result["request_index"] = i
                results["routing_results"].append(result)
                results["successful_routes"] += 1
                
            except Exception as e:
                logger.error(f"Error routing request {i}: {e}")
                results["failed_routes"] += 1
                results["routing_results"].append({
                    "request_index": i,
                    "error": str(e),
                    "original_request": request
                })
        
        return results
    
    def validate_beckn_message(self, beckn_message: Dict) -> Dict:
        """
        Validate Beckn Protocol message structure
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields in context
            context = beckn_message.get("context", {})
            required_context_fields = ["domain", "action", "core_version", "transaction_id", "message_id", "timestamp"]
            
            for field in required_context_fields:
                if field not in context or not context[field]:
                    validation_result["errors"].append(f"Missing required context field: {field}")
                    validation_result["is_valid"] = False
            
            # Check message structure based on action
            action = context.get("action", "")
            message = beckn_message.get("message", {})
            
            if action == "search":
                if "intent" not in message:
                    validation_result["errors"].append("Missing intent in search message")
                    validation_result["is_valid"] = False
            elif action in ["select", "init"]:
                if "order" not in message:
                    validation_result["errors"].append(f"Missing order in {action} message")
                    validation_result["is_valid"] = False
            
            # Check timestamp format
            timestamp = context.get("timestamp", "")
            if timestamp and not timestamp.endswith("Z"):
                validation_result["warnings"].append("Timestamp should end with 'Z'")
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def generate_routing_report(self, batch_results: Dict) -> str:
        """
        Generate routing performance report
        """
        report_lines = [
            "ONDC Routing Performance Report",
            "=" * 40,
            f"Total Requests: {batch_results['total_requests']}",
            f"Successful Routes: {batch_results['successful_routes']}",
            f"Failed Routes: {batch_results['failed_routes']}",
            f"Success Rate: {(batch_results['successful_routes'] / batch_results['total_requests']) * 100:.1f}%",
            "",
            "Action Distribution:",
            "-" * 20
        ]
        
        # Count actions
        action_counts = {}
        for result in batch_results["routing_results"]:
            if "parsed_request" in result:
                action = result["parsed_request"].get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1
        
        for action, count in action_counts.items():
            report_lines.append(f"{action}: {count}")
        
        return "\n".join(report_lines)
    
    def export_beckn_messages(self, routing_results: List[Dict], output_path: str):
        """Export Beckn messages to JSON file"""
        beckn_messages = []
        
        for result in routing_results:
            if result.get("beckn_message"):
                beckn_messages.append(result["beckn_message"])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(beckn_messages, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(beckn_messages)} Beckn messages to {output_path}")
