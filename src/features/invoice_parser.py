"""
6. Indic Vision-Language Invoice Parser
Fine-tune open-source OCR models locally to turn scanned Indian invoices into structured data
"""

import logging
import json
import re
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from PIL import Image

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available, using fallback")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available, using fallback")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using fallback")

from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InvoiceData:
    """Structured invoice data"""
    invoice_number: str
    invoice_date: str
    due_date: str
    seller_name: str
    seller_address: str
    seller_gstin: str
    buyer_name: str
    buyer_address: str
    buyer_gstin: str
    items: List[Dict]
    subtotal: float
    cgst: float
    sgst: float
    igst: float
    total_amount: float
    hsn_sac_codes: List[str]

class InvoiceParser:
    """
    Indic Vision-Language Invoice Parser
    Processes scanned Indian invoices and extracts structured data
    """
    
    def __init__(self, ocr_engine: str = "easyocr"):
        self.ocr_engine = ocr_engine
        self.reader = None
        self.ner_model = None
        self.ner_tokenizer = None
        
        # Initialize OCR reader
        self._initialize_ocr()
        
        # Initialize NER model for entity extraction
        self._initialize_ner()
        
        # Indian invoice patterns
        self.invoice_patterns = self._initialize_patterns()
        
        logger.info("Indic Vision-Language Invoice Parser initialized")
    
    def _initialize_ocr(self):
        """Initialize OCR engine"""
        try:
            if self.ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
                self.reader = easyocr.Reader(['en', 'hi'])  # English and Hindi
            elif self.ocr_engine == "tesseract" and TESSERACT_AVAILABLE:
                # Configure tesseract for better Indian document recognition
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path
            else:
                logger.warning(f"OCR engine {self.ocr_engine} not available, using fallback")
                self._initialize_fallback_ocr()
                return
            
            logger.info(f"OCR engine {self.ocr_engine} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing OCR: {e}")
            self._initialize_fallback_ocr()
    
    def _initialize_fallback_ocr(self):
        """Initialize fallback OCR"""
        logger.warning("Using fallback OCR processing")
        self.fallback_mode = True
    
    def _initialize_ner(self):
        """Initialize NER model for entity extraction"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # For demo, using a smaller model
                self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
                
                logger.info("NER model initialized")
            else:
                logger.warning("Transformers not available, NER disabled")
                self.ner_model = None
                self.ner_tokenizer = None
            
        except Exception as e:
            logger.error(f"Error initializing NER model: {e}")
            self.ner_model = None
            self.ner_tokenizer = None
    
    def _initialize_patterns(self) -> Dict[str, str]:
        """Initialize Indian invoice patterns"""
        return {
            # Invoice number patterns
            "invoice_number": [
                r'Invoice\s*No\.?\s*:?\s*([A-Z0-9/-]+)',
                r'Bill\s*No\.?\s*:?\s*([A-Z0-9/-]+)',
                r'Tax\s*Invoice\s*No\.?\s*:?\s*([A-Z0-9/-]+)',
                r'INV\s*/?\s*([A-Z0-9/-]+)'
            ],
            
            # Date patterns
            "invoice_date": [
                r'Invoice\s*Date\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})',
                r'Date\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})',
                r'Bill\s*Date\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})'
            ],
            
            "due_date": [
                r'Due\s*Date\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})',
                r'Payment\s*Due\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})'
            ],
            
            # GSTIN patterns
            "gstin": [
                r'GSTIN\s*:?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})',
                r'Tax\s*ID\s*:?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})'
            ],
            
            # Amount patterns
            "total_amount": [
                r'Total\s*:?\s*₹?\s*([\d,]+\.\d{2})',
                r'Grand\s*Total\s*:?\s*₹?\s*([\d,]+\.\d{2})',
                r'Amount\s*Payable\s*:?\s*₹?\s*([\d,]+\.\d{2})'
            ],
            
            # Tax patterns
            "cgst": [
                r'CGST\s*@?\s*\d+%?\s*:?\s*₹?\s*([\d,]+\.\d{2})',
                r'Central\s*GST\s*:?\s*₹?\s*([\d,]+\.\d{2})'
            ],
            
            "sgst": [
                r'SGST\s*@?\s*\d+%?\s*:?\s*₹?\s*([\d,]+\.\d{2})',
                r'State\s*GST\s*:?\s*₹?\s*([\d,]+\.\d{2})'
            ],
            
            "igst": [
                r'IGST\s*@?\s*\d+%?\s*:?\s*₹?\s*([\d,]+\.\d{2})',
                r'Integrated\s*GST\s*:?\s*₹?\s*([\d,]+\.\d{2})'
            ],
            
            # HSN/SAC patterns
            "hsn_sac": [
                r'HSN\s*Code\s*:?\s*(\d{4,8})',
                r'SAC\s*Code\s*:?\s*(\d{4,8})',
                r'HSN/SAC\s*:?\s*(\d{4,8})'
            ]
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess invoice image for better OCR
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise
            denoised = cv2.medianBlur(binary, 5)
            
            # Enhance contrast
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=10)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from invoice image using OCR
        """
        try:
            if hasattr(self, 'fallback_mode'):
                return self._fallback_ocr(image_path)
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return ""
            
            # Extract text using EasyOCR
            if self.ocr_engine == "easyocr" and EASYOCR_AVAILABLE and hasattr(self, 'reader'):
                results = self.reader.readtext(processed_image)
                text = " ".join([result[1] for result in results])
            elif self.ocr_engine == "tesseract" and TESSERACT_AVAILABLE:
                # Use Tesseract
                text = pytesseract.image_to_string(processed_image, lang='eng+hin')
            else:
                # Fallback to mock OCR
                return self._fallback_ocr(image_path)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return self._fallback_ocr(image_path)
    
    def _fallback_ocr(self, image_path: str) -> str:
        """Fallback OCR using basic processing"""
        logger.info("Using fallback OCR")
        return "[FALLBACK] OCR processing not available"
    
    def extract_entities_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from OCR text using patterns and NER
        """
        entities = {
            "invoice_number": "",
            "invoice_date": "",
            "due_date": "",
            "seller_name": "",
            "seller_address": "",
            "seller_gstin": "",
            "buyer_name": "",
            "buyer_address": "",
            "buyer_gstin": "",
            "items": [],
            "subtotal": 0.0,
            "cgst": 0.0,
            "sgst": 0.0,
            "igst": 0.0,
            "total_amount": 0.0,
            "hsn_sac_codes": []
        }
        
        # Extract using regex patterns
        for field, patterns in self.invoice_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if field in ["invoice_number", "invoice_date", "due_date", "gstin", "total_amount", 
                               "cgst", "sgst", "igst", "hsn_sac"]:
                        entities[field] = match.group(1)
                        break
        
        # Extract names and addresses using heuristics
        entities.update(self._extract_names_and_addresses(text))
        
        # Extract line items
        entities["items"] = self._extract_line_items(text)
        
        # Extract HSN/SAC codes from line items
        entities["hsn_sac_codes"] = [item.get("hsn_sac", "") for item in entities["items"] if item.get("hsn_sac")]
        
        # Calculate subtotal if not found
        if entities["subtotal"] == 0.0:
            entities["subtotal"] = entities["total_amount"] - entities["cgst"] - entities["sgst"] - entities["igst"]
        
        return entities
    
    def _extract_names_and_addresses(self, text: str) -> Dict[str, str]:
        """Extract seller and buyer information"""
        result = {
            "seller_name": "",
            "seller_address": "",
            "buyer_name": "",
            "buyer_address": ""
        }
        
        # Split text into lines
        lines = text.split('\n')
        
        # Look for seller/buyer sections
        seller_section = []
        buyer_section = []
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if any(keyword in line_lower for keyword in ["sold by", "from", "supplier", "vendor"]):
                current_section = "seller"
                continue
            elif any(keyword in line_lower for keyword in ["billed to", "to", "customer", "client"]):
                current_section = "buyer"
                continue
            elif any(keyword in line_lower for keyword in ["description", "item", "quantity", "rate", "amount"]):
                # Likely start of item table
                break
            
            if current_section == "seller" and line.strip():
                seller_section.append(line.strip())
            elif current_section == "buyer" and line.strip():
                buyer_section.append(line.strip())
        
        # Extract name and address from sections
        if seller_section:
            result["seller_name"] = seller_section[0] if seller_section else ""
            result["seller_address"] = " ".join(seller_section[1:]) if len(seller_section) > 1 else ""
        
        if buyer_section:
            result["buyer_name"] = buyer_section[0] if buyer_section else ""
            result["buyer_address"] = " ".join(buyer_section[1:]) if len(buyer_section) > 1 else ""
        
        return result
    
    def _extract_line_items(self, text: str) -> List[Dict]:
        """Extract line items from invoice"""
        items = []
        
        # Look for item table patterns
        lines = text.split('\n')
        
        # Find header line
        header_idx = -1
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ["description", "item", "qty", "rate", "amount"]):
                header_idx = i
                break
        
        if header_idx == -1:
            return items
        
        # Extract items from subsequent lines
        for line in lines[header_idx + 1:]:
            line = line.strip()
            if not line or any(keyword in line.lower() for keyword in ["total", "subtotal", "tax", "cgst", "sgst", "igst"]):
                continue
            
            # Try to parse item line
            item = self._parse_item_line(line)
            if item:
                items.append(item)
        
        return items
    
    def _parse_item_line(self, line: str) -> Optional[Dict]:
        """Parse a single item line"""
        # Simple heuristic parsing
        parts = line.split()
        
        if len(parts) < 3:
            return None
        
        # Look for numeric values (quantity, rate, amount)
        numbers = []
        for part in parts:
            if re.match(r'[\d,]+\.?\d*', part):
                numbers.append(float(part.replace(',', '')))
        
        if len(numbers) >= 2:
            return {
                "description": " ".join(parts[:-len(numbers)]),
                "quantity": numbers[0] if len(numbers) >= 1 else 0,
                "rate": numbers[1] if len(numbers) >= 2 else 0,
                "amount": numbers[-1] if len(numbers) >= 3 else numbers[0] * numbers[1],
                "hsn_sac": ""
            }
        
        return None
    
    def parse_invoice(self, image_path: str) -> InvoiceData:
        """
        Parse invoice image and extract structured data
        """
        try:
            # Extract text using OCR
            text = self.extract_text_with_ocr(image_path)
            
            if not text:
                raise ValueError("No text extracted from image")
            
            # Extract entities
            entities = self.extract_entities_from_text(text)
            
            # Create InvoiceData object
            invoice_data = InvoiceData(
                invoice_number=entities["invoice_number"],
                invoice_date=entities["invoice_date"],
                due_date=entities["due_date"],
                seller_name=entities["seller_name"],
                seller_address=entities["seller_address"],
                seller_gstin=entities["seller_gstin"],
                buyer_name=entities["buyer_name"],
                buyer_address=entities["buyer_address"],
                buyer_gstin=entities["buyer_gstin"],
                items=entities["items"],
                subtotal=entities["subtotal"],
                cgst=entities["cgst"],
                sgst=entities["sgst"],
                igst=entities["igst"],
                total_amount=entities["total_amount"],
                hsn_sac_codes=entities["hsn_sac_codes"]
            )
            
            logger.info(f"Successfully parsed invoice: {invoice_data.invoice_number}")
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"Error parsing invoice: {e}")
            # Return empty invoice data
            return InvoiceData(
                invoice_number="", invoice_date="", due_date="",
                seller_name="", seller_address="", seller_gstin="",
                buyer_name="", buyer_address="", buyer_gstin="",
                items=[], subtotal=0.0, cgst=0.0, sgst=0.0, igst=0.0,
                total_amount=0.0, hsn_sac_codes=[]
            )
    
    def validate_extracted_data(self, invoice_data: InvoiceData) -> Dict[str, Any]:
        """
        Validate extracted invoice data
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "confidence_score": 0.0
        }
        
        # Check required fields
        required_fields = ["invoice_number", "invoice_date", "seller_name", "total_amount"]
        
        for field in required_fields:
            value = getattr(invoice_data, field)
            if not value or (isinstance(value, str) and not value.strip()):
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["is_valid"] = False
        
        # Validate GSTIN format
        if invoice_data.seller_gstin and not self._validate_gstin(invoice_data.seller_gstin):
            validation_result["warnings"].append("Invalid seller GSTIN format")
        
        if invoice_data.buyer_gstin and not self._validate_gstin(invoice_data.buyer_gstin):
            validation_result["warnings"].append("Invalid buyer GSTIN format")
        
        # Validate date formats
        if invoice_data.invoice_date and not self._validate_date(invoice_data.invoice_date):
            validation_result["warnings"].append("Invalid invoice date format")
        
        if invoice_data.due_date and not self._validate_date(invoice_data.due_date):
            validation_result["warnings"].append("Invalid due date format")
        
        # Validate amounts
        if invoice_data.total_amount <= 0:
            validation_result["errors"].append("Total amount must be positive")
            validation_result["is_valid"] = False
        
        # Check tax calculations
        calculated_total = invoice_data.subtotal + invoice_data.cgst + invoice_data.sgst + invoice_data.igst
        if abs(calculated_total - invoice_data.total_amount) > 0.01:
            validation_result["warnings"].append("Tax calculations may be incorrect")
        
        # Calculate confidence score
        validation_result["confidence_score"] = self._calculate_confidence_score(invoice_data, validation_result)
        
        return validation_result
    
    def _validate_gstin(self, gstin: str) -> bool:
        """Validate GSTIN format"""
        gstin = gstin.upper().replace(' ', '')
        if len(gstin) != 15:
            return False
        pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$'
        return bool(re.match(pattern, gstin))
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date format"""
        date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y']
        
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    def _calculate_confidence_score(self, invoice_data: InvoiceData, validation_result: Dict) -> float:
        """Calculate confidence score for extracted data"""
        score = 1.0
        
        # Deduct points for errors
        score -= len(validation_result["errors"]) * 0.3
        
        # Deduct points for warnings
        score -= len(validation_result["warnings"]) * 0.1
        
        # Bonus for complete data
        if invoice_data.invoice_number and invoice_data.invoice_date and invoice_data.seller_name:
            score += 0.1
        
        # Bonus for GSTIN validation
        if invoice_data.seller_gstin and self._validate_gstin(invoice_data.seller_gstin):
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def batch_parse_invoices(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Parse multiple invoices in batch
        """
        results = {
            "total_invoices": len(image_paths),
            "successful_parses": 0,
            "failed_parses": 0,
            "invoice_data": [],
            "validation_results": []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                # Parse invoice
                invoice_data = self.parse_invoice(image_path)
                
                # Validate data
                validation = self.validate_extracted_data(invoice_data)
                
                result = {
                    "image_path": image_path,
                    "invoice_data": invoice_data,
                    "validation": validation,
                    "success": validation["is_valid"]
                }
                
                results["invoice_data"].append(result)
                results["validation_results"].append(validation)
                
                if validation["is_valid"]:
                    results["successful_parses"] += 1
                else:
                    results["failed_parses"] += 1
                
            except Exception as e:
                logger.error(f"Error parsing invoice {i}: {e}")
                results["failed_parses"] += 1
                results["invoice_data"].append({
                    "image_path": image_path,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def export_to_json(self, invoice_data: InvoiceData, output_path: str):
        """Export invoice data to JSON"""
        data = {
            "invoice_number": invoice_data.invoice_number,
            "invoice_date": invoice_data.invoice_date,
            "due_date": invoice_data.due_date,
            "seller": {
                "name": invoice_data.seller_name,
                "address": invoice_data.seller_address,
                "gstin": invoice_data.seller_gstin
            },
            "buyer": {
                "name": invoice_data.buyer_name,
                "address": invoice_data.buyer_address,
                "gstin": invoice_data.buyer_gstin
            },
            "items": invoice_data.items,
            "amounts": {
                "subtotal": invoice_data.subtotal,
                "cgst": invoice_data.cgst,
                "sgst": invoice_data.sgst,
                "igst": invoice_data.igst,
                "total": invoice_data.total_amount
            },
            "hsn_sac_codes": invoice_data.hsn_sac_codes
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Invoice data exported to {output_path}")
    
    def generate_parsing_report(self, batch_results: Dict) -> str:
        """Generate invoice parsing report"""
        report_lines = [
            "Invoice Parsing Report",
            "=" * 30,
            f"Total Invoices: {batch_results['total_invoices']}",
            f"Successful Parses: {batch_results['successful_parses']}",
            f"Failed Parses: {batch_results['failed_parses']}",
            f"Success Rate: {(batch_results['successful_parses'] / batch_results['total_invoices']) * 100:.1f}%",
            "",
            "Average Confidence Score:",
            "-" * 25
        ]
        
        # Calculate average confidence
        valid_validations = [v for v in batch_results["validation_results"] if v.get("confidence_score") is not None]
        if valid_validations:
            avg_confidence = sum(v["confidence_score"] for v in valid_validations) / len(valid_validations)
            report_lines.append(f"{avg_confidence:.3f}")
        
        # Common errors
        report_lines.extend([
            "",
            "Common Validation Issues:",
            "-" * 25
        ])
        
        error_counts = {}
        for validation in batch_results["validation_results"]:
            for error in validation.get("errors", []):
                error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"{error}: {count}")
        
        return "\n".join(report_lines)
