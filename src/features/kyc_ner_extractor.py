"""
3. Corporate KYC NER Extractor
Train custom NER model locally using IndicBERT to extract corporate identity data
"""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

class CorporateKYCDataset(Dataset):
    """Custom dataset for KYC NER training"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label to id mapping
        unique_labels = set()
        for label_list in labels:
            unique_labels.update(label_list)
        self.label_to_id = {label: i for i, label in enumerate(sorted(unique_labels))}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels to ids
        label_ids = [self.label_to_id[label] for label in labels]
        # Pad label ids
        label_ids = label_ids + [self.label_to_id['O']] * (self.max_length - len(label_ids))
        label_ids = label_ids[:self.max_length]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class CorporateKYCNERExtractor:
    """
    Corporate KYC NER Extractor using an open-source transformer model
    Extracts corporate identity data from documents
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.label_to_id = None
        self.id_to_label = None
        
        # Entity types for KYC
        self.entity_types = [
            'COMPANY_NAME', 'DIRECTOR_NAME', 'ADDRESS', 'PAN', 'GSTIN', 
            'EMAIL', 'PHONE', 'WEBSITE', 'CIN', 'DATE_OF_INCORPORATION',
            'AUTHORIZED_SIGNATORY', 'BANK_ACCOUNT', 'IFSC_CODE'
        ]
        
        # Load model
        self._load_model()
        
        logger.info("Corporate KYC NER Extractor initialized")
    
    def _load_model(self):
        """Load transformer model for NER"""
        try:
            labels = ['O'] + [f"B-{etype}" for etype in self.entity_types] + [f"I-{etype}" for etype in self.entity_types]

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, 
                num_labels=len(labels),
                ignore_mismatched_sizes=True,
            ).to(self.device)
            
            # Initialize label mappings
            self.label_to_id = {label: i for i, label in enumerate(labels)}
            self.id_to_label = {i: label for label, i in self.label_to_id.items()}
            
            logger.info("NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback rule-based extraction"""
        logger.warning("Using fallback rule-based extraction")
        self.fallback_mode = True
        self._setup_regex_patterns()
    
    def _setup_regex_patterns(self):
        """Setup regex patterns for entity extraction"""
        self.patterns = {
            'PAN': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            'GSTIN': r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',
            'WEBSITE': r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'CIN': r'\b[L|U]{1}[0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b',
            'IFSC_CODE': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            'DATE': r'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/]\d{4}\b'
        }
    
    def prepare_training_data(self, documents: List[Dict]) -> Tuple[List[str], List[List[str]]]:
        """
        Prepare training data from annotated documents
        """
        texts = []
        labels = []
        
        for doc in documents:
            text = doc.get('text', '')
            entities = doc.get('entities', [])
            
            # Tokenize text
            tokens = text.split()
            token_labels = ['O'] * len(tokens)
            
            # Assign labels to tokens
            for entity in entities:
                entity_type = entity['type']
                start = entity['start']
                end = entity['end']
                entity_text = text[start:end]
                
                # Find tokens that overlap with entity
                entity_tokens = entity_text.split()
                for i, token in enumerate(tokens):
                    if token in entity_tokens:
                        if i == 0 or token_labels[i-1] == 'O':
                            token_labels[i] = f"B-{entity_type}"
                        else:
                            token_labels[i] = f"I-{entity_type}"
            
            texts.append(text)
            labels.append(token_labels)
        
        return texts, labels
    
    def train_model(self, training_data: List[Dict], validation_split: float = 0.2):
        """
        Train the NER model
        """
        logger.info("Starting NER model training...")
        
        # Prepare data
        texts, labels = self.prepare_training_data(training_data)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42
        )
        
        # Create datasets
        train_dataset = CorporateKYCDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = CorporateKYCDataset(val_texts, val_labels, self.tokenizer)
        
        # Update label mappings
        self.label_to_id = train_dataset.label_to_id
        self.id_to_label = train_dataset.id_to_label
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./kyc_ner_model",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=50,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model("./kyc_ner_model")
        self.tokenizer.save_pretrained("./kyc_ner_model")
        
        logger.info("NER model training completed")
        
        return {
            "status": "completed",
            "training_samples": len(train_texts),
            "validation_samples": len(val_texts),
            "model_path": "./kyc_ner_model"
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using trained model
        """
        try:
            if hasattr(self, 'fallback_mode'):
                return self._fallback_extraction(text)
            
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert predictions to entities
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.id_to_label[prediction.item()] for prediction in predictions[0]]
            
            # Group consecutive tokens with same label
            entities = self._group_entities(tokens, predicted_labels, text)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return self._fallback_extraction(text)
    
    def _group_entities(self, tokens: List[str], labels: List[str], original_text: str) -> List[Dict]:
        """Group consecutive tokens with same entity label"""
        entities = []
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            # Skip special tokens
            if token.startswith('[') and token.endswith(']'):
                continue
            
            if label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    'type': label[2:],
                    'text': token.replace('##', ''),
                    'start': 0,  # Will be calculated later
                    'end': 0,    # Will be calculated later
                    'confidence': 0.9
                }
                current_tokens = [token]
            
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                # Continue current entity
                current_tokens.append(token)
                current_entity['text'] += token.replace('##', '')
            
            else:
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    current_tokens = []
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        # Calculate start and end positions
        for entity in entities:
            entity_text = entity['text']
            start_pos = original_text.find(entity_text)
            if start_pos != -1:
                entity['start'] = start_pos
                entity['end'] = start_pos + len(entity_text)
            else:
                # Fallback: try to find approximate position
                words = original_text.split()
                for i, word in enumerate(words):
                    if entity_text.lower() in word.lower():
                        entity['start'] = original_text.find(word)
                        entity['end'] = entity['start'] + len(word)
                        break
        
        return entities
    
    def _fallback_extraction(self, text: str) -> List[Dict]:
        """Fallback extraction using regex patterns"""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })
        
        # Extract company names (more complex pattern)
        company_entities = self._extract_company_names(text)
        entities.extend(company_entities)
        
        # Extract director names
        director_entities = self._extract_director_names(text)
        entities.extend(director_entities)
        
        return entities
    
    def _extract_company_names(self, text: str) -> List[Dict]:
        """Extract company names using patterns"""
        company_patterns = [
            r'([A-Z][a-zA-Z\s&]+(?:Pvt\.?\s*Ltd\.?|Ltd\.?|Private\s*Limited|Limited|Inc\.?|Corporation|LLP))',
            r'([A-Z][a-zA-Z\s]+(?:Technologies|Solutions|Industries|Services|Systems|Innovations))'
        ]
        
        entities = []
        for pattern in company_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'COMPANY_NAME',
                    'text': match.group().strip(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return entities
    
    def _extract_director_names(self, text: str) -> List[Dict]:
        """Extract director names using patterns"""
        # Look for patterns like "Director: John Doe" or "Mr. John Smith"
        director_patterns = [
            r'(?:Director|Directors?)[s:]*\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:-|–)\s*(?:Director|MD|CEO)'
        ]
        
        entities = []
        for pattern in director_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'type': 'DIRECTOR_NAME',
                    'text': match.group(1).strip(),
                    'start': match.start(1),
                    'end': match.end(1),
                    'confidence': 0.7
                })
        
        return entities
    
    def validate_extracted_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate and filter extracted entities"""
        validated_entities = []
        
        for entity in entities:
            entity_type = entity['type']
            entity_text = entity['text']
            
            # Type-specific validation
            if entity_type == 'PAN':
                if self._validate_pan(entity_text):
                    validated_entities.append(entity)
            elif entity_type == 'GSTIN':
                if self._validate_gstin(entity_text):
                    validated_entities.append(entity)
            elif entity_type == 'EMAIL':
                if self._validate_email(entity_text):
                    validated_entities.append(entity)
            elif entity_type == 'PHONE':
                if self._validate_phone(entity_text):
                    validated_entities.append(entity)
            elif entity_type == 'CIN':
                if self._validate_cin(entity_text):
                    validated_entities.append(entity)
            else:
                # For other types, just check basic criteria
                if len(entity_text.strip()) > 0:
                    validated_entities.append(entity)
        
        return validated_entities
    
    def _validate_pan(self, pan: str) -> bool:
        """Validate PAN format"""
        pan = pan.upper().replace(' ', '')
        return len(pan) == 10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[9].isalpha()
    
    def _validate_gstin(self, gstin: str) -> bool:
        """Validate GSTIN format"""
        gstin = gstin.upper().replace(' ', '')
        if len(gstin) != 15:
            return False
        pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$'
        return bool(re.match(pattern, gstin))
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number"""
        phone = re.sub(r'[^\d]', '', phone)
        return len(phone) == 10 and phone.startswith(('6', '7', '8', '9'))
    
    def _validate_cin(self, cin: str) -> bool:
        """Validate CIN format"""
        cin = cin.upper().replace(' ', '')
        if len(cin) != 21:
            return False
        pattern = r'^[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}$'
        return bool(re.match(pattern, cin))
    
    def process_document(self, document_text: str) -> Dict:
        """
        Process entire document and extract all entities
        """
        # Extract entities
        entities = self.extract_entities(document_text)
        
        # Validate entities
        validated_entities = self.validate_extracted_entities(entities)
        
        # Group entities by type
        grouped_entities = {}
        for entity in validated_entities:
            entity_type = entity['type']
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append(entity)
        
        # Generate summary
        summary = self._generate_summary(grouped_entities)
        
        return {
            "document_text": document_text,
            "extracted_entities": validated_entities,
            "grouped_entities": grouped_entities,
            "summary": summary,
            "total_entities": len(validated_entities)
        }
    
    def _generate_summary(self, grouped_entities: Dict) -> Dict:
        """Generate summary of extracted entities"""
        summary = {
            "company_identified": len(grouped_entities.get('COMPANY_NAME', [])) > 0,
            "directors_identified": len(grouped_entities.get('DIRECTOR_NAME', [])),
            "tax_ids": {
                "pan": len(grouped_entities.get('PAN', [])),
                "gstin": len(grouped_entities.get('GSTIN', [])),
                "cin": len(grouped_entities.get('CIN', []))
            },
            "contact_info": {
                "emails": len(grouped_entities.get('EMAIL', [])),
                "phones": len(grouped_entities.get('PHONE', [])),
                "websites": len(grouped_entities.get('WEBSITE', []))
            },
            "bank_info": {
                "accounts": len(grouped_entities.get('BANK_ACCOUNT', [])),
                "ifsc_codes": len(grouped_entities.get('IFSC_CODE', []))
            }
        }
        
        return summary
    
    def export_results(self, results: Dict, output_path: str):
        """Export extraction results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {output_path}")
