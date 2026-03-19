"""
4. Agentic SLA Breach Predictor
Train local ML models (XGBoost/Random Forest) on historical resolution patterns to predict SLA breaches
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SLATicket:
    """Data class for SLA ticket information"""
    ticket_id: str
    priority: str
    category: str
    created_time: datetime
    assigned_to: str
    customer_tier: str
    complexity: str
    resolution_time_hours: float
    sla_hours: float
    breach: bool
    features: Dict[str, Any] = None

class SLABreachPredictor:
    """
    Agentic SLA Breach Predictor using XGBoost and Random Forest
    Predicts SLA breaches and reroutes tasks autonomously
    """
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = "breach"
        
        # Model parameters
        self.model_params = {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            }
        }
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"SLA Breach Predictor initialized with {model_type}")
    
    def _initialize_model(self):
        """Initialize the ML model"""
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**self.model_params["xgboost"])
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(**self.model_params["random_forest"])
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**self.model_params["gradient_boosting"])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic SLA data for training
        """
        np.random.seed(42)
        
        data = []
        priorities = ["Low", "Medium", "High", "Critical"]
        categories = ["Technical", "Billing", "Account", "Network", "Security"]
        customer_tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        complexities = ["Simple", "Moderate", "Complex", "Very Complex"]
        
        for i in range(num_samples):
            # Generate ticket data
            priority = np.random.choice(priorities, p=[0.3, 0.4, 0.2, 0.1])
            category = np.random.choice(categories)
            customer_tier = np.random.choice(customer_tiers, p=[0.4, 0.3, 0.2, 0.1])
            complexity = np.random.choice(complexities)
            
            # SLA hours based on priority and customer tier
            base_sla = {"Low": 72, "Medium": 24, "High": 8, "Critical": 2}[priority]
            tier_multiplier = {"Bronze": 1.0, "Silver": 0.8, "Gold": 0.6, "Platinum": 0.4}[customer_tier]
            sla_hours = base_sla * tier_multiplier
            
            # Resolution time (influenced by multiple factors)
            complexity_factor = {"Simple": 0.5, "Moderate": 1.0, "Complex": 1.5, "Very Complex": 2.0}[complexity]
            category_factor = {"Technical": 1.2, "Billing": 0.8, "Account": 0.6, "Network": 1.5, "Security": 1.0}[category]
            
            # Add some randomness
            resolution_time = sla_hours * complexity_factor * category_factor * np.random.uniform(0.3, 2.0)
            
            # Determine breach
            breach = resolution_time > sla_hours
            
            # Additional features
            created_hour = np.random.randint(0, 24)
            is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
            agent_experience = np.random.uniform(0, 5)  # years
            customer_history = np.random.randint(0, 100)  # previous tickets
            
            ticket = {
                "ticket_id": f"TKT{i:06d}",
                "priority": priority,
                "category": category,
                "customer_tier": customer_tier,
                "complexity": complexity,
                "created_hour": created_hour,
                "is_weekend": is_weekend,
                "agent_experience": agent_experience,
                "customer_history": customer_history,
                "sla_hours": sla_hours,
                "resolution_time": resolution_time,
                "breach": int(breach)
            }
            
            data.append(ticket)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """
        Preprocess data for ML training
        """
        df_processed = df.copy()
        
        # Identify categorical columns
        categorical_columns = ["priority", "category", "customer_tier", "complexity"]
        numerical_columns = ["created_hour", "is_weekend", "agent_experience", "customer_history", "sla_hours"]
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in df_processed.columns:
                if fit_encoders:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_processed[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    if col in self.label_encoders:
                        df_processed[f"{col}_encoded"] = self.label_encoders[col].transform(df_processed[col])
                    else:
                        # Handle unseen categories
                        df_processed[f"{col}_encoded"] = 0
        
        # Select feature columns
        encoded_categorical = [f"{col}_encoded" for col in categorical_columns]
        self.feature_columns = encoded_categorical + numerical_columns
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        return df_processed
    
    def train_model(self, training_data: Optional[pd.DataFrame] = None, 
                   validation_split: float = 0.2) -> Dict:
        """
        Train the SLA breach prediction model
        """
        logger.info("Starting SLA breach model training...")
        
        # Generate synthetic data if not provided
        if training_data is None:
            training_data = self.generate_synthetic_data(num_samples=2000)
        
        # Preprocess data
        df_processed = self.preprocess_data(training_data, fit_encoders=True)
        
        # Prepare features and target
        X = df_processed[self.feature_columns]
        y = df_processed[self.target_column]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = (y_train_pred == y_train).mean()
        val_accuracy = (y_val_pred == y_val).mean()
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            feature_importance = {}
        
        training_results = {
            "model_type": self.model_type,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "validation_auc": val_auc,
            "feature_importance": feature_importance,
            "classification_report": classification_report(y_val, y_val_pred, output_dict=True)
        }
        
        logger.info(f"Model training completed. Validation accuracy: {val_accuracy:.3f}, AUC: {val_auc:.3f}")
        
        return training_results
    
    def predict_breach_risk(self, ticket_data: Dict) -> Dict:
        """
        Predict SLA breach risk for a single ticket
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([ticket_data])
            
            # Preprocess
            df_processed = self.preprocess_data(df, fit_encoders=False)
            
            # Prepare features
            X = df_processed[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            breach_probability = self.model.predict_proba(X_scaled)[0, 1]
            breach_prediction = int(breach_probability > 0.5)
            
            # Feature contribution (simplified)
            feature_contributions = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, feature in enumerate(self.feature_columns):
                    feature_contributions[feature] = {
                        "value": float(X.iloc[0, i]),
                        "importance": float(self.model.feature_importances_[i])
                    }
            
            # Risk categorization
            if breach_probability < 0.3:
                risk_level = "Low"
            elif breach_probability < 0.6:
                risk_level = "Medium"
            elif breach_probability < 0.8:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            # Recommendations
            recommendations = self._generate_recommendations(ticket_data, breach_probability, risk_level)
            
            return {
                "ticket_id": ticket_data.get("ticket_id", "Unknown"),
                "breach_probability": float(breach_probability),
                "breach_prediction": breach_prediction,
                "risk_level": risk_level,
                "feature_contributions": feature_contributions,
                "recommendations": recommendations,
                "confidence": min(abs(breach_probability - 0.5) * 2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error predicting breach risk: {e}")
            return {
                "error": str(e),
                "breach_probability": 0.5,
                "risk_level": "Unknown"
            }
    
    def _generate_recommendations(self, ticket_data: Dict, probability: float, risk_level: str) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level in ["High", "Critical"]:
            recommendations.append("🚨 Immediate attention required - High breach risk")
            
            # Priority-based recommendations
            if ticket_data.get("priority") == "Critical":
                recommendations.append("Escalate to senior support team immediately")
            elif ticket_data.get("priority") == "High":
                recommendations.append("Assign to experienced agent")
            
            # Complexity-based recommendations
            if ticket_data.get("complexity") in ["Complex", "Very Complex"]:
                recommendations.append("Consider splitting into sub-tasks")
                recommendations.append("Schedule expert consultation")
            
            # Time-based recommendations
            if ticket_data.get("is_weekend", 0):
                recommendations.append("Weekend deployment - Ensure on-call availability")
            
            # Customer tier recommendations
            if ticket_data.get("customer_tier") in ["Gold", "Platinum"]:
                recommendations.append("Priority handling for premium customer")
                recommendations.append("Proactive customer communication")
        
        elif risk_level == "Medium":
            recommendations.append("⚠️ Monitor closely - Medium breach risk")
            
            if ticket_data.get("agent_experience", 0) < 2:
                recommendations.append("Consider pairing with senior agent")
            
            if ticket_data.get("customer_history", 0) < 10:
                recommendations.append("New customer - Provide extra attention")
        
        else:  # Low risk
            recommendations.append("✅ Standard processing - Low breach risk")
        
        # General recommendations
        if probability > 0.7:
            recommendations.append("Set up automated monitoring alerts")
            recommendations.append("Prepare contingency plans")
        
        return recommendations
    
    def batch_predict(self, tickets: List[Dict]) -> Dict:
        """
        Predict breach risk for multiple tickets
        """
        results = {
            "total_tickets": len(tickets),
            "predictions": [],
            "summary": {}
        }
        
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for ticket in tickets:
            prediction = self.predict_breach_risk(ticket)
            results["predictions"].append(prediction)
            
            # Count risk levels
            risk_level = prediction.get("risk_level", "Unknown")
            if risk_level in ["High", "Critical"]:
                high_risk_count += 1
            elif risk_level == "Medium":
                medium_risk_count += 1
            else:
                low_risk_count += 1
        
        # Generate summary
        results["summary"] = {
            "high_risk_tickets": high_risk_count,
            "medium_risk_tickets": medium_risk_count,
            "low_risk_tickets": low_risk_count,
            "high_risk_percentage": (high_risk_count / len(tickets)) * 100,
            "average_breach_probability": np.mean([p.get("breach_probability", 0) for p in results["predictions"]])
        }
        
        return results
    
    def reroute_tasks(self, predictions: List[Dict]) -> Dict:
        """
        Autonomous task rerouting based on breach predictions
        """
        rerouting_plan = {
            "escalated_tickets": [],
            "reassigned_tickets": [],
            "monitored_tickets": [],
            "standard_processing": []
        }
        
        for prediction in predictions:
            risk_level = prediction.get("risk_level", "Unknown")
            ticket_id = prediction.get("ticket_id", "Unknown")
            
            if risk_level == "Critical":
                # Immediate escalation
                rerouting_plan["escalated_tickets"].append({
                    "ticket_id": ticket_id,
                    "action": "Escalate to senior team",
                    "priority": "Critical",
                    "reason": "Critical breach risk"
                })
            elif risk_level == "High":
                # Reassign to experienced agent
                rerouting_plan["reassigned_tickets"].append({
                    "ticket_id": ticket_id,
                    "action": "Reassign to experienced agent",
                    "priority": "High",
                    "reason": "High breach risk"
                })
            elif risk_level == "Medium":
                # Monitor closely
                rerouting_plan["monitored_tickets"].append({
                    "ticket_id": ticket_id,
                    "action": "Monitor closely",
                    "priority": "Medium",
                    "reason": "Medium breach risk"
                })
            else:
                # Standard processing
                rerouting_plan["standard_processing"].append({
                    "ticket_id": ticket_id,
                    "action": "Standard processing",
                    "priority": "Normal",
                    "reason": "Low breach risk"
                })
        
        return rerouting_plan
    
    def save_model(self, model_path: str):
        """Save trained model and preprocessing objects"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and preprocessing objects"""
        model_data = joblib.load(model_path)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoders = model_data["label_encoders"]
        self.feature_columns = model_data["feature_columns"]
        self.model_type = model_data["model_type"]
        
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data"""
        # Preprocess
        df_processed = self.preprocess_data(test_data, fit_encoders=False)
        
        # Prepare features
        X = df_processed[self.feature_columns]
        y = df_processed[self.target_column]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y).mean()
        auc = roc_auc_score(y, y_proba)
        
        return {
            "accuracy": float(accuracy),
            "auc_score": float(auc),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }
    
    def generate_performance_report(self, evaluation_results: Dict) -> str:
        """Generate performance report"""
        report_lines = [
            "SLA Breach Prediction Model Performance Report",
            "=" * 50,
            f"Model Type: {self.model_type}",
            f"Accuracy: {evaluation_results['accuracy']:.3f}",
            f"AUC Score: {evaluation_results['auc_score']:.3f}",
            "",
            "Classification Report:",
            "-" * 30
        ]
        
        # Add classification report details
        class_report = evaluation_results["classification_report"]
        for class_name in ["0", "1"]:
            if class_name in class_report:
                metrics = class_report[class_name]
                report_lines.append(f"Class {class_name}:")
                report_lines.append(f"  Precision: {metrics['precision']:.3f}")
                report_lines.append(f"  Recall: {metrics['recall']:.3f}")
                report_lines.append(f"  F1-Score: {metrics['f1-score']:.3f}")
        
        return "\n".join(report_lines)
