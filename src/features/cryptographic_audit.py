"""
7. Zero-Cost Cryptographic Audit Trail
Secure automated decision logs using immutable ledger database (local implementation)
"""

import logging
import json
import hashlib
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import hmac
import secrets

logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
    """Audit log entry"""
    timestamp: str
    action: str
    user_id: str
    resource: str
    details: Dict[str, Any]
    hash: str = ""
    previous_hash: str = ""
    signature: str = ""

class CryptographicAuditTrail:
    """
    Zero-Cost Cryptographic Audit Trail
    Implements immutable ledger using local SQLite with cryptographic verification
    """
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        # Generate or load master key
        self.master_key = self._get_or_create_master_key()
        
        logger.info("Cryptographic Audit Trail initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for audit trail"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create audit table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        action TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        details TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        previous_hash TEXT,
                        signature TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_entries(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_entries(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_action ON audit_entries(action)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_resource ON audit_entries(resource)')
                
                # Create metadata table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _get_or_create_master_key(self) -> str:
        """Get or create master key for signing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if master key exists
                cursor.execute("SELECT value FROM audit_metadata WHERE key = 'master_key'")
                result = cursor.fetchone()
                
                if result:
                    return result[0]
                else:
                    # Generate new master key
                    master_key = secrets.token_hex(32)
                    cursor.execute(
                        "INSERT INTO audit_metadata (key, value) VALUES (?, ?)",
                        ('master_key', master_key)
                    )
                    conn.commit()
                    return master_key
                    
        except Exception as e:
            logger.error(f"Error managing master key: {e}")
            # Fallback to a default key (not recommended for production)
            return "fallback_master_key_32_chars_long"
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _calculate_entry_hash(self, entry: AuditEntry) -> str:
        """Calculate hash for audit entry"""
        # Create canonical representation
        canonical_data = json.dumps({
            "timestamp": entry.timestamp,
            "action": entry.action,
            "user_id": entry.user_id,
            "resource": entry.resource,
            "details": entry.details,
            "previous_hash": entry.previous_hash
        }, sort_keys=True, separators=(',', ':'))
        
        return self._calculate_hash(canonical_data)
    
    def _sign_entry(self, entry_hash: str) -> str:
        """Sign entry hash using HMAC with master key"""
        return hmac.new(
            self.master_key.encode(),
            entry_hash.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _verify_signature(self, entry_hash: str, signature: str) -> bool:
        """Verify entry signature"""
        expected_signature = self._sign_entry(entry_hash)
        return hmac.compare_digest(signature, expected_signature)
    
    def _get_previous_hash(self) -> str:
        """Get hash of the most recent entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT hash FROM audit_entries ORDER BY id DESC LIMIT 1"
                )
                result = cursor.fetchone()
                return result[0] if result else "genesis_hash"
                
        except Exception as e:
            logger.error(f"Error getting previous hash: {e}")
            return "genesis_hash"
    
    def log_action(self, action: str, user_id: str, resource: str, 
                   details: Dict[str, Any]) -> str:
        """
        Log an action to the audit trail
        
        Returns:
            str: Entry ID
        """
        with self.lock:
            try:
                # Get previous hash
                previous_hash = self._get_previous_hash()
                
                # Create audit entry
                entry = AuditEntry(
                    timestamp=datetime.now().isoformat(),
                    action=action,
                    user_id=user_id,
                    resource=resource,
                    details=details,
                    previous_hash=previous_hash
                )
                
                # Calculate entry hash
                entry.hash = self._calculate_entry_hash(entry)
                
                # Sign entry
                entry.signature = self._sign_entry(entry.hash)
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO audit_entries 
                        (timestamp, action, user_id, resource, details, hash, previous_hash, signature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.timestamp,
                        entry.action,
                        entry.user_id,
                        entry.resource,
                        json.dumps(entry.details),
                        entry.hash,
                        entry.previous_hash,
                        entry.signature
                    ))
                    
                    entry_id = cursor.lastrowid
                    conn.commit()
                
                logger.info(f"Audit entry logged: {entry_id} - {action}")
                return str(entry_id)
                
            except Exception as e:
                logger.error(f"Error logging action: {e}")
                raise
    
    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get specific audit entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM audit_entries WHERE id = ?",
                    (int(entry_id),)
                )
                row = cursor.fetchone()
                
                if row:
                    return AuditEntry(
                        timestamp=row[1],
                        action=row[2],
                        user_id=row[3],
                        resource=row[4],
                        details=json.loads(row[5]),
                        hash=row[6],
                        previous_hash=row[7],
                        signature=row[8]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting entry {entry_id}: {e}")
            return None
    
    def get_entries_by_user(self, user_id: str, limit: int = 100) -> List[AuditEntry]:
        """Get audit entries for a specific user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM audit_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit)
                )
                
                entries = []
                for row in cursor.fetchall():
                    entries.append(AuditEntry(
                        timestamp=row[1],
                        action=row[2],
                        user_id=row[3],
                        resource=row[4],
                        details=json.loads(row[5]),
                        hash=row[6],
                        previous_hash=row[7],
                        signature=row[8]
                    ))
                
                return entries
                
        except Exception as e:
            logger.error(f"Error getting entries for user {user_id}: {e}")
            return []
    
    def get_entries_by_resource(self, resource: str, limit: int = 100) -> List[AuditEntry]:
        """Get audit entries for a specific resource"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM audit_entries WHERE resource = ? ORDER BY timestamp DESC LIMIT ?",
                    (resource, limit)
                )
                
                entries = []
                for row in cursor.fetchall():
                    entries.append(AuditEntry(
                        timestamp=row[1],
                        action=row[2],
                        user_id=row[3],
                        resource=row[4],
                        details=json.loads(row[5]),
                        hash=row[6],
                        previous_hash=row[7],
                        signature=row[8]
                    ))
                
                return entries
                
        except Exception as e:
            logger.error(f"Error getting entries for resource {resource}: {e}")
            return []
    
    def get_entries_by_timerange(self, start_time: str, end_time: str) -> List[AuditEntry]:
        """Get audit entries within a time range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM audit_entries WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                    (start_time, end_time)
                )
                
                entries = []
                for row in cursor.fetchall():
                    entries.append(AuditEntry(
                        timestamp=row[1],
                        action=row[2],
                        user_id=row[3],
                        resource=row[4],
                        details=json.loads(row[5]),
                        hash=row[6],
                        previous_hash=row[7],
                        signature=row[8]
                    ))
                
                return entries
                
        except Exception as e:
            logger.error(f"Error getting entries by time range: {e}")
            return []
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire audit chain
        
        Returns:
            Dict with verification results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM audit_entries ORDER BY id")
                
                entries = cursor.fetchall()
                verification_result = {
                    "total_entries": len(entries),
                    "valid_entries": 0,
                    "invalid_entries": 0,
                    "chain_broken": False,
                    "issues": []
                }
                
                previous_hash = "genesis_hash"
                
                for i, row in enumerate(entries):
                    entry_id = row[0]
                    timestamp = row[1]
                    action = row[2]
                    user_id = row[3]
                    resource = row[4]
                    details = json.loads(row[5])
                    stored_hash = row[6]
                    stored_previous_hash = row[7]
                    signature = row[8]
                    
                    # Recreate entry
                    entry = AuditEntry(
                        timestamp=timestamp,
                        action=action,
                        user_id=user_id,
                        resource=resource,
                        details=details,
                        previous_hash=stored_previous_hash
                    )
                    
                    # Calculate expected hash
                    expected_hash = self._calculate_entry_hash(entry)
                    
                    # Verify hash
                    if stored_hash != expected_hash:
                        verification_result["invalid_entries"] += 1
                        verification_result["issues"].append(f"Entry {entry_id}: Hash mismatch")
                        continue
                    
                    # Verify previous hash link
                    if stored_previous_hash != previous_hash:
                        verification_result["chain_broken"] = True
                        verification_result["issues"].append(f"Entry {entry_id}: Chain broken at previous hash")
                    
                    # Verify signature
                    if not self._verify_signature(stored_hash, signature):
                        verification_result["invalid_entries"] += 1
                        verification_result["issues"].append(f"Entry {entry_id}: Invalid signature")
                        continue
                    
                    verification_result["valid_entries"] += 1
                    previous_hash = stored_hash
                
                verification_result["integrity_score"] = (
                    verification_result["valid_entries"] / verification_result["total_entries"]
                    if verification_result["total_entries"] > 0 else 0
                )
                
                return verification_result
                
        except Exception as e:
            logger.error(f"Error verifying chain integrity: {e}")
            return {
                "error": str(e),
                "integrity_score": 0.0
            }
    
    def export_audit_trail(self, output_path: str, format: str = "json"):
        """Export audit trail to file"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM audit_entries ORDER BY timestamp")
                
                entries = []
                for row in cursor.fetchall():
                    entry = {
                        "id": row[0],
                        "timestamp": row[1],
                        "action": row[2],
                        "user_id": row[3],
                        "resource": row[4],
                        "details": json.loads(row[5]),
                        "hash": row[6],
                        "previous_hash": row[7],
                        "signature": row[8]
                    }
                    entries.append(entry)
                
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_entries": len(entries),
                    "entries": entries
                }
                
                if format.lower() == "json":
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                elif format.lower() == "csv":
                    # Convert to CSV format
                    import pandas as pd
                    df = pd.DataFrame(entries)
                    df.to_csv(output_path, index=False)
                
                logger.info(f"Audit trail exported to {output_path}")
                
        except Exception as e:
            logger.error(f"Error exporting audit trail: {e}")
            raise
    
    def search_audit_trail(self, query: str, search_fields: List[str] = None) -> List[AuditEntry]:
        """Search audit trail for specific terms"""
        if search_fields is None:
            search_fields = ["action", "user_id", "resource"]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build search query
                conditions = []
                params = []
                
                for field in search_fields:
                    conditions.append(f"{field} LIKE ?")
                    params.append(f"%{query}%")
                
                # Also search in details JSON
                conditions.append("details LIKE ?")
                params.append(f"%{query}%")
                
                sql = f"SELECT * FROM audit_entries WHERE {' OR '.join(conditions)} ORDER BY timestamp DESC"
                
                cursor.execute(sql, params)
                
                entries = []
                for row in cursor.fetchall():
                    entries.append(AuditEntry(
                        timestamp=row[1],
                        action=row[2],
                        user_id=row[3],
                        resource=row[4],
                        details=json.loads(row[5]),
                        hash=row[6],
                        previous_hash=row[7],
                        signature=row[8]
                    ))
                
                return entries
                
        except Exception as e:
            logger.error(f"Error searching audit trail: {e}")
            return []
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total entries
                cursor.execute("SELECT COUNT(*) FROM audit_entries")
                total_entries = cursor.fetchone()[0]
                
                # Entries by action
                cursor.execute("""
                    SELECT action, COUNT(*) 
                    FROM audit_entries 
                    GROUP BY action 
                    ORDER BY COUNT(*) DESC
                """)
                actions = dict(cursor.fetchall())
                
                # Entries by user
                cursor.execute("""
                    SELECT user_id, COUNT(*) 
                    FROM audit_entries 
                    GROUP BY user_id 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                """)
                top_users = dict(cursor.fetchall())
                
                # Entries by resource
                cursor.execute("""
                    SELECT resource, COUNT(*) 
                    FROM audit_entries 
                    GROUP BY resource 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                """)
                top_resources = dict(cursor.fetchall())
                
                # Date range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM audit_entries")
                date_range = cursor.fetchone()
                
                return {
                    "total_entries": total_entries,
                    "actions": actions,
                    "top_users": top_users,
                    "top_resources": top_resources,
                    "date_range": {
                        "earliest": date_range[0],
                        "latest": date_range[1]
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting audit statistics: {e}")
            return {}
    
    def generate_integrity_report(self) -> str:
        """Generate integrity verification report"""
        verification = self.verify_chain_integrity()
        
        report_lines = [
            "Audit Trail Integrity Report",
            "=" * 40,
            f"Total Entries: {verification['total_entries']}",
            f"Valid Entries: {verification['valid_entries']}",
            f"Invalid Entries: {verification['invalid_entries']}",
            f"Integrity Score: {verification['integrity_score']:.3f}",
            f"Chain Integrity: {'✓ Intact' if not verification['chain_broken'] else '✗ Broken'}",
            "",
            "Issues Found:",
            "-" * 15
        ]
        
        if verification['issues']:
            for issue in verification['issues'][:10]:  # Limit to first 10 issues
                report_lines.append(f"• {issue}")
        else:
            report_lines.append("No issues found. Audit trail is fully intact.")
        
        return "\n".join(report_lines)
    
    def create_backup(self, backup_path: str):
        """Create backup of audit trail database"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Audit trail backup created at {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
