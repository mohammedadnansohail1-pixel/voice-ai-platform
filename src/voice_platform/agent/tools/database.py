"""
SQLite database for appointment persistence.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from ...logging import get_logger

logger = get_logger("agent.database")


class AppointmentDatabase:
    """SQLite database for storing appointments."""
    
    def __init__(self, db_path: str = "data/appointments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("database_initialized", path=str(self.db_path))
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                confirmation_number TEXT UNIQUE NOT NULL,
                reason TEXT NOT NULL,
                day TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT DEFAULT 'confirmed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                appointment_id INTEGER,
                transcript TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_appointment(
        self,
        confirmation_number: str,
        reason: str,
        day: str,
        time: str,
    ) -> Dict[str, Any]:
        """Save appointment to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO appointments (confirmation_number, reason, day, time)
            VALUES (?, ?, ?, ?)
        """, (confirmation_number, reason, day, time))
        
        appointment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(
            "appointment_saved",
            id=appointment_id,
            confirmation=confirmation_number,
        )
        
        return {
            "id": appointment_id,
            "confirmation_number": confirmation_number,
            "reason": reason,
            "day": day,
            "time": time,
        }
    
    def get_appointment(self, confirmation_number: str) -> Optional[Dict[str, Any]]:
        """Get appointment by confirmation number."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM appointments WHERE confirmation_number = ?",
            (confirmation_number,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def list_appointments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent appointments."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM appointments ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_next_confirmation_number(self) -> str:
        """Generate next confirmation number."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM appointments")
        count = cursor.fetchone()[0]
        conn.close()
        
        return f"APT-{count + 1001}"
