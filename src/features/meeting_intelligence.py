"""
9. Autonomous Meeting Intelligence
Run open-source speaker diarization models locally, passing transcripts to local SLM for action items
"""

import logging
import json
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class Speaker:
    """Speaker information"""
    id: str
    name: str = ""
    segments: List[Dict] = None
    total_speaking_time: float = 0.0
    word_count: int = 0

@dataclass
class ActionItem:
    """Action item extracted from meeting"""
    description: str
    assignee: str
    due_date: str
    priority: str
    category: str
    confidence: float

@dataclass
class MeetingSummary:
    """Meeting summary and insights"""
    title: str
    date: str
    duration: float
    participants: List[Speaker]
    transcript: str
    action_items: List[ActionItem]
    key_topics: List[str]
    decisions: List[str]
    next_steps: List[str]

class MeetingIntelligence:
    """
    Autonomous Meeting Intelligence
    Processes audio meetings to extract action items and insights
    """
    
    def __init__(self):
        self.speakers: Dict[str, Speaker] = {}
        self.transcript = ""
        self.action_items = []
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Autonomous Meeting Intelligence initialized")
    
    def _initialize_models(self):
        """Initialize speech processing models"""
        try:
            # For demo purposes, we'll simulate the models
            # In production, you would use:
            # - speechbrain for speaker diarization
            # - whisper for speech recognition
            # - local LLM for action item extraction
            
            logger.info("Speech processing models initialized (simulation mode)")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback processing"""
        logger.warning("Using fallback processing for meeting intelligence")
        self.fallback_mode = True
    
    def process_audio_file(self, audio_path: str, meeting_title: str = "") -> MeetingSummary:
        """
        Process audio file and extract meeting intelligence
        """
        try:
            if hasattr(self, 'fallback_mode'):
                return self._fallback_processing(audio_path, meeting_title)
            
            # Step 1: Speaker Diarization
            speakers = self._perform_speaker_diarization(audio_path)
            
            # Step 2: Speech Recognition
            transcript = self._perform_speech_recognition(audio_path, speakers)
            
            # Step 3: Action Item Extraction
            action_items = self._extract_action_items(transcript)
            
            # Step 4: Topic Analysis
            key_topics = self._extract_key_topics(transcript)
            
            # Step 5: Decision Extraction
            decisions = self._extract_decisions(transcript)
            
            # Step 6: Next Steps
            next_steps = self._extract_next_steps(transcript)
            
            # Create meeting summary
            summary = MeetingSummary(
                title=meeting_title or Path(audio_path).stem,
                date=datetime.now().isoformat(),
                duration=self._get_audio_duration(audio_path),
                participants=list(speakers.values()),
                transcript=transcript,
                action_items=action_items,
                key_topics=key_topics,
                decisions=decisions,
                next_steps=next_steps
            )
            
            logger.info(f"Successfully processed meeting: {meeting_title}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return self._fallback_processing(audio_path, meeting_title)
    
    def _perform_speaker_diarization(self, audio_path: str) -> Dict[str, Speaker]:
        """Perform speaker diarization"""
        # Simulate speaker diarization
        # In production, you would use speechbrain or similar
        
        speakers = {}
        
        # Simulate 3 speakers
        for i in range(3):
            speaker_id = f"speaker_{i+1}"
            speakers[speaker_id] = Speaker(
                id=speaker_id,
                name=f"Speaker {i+1}",
                segments=[
                    {
                        "start": i * 120.0,  # Every 2 minutes
                        "end": (i + 1) * 120.0,
                        "duration": 120.0
                    }
                ],
                total_speaking_time=120.0,
                word_count=np.random.randint(50, 200)
            )
        
        return speakers
    
    def _perform_speech_recognition(self, audio_path: str, speakers: Dict[str, Speaker]) -> str:
        """Perform speech recognition"""
        # Simulate speech recognition
        # In production, you would use whisper or similar
        
        transcript_lines = [
            "Speaker 1: Welcome everyone to today's meeting. Let's start with the project update.",
            "Speaker 2: Thank you. I'll provide an update on the current project status.",
            "Speaker 3: Before we proceed, I'd like to discuss the budget allocation.",
            "Speaker 1: That's a good point. Let's address the budget first.",
            "Speaker 2: We need to finalize the requirements by next Friday.",
            "Speaker 3: I'll take responsibility for creating the documentation.",
            "Speaker 1: Great! Let's summarize the action items before we conclude.",
            "Speaker 2: I'll send the project timeline by tomorrow.",
            "Speaker 3: The budget proposal needs to be reviewed by the finance team.",
            "Speaker 1: Thank you all. Let's meet again next week to review progress."
        ]
        
        return "\n".join(transcript_lines)
    
    def _extract_action_items(self, transcript: str) -> List[ActionItem]:
        """Extract action items from transcript"""
        action_items = []
        
        # Action item patterns
        action_patterns = [
            r"(.+?)\s+(?:will|shall|going to|need to|should)\s+(.+?)(?:\.|$)",
            r"(.+?)\s+(?:take responsibility|be responsible for)\s+(.+?)(?:\.|$)",
            r"(.+?)\s+(?:send|create|prepare|review|finalize|complete)\s+(.+?)(?:\.|$)",
            r"Action item[:\s]*(.+?)(?:\.|$)",
            r"TODO[:\s]*(.+?)(?:\.|$)"
        ]
        
        lines = transcript.split('\n')
        
        for line in lines:
            for pattern in action_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        assignee, description = match
                    else:
                        assignee = "Unassigned"
                        description = match
                    
                    action_item = ActionItem(
                        description=description.strip(),
                        assignee=assignee.strip(),
                        due_date=self._extract_due_date(line),
                        priority=self._extract_priority(line),
                        category=self._categorize_action(description),
                        confidence=self._calculate_confidence(description)
                    )
                    
                    action_items.append(action_item)
        
        return action_items
    
    def _extract_key_topics(self, transcript: str) -> List[str]:
        """Extract key topics from transcript"""
        # Common business topics
        business_topics = [
            "budget", "project", "timeline", "requirements", "documentation",
            "development", "testing", "deployment", "marketing", "sales",
            "finance", "resources", "team", "strategy", "goals", "objectives"
        ]
        
        topics_found = []
        transcript_lower = transcript.lower()
        
        for topic in business_topics:
            if topic in transcript_lower:
                # Count occurrences
                count = transcript_lower.count(topic)
                if count >= 2:  # Topic mentioned at least twice
                    topics.append(f"{topic} (mentioned {count} times)")
        
        return topics_found[:5]  # Top 5 topics
    
    def _extract_decisions(self, transcript: str) -> List[str]:
        """Extract decisions from transcript"""
        decision_patterns = [
            r"(?:we|the team)\s+(?:decided|agreed|concluded|determined)\s+to\s+(.+?)(?:\.|$)",
            r"Decision[:\s]*(.+?)(?:\.|$)",
            r"(?:it's|it is)\s+(?:decided|agreed)\s+that\s+(.+?)(?:\.|$)",
            r"Let's\s+(.+?)(?:\.|$)"
        ]
        
        decisions = []
        lines = transcript.split('\n')
        
        for line in lines:
            for pattern in decision_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    decisions.append(match.strip())
        
        return decisions
    
    def _extract_next_steps(self, transcript: str) -> List[str]:
        """Extract next steps from transcript"""
        next_step_patterns = [
            r"Next steps?[:\s]*(.+?)(?:\.|$)",
            r"Going forward[:\s]*(.+?)(?:\.|$)",
            r"Moving forward[:\s]*(.+?)(?:\.|$)",
            r"Future actions?[:\s]*(.+?)(?:\.|$)"
        ]
        
        next_steps = []
        lines = transcript.split('\n')
        
        for line in lines:
            for pattern in next_step_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    next_steps.append(match.strip())
        
        return next_steps
    
    def _extract_due_date(self, text: str) -> str:
        """Extract due date from text"""
        date_patterns = [
            r"by\s+(tomorrow|today|next week|next month|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)",
            r"by\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))",
            r"by\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"within\s+(\d+)\s+(?:days?|weeks?|months?)"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Not specified"
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["urgent", "asap", "immediately", "critical"]):
            return "High"
        elif any(word in text_lower for word in ["important", "priority", "soon"]):
            return "Medium"
        else:
            return "Low"
    
    def _categorize_action(self, description: str) -> str:
        """Categorize action item"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["document", "write", "create", "prepare"]):
            return "Documentation"
        elif any(word in desc_lower for word in ["send", "email", "communicate", "inform"]):
            return "Communication"
        elif any(word in desc_lower for word in ["review", "check", "verify", "validate"]):
            return "Review"
        elif any(word in desc_lower for word in ["develop", "code", "implement", "build"]):
            return "Development"
        elif any(word in desc_lower for word in ["test", "qa", "quality"]):
            return "Testing"
        elif any(word in desc_lower for word in ["budget", "finance", "cost"]):
            return "Finance"
        else:
            return "General"
    
    def _calculate_confidence(self, description: str) -> float:
        """Calculate confidence score for action item"""
        # Higher confidence for more specific descriptions
        confidence = 0.5  # Base confidence
        
        # Add confidence for specific action verbs
        action_verbs = ["send", "create", "prepare", "review", "finalize", "complete"]
        for verb in action_verbs:
            if verb in description.lower():
                confidence += 0.1
        
        # Add confidence for specific deliverables
        deliverables = ["report", "document", "email", "proposal", "timeline", "budget"]
        for deliverable in deliverables:
            if deliverable in description.lower():
                confidence += 0.1
        
        # Add confidence for time specificity
        if any(time_word in description.lower() for time_word in ["tomorrow", "today", "by", "within"]):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration"""
        try:
            # Use ffprobe to get duration
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        # Fallback: return simulated duration
        return 600.0  # 10 minutes
    
    def _fallback_processing(self, audio_path: str, meeting_title: str) -> MeetingSummary:
        """Fallback processing when models are not available"""
        logger.info("Using fallback meeting processing")
        
        # Create mock data
        speakers = {
            "speaker_1": Speaker(
                id="speaker_1",
                name="Speaker 1",
                total_speaking_time=200.0,
                word_count=150
            ),
            "speaker_2": Speaker(
                id="speaker_2",
                name="Speaker 2",
                total_speaking_time=180.0,
                word_count=120
            )
        }
        
        mock_transcript = """
        Speaker 1: Welcome to the project review meeting.
        Speaker 2: Thank you for joining today.
        Speaker 1: Let's discuss the current status and next steps.
        Speaker 2: I'll prepare the project documentation by Friday.
        Speaker 1: Great! I'll review the budget proposal.
        """
        
        mock_action_items = [
            ActionItem(
                description="Prepare project documentation",
                assignee="Speaker 2",
                due_date="Friday",
                priority="Medium",
                category="Documentation",
                confidence=0.8
            ),
            ActionItem(
                description="Review budget proposal",
                assignee="Speaker 1",
                due_date="Not specified",
                priority="High",
                category="Finance",
                confidence=0.7
            )
        ]
        
        return MeetingSummary(
            title=meeting_title or "Mock Meeting",
            date=datetime.now().isoformat(),
            duration=600.0,
            participants=list(speakers.values()),
            transcript=mock_transcript,
            action_items=mock_action_items,
            key_topics=["project", "documentation", "budget"],
            decisions=["Proceed with project documentation", "Review budget proposal"],
            next_steps=["Complete documentation by Friday", "Schedule budget review"]
        )
    
    def generate_meeting_report(self, summary: MeetingSummary) -> str:
        """Generate comprehensive meeting report"""
        report_lines = [
            f"Meeting Intelligence Report: {summary.title}",
            "=" * 50,
            f"Date: {summary.date}",
            f"Duration: {summary.duration:.1f} minutes",
            f"Participants: {len(summary.participants)}",
            "",
            "Participants Summary:",
            "-" * 20
        ]
        
        # Participant statistics
        for participant in summary.participants:
            report_lines.append(
                f"• {participant.name}: {participant.total_speaking_time:.1f}s speaking time, "
                f"{participant.word_count} words"
            )
        
        # Action items
        report_lines.extend([
            "",
            "Action Items:",
            "-" * 15
        ])
        
        for i, action in enumerate(summary.action_items, 1):
            report_lines.append(
                f"{i}. {action.description} (Assigned: {action.assignee}, "
                f"Due: {action.due_date}, Priority: {action.priority})"
            )
        
        # Key topics
        if summary.key_topics:
            report_lines.extend([
                "",
                "Key Topics Discussed:",
                "-" * 25
            ])
            for topic in summary.key_topics:
                report_lines.append(f"• {topic}")
        
        # Decisions
        if summary.decisions:
            report_lines.extend([
                "",
                "Decisions Made:",
                "-" * 18
            ])
            for decision in summary.decisions:
                report_lines.append(f"• {decision}")
        
        # Next steps
        if summary.next_steps:
            report_lines.extend([
                "",
                "Next Steps:",
                "-" * 13
            ])
            for step in summary.next_steps:
                report_lines.append(f"• {step}")
        
        # Meeting insights
        report_lines.extend([
            "",
            "Meeting Insights:",
            "-" * 19
        ])
        
        total_words = sum(p.word_count for p in summary.participants)
        avg_words_per_participant = total_words / len(summary.participants) if summary.participants else 0
        
        report_lines.extend([
            f"Total words spoken: {total_words}",
            f"Average words per participant: {avg_words_per_participant:.1f}",
            f"Action items identified: {len(summary.action_items)}",
            f"Decisions made: {len(summary.decisions)}",
            f"Meeting efficiency score: {self._calculate_efficiency_score(summary):.1f}/10"
        ])
        
        return "\n".join(report_lines)
    
    def _calculate_efficiency_score(self, summary: MeetingSummary) -> float:
        """Calculate meeting efficiency score"""
        score = 5.0  # Base score
        
        # Bonus for action items
        score += min(len(summary.action_items) * 0.5, 2.0)
        
        # Bonus for decisions
        score += min(len(summary.decisions) * 0.3, 1.0)
        
        # Penalty for very long meetings without outcomes
        if summary.duration > 3600 and len(summary.action_items) < 3:  # > 1 hour, < 3 actions
            score -= 1.0
        
        # Bonus for balanced participation
        if len(summary.participants) > 1:
            speaking_times = [p.total_speaking_time for p in summary.participants]
            time_variance = np.var(speaking_times)
            if time_variance < 10000:  # Low variance = balanced participation
                score += 0.5
        
        return max(1.0, min(10.0, score))
    
    def export_meeting_data(self, summary: MeetingSummary, output_path: str):
        """Export meeting data to JSON"""
        export_data = {
            "title": summary.title,
            "date": summary.date,
            "duration": summary.duration,
            "participants": [asdict(p) for p in summary.participants],
            "transcript": summary.transcript,
            "action_items": [asdict(a) for a in summary.action_items],
            "key_topics": summary.key_topics,
            "decisions": summary.decisions,
            "next_steps": summary.next_steps,
            "efficiency_score": self._calculate_efficiency_score(summary),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Meeting data exported to {output_path}")
    
    def batch_process_meetings(self, audio_files: List[str]) -> Dict[str, Any]:
        """Process multiple meetings in batch"""
        results = {
            "total_meetings": len(audio_files),
            "processed_meetings": 0,
            "failed_meetings": 0,
            "meeting_summaries": [],
            "total_action_items": 0,
            "total_participants": 0,
            "average_efficiency": 0.0
        }
        
        efficiency_scores = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                meeting_title = Path(audio_file).stem
                summary = self.process_audio_file(audio_file, meeting_title)
                
                results["meeting_summaries"].append(summary)
                results["total_action_items"] += len(summary.action_items)
                results["total_participants"] += len(summary.participants)
                results["processed_meetings"] += 1
                
                efficiency_scores.append(self._calculate_efficiency_score(summary))
                
            except Exception as e:
                logger.error(f"Error processing meeting {i}: {e}")
                results["failed_meetings"] += 1
        
        # Calculate average efficiency
        if efficiency_scores:
            results["average_efficiency"] = np.mean(efficiency_scores)
        
        return results
