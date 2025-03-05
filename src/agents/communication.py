# src/agents/communication.py
import uuid
import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

class MessageType(Enum):
    """Types of messages agents can exchange."""
    STATEMENT = "statement"       # Presenting a position or information
    QUESTION = "question"         # Asking for clarification or more information
    PROPOSAL = "proposal"         # Suggesting a solution or action
    CRITIQUE = "critique"         # Evaluating another agent's message
    AGREEMENT = "agreement"       # Expressing consensus
    DISAGREEMENT = "disagreement" # Expressing dissent
    REASONING = "reasoning"       # Explaining thought process

class Message:
    """A message passed between agents in a conversation."""
    
    def __init__(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: MessageType,
        references: Optional[List[str]] = None,
        reasoning: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now().isoformat()
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type
        self.references = references or []  # IDs of messages this one references
        self.reasoning = reasoning  # Optional internal reasoning (not shared directly)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type.value,
            "references": self.references,
            "reasoning": self.reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message instance from dictionary."""
        message = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            references=data.get("references", []),
            reasoning=data.get("reasoning")
        )
        message.id = data["id"]
        message.timestamp = data["timestamp"]
        return message
    
    def __str__(self) -> str:
        """Human-readable representation of the message."""
        return f"{self.sender} to {self.recipient} ({self.message_type.value}): {self.content[:50]}..."