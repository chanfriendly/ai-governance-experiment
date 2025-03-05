# src/agents/conversation.py
import json
import time
from typing import Dict, List, Any, Optional
from .communication import Message, MessageType

class Conversation:
    """Controls the flow of conversation between agents."""
    
    def __init__(self, scenario: str, agents: Dict[str, Any]):
        self.scenario = scenario
        self.agents = agents
        self.messages = []
        self.current_turn = 0
        self.max_turns = 10  # Default limit to prevent endless conversations
        self.active = True
        self.outcome = None
    
    def start(self, first_agent_name: Optional[str] = None) -> Message:
        """
        Start the conversation by presenting the scenario to the first agent.
        
        Args:
            first_agent_name: Name of agent to start. If None, the first agent in the dictionary is used.
            
        Returns:
            The first message in the conversation.
        """
        # Determine first agent
        if first_agent_name and first_agent_name in self.agents:
            agent_name = first_agent_name
        else:
            agent_name = list(self.agents.keys())[0]
        
        agent = self.agents[agent_name]
        
        # Have the first agent respond to the scenario
        system_message = Message(
            sender="System",
            recipient=agent_name,
            content=self.scenario,
            message_type=MessageType.STATEMENT
        )
        
        # Add system message to history
        self.messages.append(system_message)
        
        # Generate agent response
        response = agent.respond(self.scenario)
        
        # Create message from agent response
        agent_message = Message(
            sender=agent_name,
            recipient="all",  # Initial message is to all participants
            content=response["decision"],
            message_type=MessageType.STATEMENT,
            reasoning=response["reasoning"]
        )
        
        # Add agent message to history
        self.messages.append(agent_message)
        self.current_turn += 1
        
        return agent_message
    
    def next_turn(self, next_agent_name: Optional[str] = None) -> Optional[Message]:
        """
        Advance to the next turn in the conversation.
        
        Args:
            next_agent_name: Name of the next agent to speak. If None, the next agent is determined automatically.
            
        Returns:
            The next message in the conversation or None if conversation is complete.
        """
        if not self.active or self.current_turn >= self.max_turns:
            return None
        
        # Determine next agent
        if next_agent_name and next_agent_name in self.agents:
            agent_name = next_agent_name
        else:
            # Simple round-robin agent selection
            agent_names = list(self.agents.keys())
            agent_name = agent_names[self.current_turn % len(agent_names)]
        
        agent = self.agents[agent_name]
        
        # Get the most recent message not from this agent
        other_messages = [m for m in self.messages if m.sender != agent_name and m.sender != "System"]
        if not other_messages:
            return None
        
        last_message = other_messages[-1]
        
        # Format the conversation history for this agent
        history = [
            {"role": m.sender, "content": m.content}
            for m in self.messages
        ]
        
        # Have the agent respond to the last message
        response = agent.respond(
            scenario=last_message.content,
            conversation_history=history
        )
        
        # Create message from agent response
        agent_message = Message(
            sender=agent_name,
            recipient=last_message.sender,
            content=response["decision"],
            message_type=MessageType.STATEMENT,
            references=[last_message.id],
            reasoning=response["reasoning"]
        )
        
        # Add agent message to history
        self.messages.append(agent_message)
        self.current_turn += 1
        
        return agent_message
    
    def run_dialogue(self, turns: int = 4) -> List[Message]:
        """
        Run a dialogue for a specified number of turns.
        
        Args:
            turns: Number of message exchanges to run
            
        Returns:
            List of all messages generated during the dialogue
        """
        # Start the conversation if it hasn't already started
        if not self.messages:
            self.start()
        
        # Run specified number of turns
        remaining_turns = min(turns, self.max_turns - self.current_turn)
        for _ in range(remaining_turns):
            message = self.next_turn()
            if not message or not self.active:
                break
        
        return self.messages
    
    def save_transcript(self, filepath: str) -> None:
        """
        Save the conversation transcript to a file.
        
        Args:
            filepath: Path to save the transcript
        """
        summary = {
            "scenario": self.scenario,
            "agents": list(self.agents.keys()),
            "turns": self.current_turn,
            "message_count": len(self.messages),
            "active": self.active,
            "messages": [m.to_dict() for m in self.messages]
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)