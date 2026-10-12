import time
import re
from typing import List, Dict, Any, Optional

class ConversationMemory:
    """
    Manages multi-turn conversation sessions and performs contextual
    query condensation for follow-up questions.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, max_history_turns: int = 6):
        self.max_history_turns = max_history_turns
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns for a session."""
        return self.sessions.get(session_id, [])

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        retrieval_meta: Optional[Dict[str, Any]] = None
    ):
        """Append a completed turn to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        turn = {
            "timestamp": time.time(),
            "user": user_query,
            "assistant": assistant_response,
            "citations": citations or [],
            "retrieval_meta": retrieval_meta or {}
        }
        self.sessions[session_id].append(turn)

        # Truncate to maximum turns
        if len(self.sessions[session_id]) > self.max_history_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_turns:]

    def clear_session(self, session_id: str):
        """Clear memory for a given session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def contextualize_query(self, session_id: str, current_query: str) -> str:
        """
        Rewrites a conversational follow-up query containing pronouns
        (e.g. 'what about that?', 'can it be extended?') into a standalone
        search query based on prior conversation context.
        """
        history = self.get_history(session_id)
        if not history:
            return current_query

        # Check if current query needs contextualization
        # Only rewrite if query contains pronouns/anaphora or conversational follow-up connectors
        pronoun_pattern = re.compile(
            r"\b(it|its|they|them|that|those|these|the same|and then|what about|how about|can it|does it|is that)\b",
            re.IGNORECASE
        )
        has_pronoun = bool(pronoun_pattern.search(current_query))
        starts_with_followup = bool(re.match(r"^(what about|how about|and\b|can it\b|does it\b|why\?|why so\?)", current_query.strip(), re.IGNORECASE))

        if not (has_pronoun or starts_with_followup):
            return current_query

        # Extract primary subject from last user question or assistant response
        last_turn = history[-1]
        last_user = last_turn["user"]
        
        # Extract main nouns/entities from last question
        words = re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", last_user)
        stopwords = {
            "what", "when", "where", "how", "why", "does", "have", "with", "from",
            "tell", "show", "are", "the", "rules", "for", "and", "about", "can",
            "you", "please", "this", "that", "these", "those", "policy", "policies"
        }
        topic_terms = [w for w in words if w.lower() not in stopwords]

        if topic_terms:
            context_hint = " ".join(topic_terms[:3])
            # Construct reformulated query
            reformulated = f"{current_query} regarding {context_hint}"
            return reformulated

        return current_query
