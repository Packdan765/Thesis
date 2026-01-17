"""
Dialogue Planner for Museum Dialogue Agent

This module implements structured prompting for LLM-based dialogue generation.
It translates high-level options and subactions into structured prompts that
guide LLM generation while maintaining dialogue coherence.
"""

import re
from typing import List, Optional


def build_prompt(option: str, subaction: str, ex_id: Optional[str], 
                last_utt: str, facts_all: List[str], facts_used: List[str], 
                selected_fact: Optional[str], dialogue_history: List[str] = None,
                exhibit_names: List[str] = None, knowledge_graph=None,
                target_exhibit: str = None, coverage_dict: dict = None) -> str:
    """Build structured prompt for LLM dialogue generation"""
    
    # Build context
    show_facts = (option == "Explain") or (option == "Conclude" and subaction == "SummarizeKeyPoints")
    context_section = _build_context_section(ex_id, last_utt, facts_all, facts_used, dialogue_history, show_facts=show_facts)

    # Calculate current exhibit completion
    current_completion = 0.0
    if coverage_dict and ex_id:
        current_completion = coverage_dict.get(ex_id, {"coverage": 0.0})["coverage"]
    
    # Route to specific subaction function
    if option == "Explain":
        if subaction == "ExplainNewFact":
            return build_explain_new_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)
        elif subaction == "RepeatFact":
            return build_repeat_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)
        elif subaction == "ClarifyFact":
            return build_clarify_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)

    elif option == "AskQuestion":
        if subaction == "AskOpinion":
            return build_ask_opinion_prompt(ex_id, context_section, current_completion)
        elif subaction == "AskMemory":
            return build_ask_memory_prompt(ex_id, context_section, current_completion)
        elif subaction == "AskClarification":
            return build_ask_clarification_prompt(ex_id, context_section, current_completion)

    elif option == "OfferTransition":
        if subaction in ("SuggestMove", "SummarizeAndSuggest"):
            return build_offer_transition_prompt(ex_id, context_section, target_exhibit, coverage_dict)

    elif option == "Conclude":
        if subaction == "WrapUp":
            return build_wrap_up_prompt(ex_id, context_section, current_completion)
    
    raise ValueError(f"Unknown option '{option}' or subaction '{subaction}'")


def _build_context_section(ex_id: Optional[str], last_utt: str, facts_all: List[str], 
                          facts_used: List[str], dialogue_history: List = None, show_facts: bool = True) -> str:
    """Build context section for prompts with natural conversation flow"""
    context_parts = []
    
    if ex_id:
        context_parts.append(f"CURRENT EXHIBIT: {ex_id.replace('_', ' ')}")
    
    # Visitor's current message - present naturally
    if last_utt.strip():
        context_parts.append("")
        context_parts.append("VISITOR'S MESSAGE:")
        context_parts.append(f'"{last_utt}"')
        context_parts.append("")
    
    # Dialogue history - show conversation flow for natural continuity
    fact_ids_in_context = set()
    if dialogue_history and len(dialogue_history) > 0:
        recent_context = dialogue_history[-4:] if len(dialogue_history) > 4 else dialogue_history
        context_parts.append("CONVERSATION CONTEXT (for natural flow):")
        for i, utterance_tuple in enumerate(recent_context, 1):
            if len(utterance_tuple) >= 2:
                role, utterance = utterance_tuple[0], utterance_tuple[1]
            else:
                continue
            
            role_label = "AGENT" if role == "agent" else "VISITOR"
            context_parts.append(f'  {role_label}: "{utterance}"')
            
            fact_ids = re.findall(r'\[([A-Z]{2}_\d{3})\]', utterance)
            fact_ids_in_context.update(fact_ids)
        
        if fact_ids_in_context:
            context_parts.append("")
            context_parts.append(f"FACTS ALREADY SHARED: {sorted(fact_ids_in_context)}")
            context_parts.append("(Use conversation context to build naturally on what was discussed)")
        context_parts.append("")

    if not show_facts:
        context_parts.append("NOTE: This response type does not include new facts.")
        context_parts.append("")
    
    return "\n".join(context_parts)


# ===== EXPLAIN OPTION FUNCTIONS =====

def build_explain_new_fact_prompt(ex_id: Optional[str], context_section: str,
                                facts_all: List[str], facts_used: List[str],
                                selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for explaining a new fact"""
    
    # Filter to only NEW/unused facts
    used_ids = set()
    for fact in facts_used:
        match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact)
        if match:
            used_ids.add(match.group(1))
    
    new_facts = []
    for fact in facts_all:
        match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact)
        if match and match.group(1) not in used_ids:
            new_facts.append(fact)

    if not new_facts:
        return f"""Museum guide at: {ex_id}
{context_section}
No new facts available. Ask if they'd like to explore another exhibit.
Response (1-2 sentences):"""

    new_facts_list = "\n".join([f"  - {fact}" for fact in new_facts])
    
    return f"""Museum guide at: {ex_id}
{context_section}
NEW FACTS AVAILABLE (pick 1-2):
{new_facts_list}

INSTRUCTIONS:
1. Respond naturally to the visitor's message - use conversation context to maintain flow
2. Share 1-2 NEW facts with exact IDs in brackets [ID]
3. Keep it conversational (2-3 sentences)
4. DO NOT quote or repeat what the visitor said verbatim (e.g., avoid "You said...")
5. Use conversation history to build naturally on what was discussed - reference past facts/exhibits when relevant, but don't quote verbatim

STRATEGY: Explain/ExplainNewFact - Reference what was already shared to build on it naturally. Use history to know what facts were mentioned and continue the educational flow.

Response:"""


def build_repeat_fact_prompt(ex_id: Optional[str], context_section: str,
                           facts_all: List[str], facts_used: List[str],
                           selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for repeating a previously shared fact"""
    if facts_used:
        fact_to_repeat = selected_fact if selected_fact else facts_used[-1]
        fact_id_match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact_to_repeat)
        fact_id = fact_id_match.group(1) if fact_id_match else ""
        fact_content = re.sub(r'\[([A-Z]{2}_\d{3})\]\s*', '', fact_to_repeat).strip()
        
        return f"""Museum guide at: {ex_id}
{context_section}
REPHRASE this fact naturally: "{fact_content}"
Include the fact ID: [{fact_id}]

INSTRUCTIONS:
- Use conversation context to know what to repeat
- Respond naturally to the visitor's message
- DO NOT quote what the visitor said verbatim
- Rephrase the fact in fresh, clearer words

STRATEGY: Explain/RepeatFact - Use history to identify which fact needs repeating based on the conversation flow.

Response (2-3 sentences):"""
    else:
        return f"""Museum guide at: {ex_id}
{context_section}
No facts shared yet. Share an interesting fact about this exhibit.
Response (2-3 sentences):"""


def build_clarify_fact_prompt(ex_id: Optional[str], context_section: str,
                            facts_all: List[str], facts_used: List[str],
                            selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for clarifying a fact"""
    if facts_used:
        fact_to_clarify = selected_fact if selected_fact else facts_used[-1]
        return f"""Museum guide at: {ex_id}
{context_section}
CLARIFY this fact using a simple analogy: "{fact_to_clarify}"
NO new facts or [IDs] - just clarify.

INSTRUCTIONS:
- Use conversation context to understand what needs clarification
- Respond naturally to the visitor's message
- DO NOT quote what the visitor said verbatim
- Use a simple analogy or everyday example

STRATEGY: Explain/ClarifyFact - Reference what was discussed from history to understand what needs clarification.

Response (2-3 sentences):"""
    else:
        return f"""Museum guide at: {ex_id}
{context_section}
Clarify an interesting fact about this exhibit.
Response (2-3 sentences):"""


# ===== ASK QUESTION OPTION FUNCTIONS =====

def build_ask_opinion_prompt(ex_id: Optional[str], context_section: str, current_completion: float = 0.0) -> str:
    """Build prompt for asking visitor's opinion"""
    return f"""Museum guide at: {ex_id}
{context_section}
Ask a genuine question about their opinion or feeling.

INSTRUCTIONS:
- Use conversation history to ask a relevant follow-up question
- Respond naturally to what they said
- DO NOT quote what the visitor said verbatim
- NO facts or [IDs]

STRATEGY: AskQuestion/AskOpinion - Use history to ask relevant follow-ups based on what was discussed.

Response (1-2 sentences):"""


def build_ask_memory_prompt(ex_id: Optional[str], context_section: str, current_completion: float = 0.0) -> str:
    """Build prompt for checking visitor's memory"""
    return f"""Museum guide at: {ex_id}
{context_section}
Playfully check if they remember something from earlier.

INSTRUCTIONS:
- Use conversation history to reference what was discussed earlier
- Reference past topics naturally, don't quote verbatim
- DO NOT quote what the visitor said verbatim
- NO [FACT_IDs]

STRATEGY: AskQuestion/AskMemory - Naturally reference past conversation to check memory, don't quote verbatim.

Response (1-2 sentences):"""


def build_ask_clarification_prompt(ex_id: Optional[str], context_section: str, current_completion: float = 0.0) -> str:
    """Build prompt for asking for clarification"""
    return f"""Museum guide at: {ex_id}
{context_section}
Ask a clarifying question about what interests them.

INSTRUCTIONS:
- Use conversation history to understand what needs clarification
- Respond naturally to what they said
- DO NOT quote what the visitor said verbatim
- NO [FACT_IDs]

STRATEGY: AskQuestion/AskClarification - Use history to understand what needs clarification and ask relevant questions.

Response (1-2 sentences):"""


# ===== OFFER TRANSITION OPTION FUNCTIONS =====

def build_offer_transition_prompt(ex_id: Optional[str], context_section: str,
                                target_exhibit: str = None, coverage_dict: dict = None) -> str:
    """Build prompt for transitioning to another exhibit"""
    if not target_exhibit:
        return f"""Museum guide at: {ex_id}
{context_section}
Suggest moving to a different exhibit.
Response (2 sentences):"""

    target_name = target_exhibit.replace('_', ' ')
    current_name = ex_id.replace('_', ' ') if ex_id else 'current exhibit'

    return f"""Museum guide SUGGESTING move from "{current_name}" to "{target_name}"
{context_section}
1. Briefly wrap up current exhibit (1 sentence) - use conversation history to reference what was discussed
2. Introduce the new exhibit by name

INSTRUCTIONS:
- Use conversation history to reference what was discussed naturally
- DO NOT quote what the visitor said verbatim
- Transition smoothly based on the conversation flow

STRATEGY: OfferTransition/SummarizeAndSuggest - Reference what was discussed to transition smoothly. Use history to wrap up naturally.

Response (2-3 sentences):"""


# ===== CONCLUDE OPTION FUNCTIONS =====

def build_wrap_up_prompt(ex_id: Optional[str], context_section: str, current_completion: float = 0.0) -> str:
    """Build prompt for wrapping up the visit"""
    return f"""Museum guide at: {ex_id}
{context_section}
Thank them warmly for visiting.

INSTRUCTIONS:
- Use conversation history to summarize naturally what was discussed
- DO NOT quote what the visitor said verbatim
- NO [FACT_ID] tags

STRATEGY: Conclude/WrapUp - Summarize naturally using history context. Reference the overall experience without quoting verbatim.

Response (2 sentences):"""

