#!/usr/bin/env python3
"""
L9 Hook Formatters - Shared Formatting Functions
Centralizes all formatting logic used across hooks to eliminate code duplication
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .utilities import estimate_tokens


def format_session_metadata(session_data: Dict[str, Any], token_limit: int = 300) -> str:
    """Format session metadata consistently across hooks"""
    if not session_data:
        return "No session metadata available"
    
    timestamp = session_data.get('timestamp', datetime.now().isoformat())
    project_name = session_data.get('project_name', 'Unknown Project')
    session_type = session_data.get('session_type', 'development')
    
    formatted = f"""📅 Session: {timestamp[:19]}
📋 Project: {project_name}
🎯 Type: {session_type}"""
    
    # Truncate if over token limit
    if estimate_tokens(formatted) > token_limit:
        return f"📅 {timestamp[:16]} | 📋 {project_name} | 🎯 {session_type}"
    
    return formatted


def format_file_references(file_refs: List[str], max_files: int = 10, token_limit: int = 400) -> str:
    """Format file references with consistent truncation"""
    if not file_refs:
        return "No files referenced"
    
    # Limit number of files
    limited_files = file_refs[:max_files]
    
    formatted_lines = ["📁 FILES REFERENCED:"]
    for i, file_path in enumerate(limited_files, 1):
        # Use just filename for brevity
        filename = file_path.split('/')[-1] if '/' in file_path else file_path
        formatted_lines.append(f"  {i}. {filename}")
    
    if len(file_refs) > max_files:
        formatted_lines.append(f"  ... and {len(file_refs) - max_files} more files")
    
    formatted = "\n".join(formatted_lines)
    
    # Truncate if over token limit
    if estimate_tokens(formatted) > token_limit:
        # Show only top 3 files
        top_files = [f.split('/')[-1] for f in file_refs[:3]]
        return f"📁 FILES: {', '.join(top_files)}" + (f" (+{len(file_refs)-3})" if len(file_refs) > 3 else "")
    
    return formatted


def format_technical_context(tech_context: Dict[str, List[str]], token_limit: int = 500) -> str:
    """Format technical context (technologies, operations, outcomes)"""
    if not tech_context:
        return "No technical context available"
    
    sections = []
    
    # Technologies
    if tech_context.get('technologies'):
        tech_list = tech_context['technologies'][:8]  # Limit to 8 items
        sections.append(f"🔧 Tech: {', '.join(tech_list)}")
    
    # Operations/Tools used
    if tech_context.get('operations'):
        ops_list = tech_context['operations'][:6]  # Limit to 6 items
        sections.append(f"⚙️ Tools: {', '.join(ops_list)}")
    
    # Outcomes
    if tech_context.get('outcomes'):
        outcomes = tech_context['outcomes'][:3]  # Limit to 3 outcomes
        outcome_text = "; ".join(outcomes)
        if len(outcome_text) > 100:  # Truncate long outcomes
            outcome_text = outcome_text[:97] + "..."
        sections.append(f"✨ Results: {outcome_text}")
    
    formatted = "\n".join(sections)
    
    # Final truncation check
    if estimate_tokens(formatted) > token_limit:
        # Ultra-compact format
        tech = tech_context.get('technologies', [])[:3]
        ops = tech_context.get('operations', [])[:3]
        return f"🔧 {', '.join(tech)} | ⚙️ {', '.join(ops)}"
    
    return formatted


def format_key_decisions(decisions: List[str], max_decisions: int = 5, token_limit: int = 600) -> str:
    """Format key decisions with consistent structure"""
    if not decisions:
        return "No key decisions recorded"
    
    limited_decisions = decisions[:max_decisions]
    
    formatted_lines = ["🎯 KEY DECISIONS:"]
    for i, decision in enumerate(limited_decisions, 1):
        # Truncate individual decisions
        clean_decision = decision.strip()
        if len(clean_decision) > 120:
            clean_decision = clean_decision[:117] + "..."
        formatted_lines.append(f"  {i}. {clean_decision}")
    
    if len(decisions) > max_decisions:
        formatted_lines.append(f"  ... and {len(decisions) - max_decisions} more decisions")
    
    formatted = "\n".join(formatted_lines)
    
    # Token limit check
    if estimate_tokens(formatted) > token_limit:
        # Show only top 2 decisions
        top_2 = [d.strip()[:60] + "..." if len(d.strip()) > 60 else d.strip() for d in decisions[:2]]
        return "🎯 DECISIONS: " + "; ".join(top_2)
    
    return formatted


def format_mcp_suggestions(suggestions: List[Dict[str, Any]], max_suggestions: int = 5) -> str:
    """Format MCP enhancement suggestions"""
    if not suggestions:
        return "No MCP suggestions available"
    
    limited_suggestions = suggestions[:max_suggestions]
    
    formatted_lines = ["💡 MCP SUGGESTIONS:"]
    for i, suggestion in enumerate(limited_suggestions, 1):
        query = suggestion.get('query', 'No query')
        tool = suggestion.get('tool', 'unknown')
        priority = suggestion.get('priority', 'medium')
        
        # Truncate query for display
        display_query = query[:60] + "..." if len(query) > 60 else query
        priority_emoji = "🔥" if priority == "high" else "⚡" if priority == "medium" else "💡"
        
        formatted_lines.append(f"  {i}. {priority_emoji} {tool}: {display_query}")
    
    return "\n".join(formatted_lines)


def format_project_overview(project_data: Dict[str, Any], token_limit: int = 400) -> str:
    """Format project overview with consistent structure"""
    if not project_data:
        return "No project overview available"
    
    name = project_data.get('name', 'Unknown')
    total_files = project_data.get('total_files', 0)
    languages = project_data.get('main_languages', [])
    
    # Build feature list
    features = []
    if project_data.get('has_docker'):
        features.append("Docker")
    if project_data.get('has_package_json'):
        features.append("Node.js")
    if project_data.get('has_requirements'):
        features.append("Python")
    
    sections = [
        f"📂 Project: {name}",
        f"📊 Files: {total_files}",
        f"💻 Languages: {', '.join(languages[:3])}" if languages else "💻 Languages: Unknown",
    ]
    
    if features:
        sections.append(f"🛠️ Stack: {', '.join(features[:4])}")
    
    formatted = " | ".join(sections)
    
    # Token limit check
    if estimate_tokens(formatted) > token_limit:
        return f"📂 {name} | 📊 {total_files} files | 💻 {languages[0] if languages else 'Mixed'}"
    
    return formatted


def format_historical_context(history_data: Dict[str, Any], token_limit: int = 300) -> str:
    """Format historical context (recent files, changes, etc.)"""
    if not history_data:
        return "No historical context available"
    
    recent_files = history_data.get('recent_files', [])
    period = history_data.get('analysis_period', '7 days')
    
    if not recent_files:
        return f"📈 No recent changes ({period})"
    
    if isinstance(recent_files[0], str) and recent_files[0] == "Recent file analysis unavailable":
        return "📈 Recent file analysis unavailable"
    
    # Format recent files
    file_count = len(recent_files)
    if file_count <= 3:
        file_list = ", ".join(f.split('/')[-1] for f in recent_files)
        formatted = f"📈 Recent ({period}): {file_list}"
    else:
        top_files = [f.split('/')[-1] for f in recent_files[:3]]
        formatted = f"📈 Recent ({period}): {', '.join(top_files)} (+{file_count-3} more)"
    
    # Token limit check
    if estimate_tokens(formatted) > token_limit:
        return f"📈 {file_count} files modified ({period})"
    
    return formatted


def format_session_outcomes(outcomes: List[str], token_limit: int = 400) -> str:
    """Format session outcomes with consistent emoji and structure"""
    if not outcomes:
        return "No session outcomes recorded"
    
    formatted_lines = []
    for outcome in outcomes[:5]:  # Limit to 5 outcomes
        # Clean up outcome text
        clean_outcome = outcome.strip()
        if len(clean_outcome) > 80:
            clean_outcome = clean_outcome[:77] + "..."
        
        # Add appropriate emoji based on content
        if any(marker in clean_outcome for marker in ['✅', '✓', 'completed', 'successful', 'working', 'fixed']):
            emoji = "✅"
        elif any(marker in clean_outcome for marker in ['❌', '✗', 'failed', 'error', 'issue']):
            emoji = "❌"
        elif any(marker in clean_outcome for marker in ['⚠️', 'warning', 'caution', 'note']):
            emoji = "⚠️"
        else:
            emoji = "📋"
        
        formatted_lines.append(f"{emoji} {clean_outcome}")
    
    formatted = "\n".join(formatted_lines)
    
    # Token limit check
    if estimate_tokens(formatted) > token_limit:
        # Ultra-compact: just count outcomes by type
        success_count = sum(1 for o in outcomes if any(marker in o.lower() for marker in ['completed', 'successful', 'fixed']))
        error_count = sum(1 for o in outcomes if any(marker in o.lower() for marker in ['failed', 'error', 'issue']))
        return f"📊 Outcomes: {success_count} completed, {error_count} issues"
    
    return formatted


def format_context_summary(context: Dict[str, Any], max_tokens: int = 2000) -> str:
    """
    Master formatter that creates a complete context summary
    Automatically balances different sections within token budget
    """
    sections = []
    remaining_tokens = max_tokens
    
    # Reserve tokens for structure
    structure_tokens = 100
    remaining_tokens -= structure_tokens
    
    # Calculate section budgets (percentages of remaining tokens)
    budgets = {
        'metadata': int(remaining_tokens * 0.15),    # 15% for session metadata
        'project': int(remaining_tokens * 0.20),     # 20% for project overview
        'technical': int(remaining_tokens * 0.25),   # 25% for technical context
        'decisions': int(remaining_tokens * 0.20),   # 20% for key decisions
        'files': int(remaining_tokens * 0.10),       # 10% for file references
        'history': int(remaining_tokens * 0.10),     # 10% for historical context
    }
    
    # Format each section with its budget
    if context.get('metadata'):
        sections.append(format_session_metadata(context['metadata'], budgets['metadata']))
    
    if context.get('project_data'):
        sections.append(format_project_overview(context['project_data'], budgets['project']))
    
    if context.get('technical_context'):
        sections.append(format_technical_context(context['technical_context'], budgets['technical']))
    
    if context.get('key_decisions'):
        sections.append(format_key_decisions(context['key_decisions'], token_limit=budgets['decisions']))
    
    if context.get('file_references'):
        sections.append(format_file_references(context['file_references'], token_limit=budgets['files']))
    
    if context.get('historical'):
        sections.append(format_historical_context(context['historical'], budgets['history']))
    
    # Join with consistent separator
    full_context = "\n\n".join(sections)
    
    # Final safety check
    actual_tokens = estimate_tokens(full_context)
    if actual_tokens > max_tokens:
        # Emergency truncation
        truncation_target = max_tokens * 4  # Rough character count
        truncated = full_context[:truncation_target] + "\n\n...[Context truncated to fit token budget]"
        return truncated
    
    return full_context