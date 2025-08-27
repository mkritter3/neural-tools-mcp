#!/usr/bin/env python3
"""
L9 Safety Checker - Pre-tool execution safety validation
Runs before each tool use to prevent dangerous operations
"""

import sys
import os
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_tool_safety(tool_name: str, tool_args: dict) -> bool:
    """Check if tool usage is safe"""
    
    # Get tool arguments
    file_path = tool_args.get('file_path', '')
    command = tool_args.get('command', '')
    old_string = tool_args.get('old_string', '')
    new_string = tool_args.get('new_string', '')
    
    # Critical safety checks
    if tool_name in ['Read', 'Edit', 'Write', 'MultiEdit']:
        if file_path:
            # Check for sensitive file patterns
            sensitive_patterns = [
                '.env', 'secret', 'credential', 'password',
                '.ssh/', '.aws/', '.git/config', 'private'
            ]
            
            file_lower = file_path.lower()
            for pattern in sensitive_patterns:
                if pattern in file_lower:
                    logger.warning(f"🔒 SAFETY: Blocking access to sensitive file: {file_path}")
                    return False
                    
    elif tool_name == 'Bash':
        if command:
            # Check for dangerous command patterns
            dangerous_patterns = [
                'rm -rf', 'rm -r', 'sudo', 'curl', 'wget',
                'chmod 777', 'kill -9', 'format', 'mkfs'
            ]
            
            command_lower = command.lower()
            for pattern in dangerous_patterns:
                if pattern in command_lower:
                    logger.warning(f"🔒 SAFETY: Blocking dangerous command: {command}")
                    return False
                    
    # Check for suspicious content changes
    if tool_name in ['Edit', 'MultiEdit'] and new_string:
        suspicious_content = [
            'password', 'secret', 'api_key', 'private_key',
            'rm -rf', 'sudo', 'exec(', 'eval('
        ]
        
        new_string_lower = new_string.lower()
        for content in suspicious_content:
            if content in new_string_lower:
                logger.warning(f"🔒 SAFETY: Suspicious content change detected")
                return False
                
    logger.info(f"✅ SAFETY: Tool {tool_name} usage approved")
    return True

def main():
    """Main safety checker entry point"""
    
    # Check if tool usage data is provided via environment or args
    if len(sys.argv) > 1:
        # Tool data passed as argument
        tool_data = json.loads(sys.argv[1])
    else:
        # Check environment variables
        tool_name = os.getenv('CLAUDE_TOOL_NAME', '')
        tool_args_str = os.getenv('CLAUDE_TOOL_ARGS', '{}')
        
        if not tool_name:
            logger.info("✅ SAFETY: No tool execution detected")
            return 0
            
        try:
            tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError:
            tool_args = {}
            
        tool_data = {
            'tool_name': tool_name,
            'arguments': tool_args
        }
    
    # Perform safety check
    is_safe = check_tool_safety(
        tool_data.get('tool_name', ''),
        tool_data.get('arguments', {})
    )
    
    if not is_safe:
        logger.error("❌ SAFETY: Tool usage blocked for security reasons")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)