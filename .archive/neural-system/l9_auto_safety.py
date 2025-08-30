#!/usr/bin/env python3
"""
L9 Auto-Safety System - Zero-Config Protection for Vibe Coders
Automatically detects and protects sensitive files and dangerous operations
"""

import os
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SafetyRule:
    """Individual safety rule"""
    pattern: str
    rule_type: str  # "deny", "ask", "allow"
    description: str
    severity: str   # "critical", "high", "medium", "low"
    auto_detected: bool = True

@dataclass
class ProjectSafetyProfile:
    """Complete safety profile for a project"""
    project_path: str
    sensitive_files: List[str]
    dangerous_commands: List[str]
    critical_configs: List[str]
    safety_rules: List[SafetyRule]
    protection_level: str
    last_updated: str

class L9AutoSafetySystem:
    """
    Zero-configuration safety system for vibe coders
    Automatically protects projects without requiring setup
    """
    
    # Universal sensitive file patterns
    UNIVERSAL_SENSITIVE_PATTERNS = [
        # Environment and secrets
        ".env*", "*.env", ".environment*", "env.*",
        "secrets/**", "secret.*", "*.secret*", "*.secrets",
        "credentials.*", "*.credentials", "creds.*", "*.creds",
        
        # Authentication and keys
        "*.pem", "*.key", "*.p12", "*.pfx", "*.crt", "*.cer",
        ".ssh/**", "ssh_*", "*_rsa", "*_dsa", "*_ecdsa",
        "*.ppk", "known_hosts", "authorized_keys",
        
        # Cloud and service configs
        ".aws/**", ".gcp/**", ".azure/**", 
        "gcloud/**", "kubectl/**", "terraform.tfstate*",
        "docker-compose.prod.*", "docker-compose.production.*",
        
        # Database and production configs
        "config/database.*", "config/production.*", "config/secrets.*",
        "database.yml", "production.yml", "secrets.yml",
        
        # Version control sensitive
        ".git/config", ".gitignore_global", ".git-credentials",
        
        # IDE and editor sensitive
        ".vscode/settings.json", ".idea/**", "*.sublime-workspace"
    ]
    
    # Universal dangerous commands
    UNIVERSAL_DANGEROUS_COMMANDS = [
        # File system destruction
        "rm -rf", "rm -r", "rmdir /s", "del /f /s /q",
        "rm --recursive --force", "find . -delete",
        
        # Network operations
        "curl", "wget", "fetch", "http", "https",
        
        # System operations
        "sudo", "su -", "chmod 777", "chmod -R 777",
        "chown -R", "chgrp -R",
        
        # Process and service control
        "kill -9", "killall", "pkill",
        "systemctl", "service", "launchctl",
        
        # Package and system modification
        "npm publish", "pip install --global", "gem install",
        "apt install", "yum install", "brew install --cask",
        
        # Git dangerous operations
        "git push --force", "git push -f", "git reset --hard HEAD~",
        "git clean -fd", "git checkout -- .",
        
        # Docker dangerous operations
        "docker system prune", "docker volume prune", 
        "docker network prune", "docker container prune"
    ]
    
    # Ask-before patterns (risky but sometimes necessary)
    ASK_BEFORE_PATTERNS = [
        # Publishing and deployment
        "git push", "npm publish", "pip upload", "docker build",
        "docker push", "terraform apply", "kubectl apply",
        
        # Configuration changes
        "package.json", "requirements.txt", "Cargo.toml",
        "tsconfig.json", "webpack.config.*", "babel.config.*",
        "Dockerfile", "docker-compose.*", "Makefile",
        
        # Database operations
        "migrate", "seed", "schema", "database"
    ]
    
    def __init__(self):
        self.project_profiles = {}
        logger.info("🔒 Initializing L9 Auto-Safety System...")
        
    def scan_project_for_risks(self, project_path: str) -> ProjectSafetyProfile:
        """Comprehensively scan project for security risks"""
        project_path = Path(project_path).resolve()
        
        logger.info(f"🔍 Scanning project for risks: {project_path}")
        
        # Scan for sensitive files
        sensitive_files = self._detect_sensitive_files(project_path)
        
        # Analyze project structure for risky patterns
        dangerous_commands = self._identify_dangerous_commands(project_path)
        
        # Identify critical configuration files
        critical_configs = self._identify_critical_configs(project_path)
        
        # Generate safety rules
        safety_rules = self._generate_safety_rules(
            sensitive_files, dangerous_commands, critical_configs
        )
        
        # Determine protection level
        protection_level = self._assess_protection_level(safety_rules)
        
        profile = ProjectSafetyProfile(
            project_path=str(project_path),
            sensitive_files=sensitive_files,
            dangerous_commands=dangerous_commands,
            critical_configs=critical_configs,
            safety_rules=safety_rules,
            protection_level=protection_level,
            last_updated=self._get_timestamp()
        )
        
        self.project_profiles[str(project_path)] = profile
        
        logger.info(f"✅ Risk scan complete: {len(sensitive_files)} sensitive files, "
                   f"{len(dangerous_commands)} dangerous commands, "
                   f"protection level: {protection_level}")
        
        return profile
        
    def _detect_sensitive_files(self, project_path: Path) -> List[str]:
        """Detect sensitive files using pattern matching"""
        sensitive_files = set()
        
        for pattern in self.UNIVERSAL_SENSITIVE_PATTERNS:
            try:
                # Use glob to find matching files
                matches = glob.glob(
                    str(project_path / pattern), 
                    recursive=True
                )
                
                # Convert to relative paths
                for match in matches:
                    rel_path = os.path.relpath(match, project_path)
                    if self._is_actual_file(match):
                        sensitive_files.add(rel_path)
                        
            except Exception as e:
                logger.debug(f"Pattern {pattern} scan error: {e}")
                
        # Additional heuristic detection
        sensitive_files.update(self._heuristic_sensitive_detection(project_path))
        
        return sorted(list(sensitive_files))
        
    def _heuristic_sensitive_detection(self, project_path: Path) -> Set[str]:
        """Use heuristics to detect sensitive files"""
        sensitive = set()
        
        # Scan common files for sensitive content
        common_files = [
            "config.py", "settings.py", "config.js", "config.json",
            "package.json", "requirements.txt", "Dockerfile"
        ]
        
        for file_name in common_files:
            file_path = project_path / file_name
            if file_path.exists() and file_path.is_file():
                if self._contains_sensitive_content(file_path):
                    sensitive.add(str(file_path.relative_to(project_path)))
                    
        return sensitive
        
    def _contains_sensitive_content(self, file_path: Path) -> bool:
        """Check if file contains sensitive content patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                
            sensitive_keywords = [
                "password", "secret", "api_key", "private_key",
                "token", "credential", "auth_token", "access_key",
                "database_url", "connection_string", "jwt_secret"
            ]
            
            return any(keyword in content for keyword in sensitive_keywords)
            
        except Exception:
            return False
            
    def _identify_dangerous_commands(self, project_path: Path) -> List[str]:
        """Identify dangerous commands specific to this project"""
        dangerous = set(self.UNIVERSAL_DANGEROUS_COMMANDS)
        
        # Project-specific dangerous commands
        if (project_path / "package.json").exists():
            dangerous.update([
                "npm install --global", "npm link", "npm run build:prod"
            ])
            
        if (project_path / "requirements.txt").exists():
            dangerous.update([
                "pip install -e .", "python setup.py install"
            ])
            
        if (project_path / "Dockerfile").exists():
            dangerous.update([
                "docker build --no-cache", "docker run --privileged"
            ])
            
        return sorted(list(dangerous))
        
    def _identify_critical_configs(self, project_path: Path) -> List[str]:
        """Identify critical configuration files"""
        critical_configs = []
        
        config_patterns = [
            "package.json", "requirements.txt", "Cargo.toml", "go.mod",
            "tsconfig.json", "webpack.config.*", "babel.config.*",
            "Dockerfile", "docker-compose.*", "Makefile", "CMakeLists.txt",
            "pyproject.toml", "poetry.lock", "Pipfile", "setup.py"
        ]
        
        for pattern in config_patterns:
            matches = glob.glob(str(project_path / pattern))
            for match in matches:
                rel_path = os.path.relpath(match, project_path)
                if self._is_actual_file(match):
                    critical_configs.append(rel_path)
                    
        return sorted(critical_configs)
        
    def _generate_safety_rules(self, 
                             sensitive_files: List[str],
                             dangerous_commands: List[str],
                             critical_configs: List[str]) -> List[SafetyRule]:
        """Generate comprehensive safety rules"""
        rules = []
        
        # Rules for sensitive files (DENY)
        for file_path in sensitive_files:
            rules.append(SafetyRule(
                pattern=f"Read({file_path})",
                rule_type="deny",
                description=f"Protect sensitive file: {file_path}",
                severity="critical",
                auto_detected=True
            ))
            
        # Rules for dangerous commands (DENY)
        for command in dangerous_commands:
            rules.append(SafetyRule(
                pattern=f"Bash({command}:*)",
                rule_type="deny", 
                description=f"Block dangerous command: {command}",
                severity="high",
                auto_detected=True
            ))
            
        # Rules for critical configs (ASK)
        for config_file in critical_configs:
            rules.append(SafetyRule(
                pattern=f"Edit({config_file})",
                rule_type="ask",
                description=f"Confirm changes to critical config: {config_file}",
                severity="medium",
                auto_detected=True
            ))
            
        # Ask-before patterns
        for pattern in self.ASK_BEFORE_PATTERNS:
            if "." in pattern and not pattern.startswith("git "):
                # File pattern
                rules.append(SafetyRule(
                    pattern=f"Edit({pattern})",
                    rule_type="ask",
                    description=f"Confirm changes to: {pattern}",
                    severity="medium",
                    auto_detected=True
                ))
            else:
                # Command pattern
                rules.append(SafetyRule(
                    pattern=f"Bash({pattern}:*)",
                    rule_type="ask",
                    description=f"Confirm execution: {pattern}",
                    severity="medium", 
                    auto_detected=True
                ))
                
        return rules
        
    def _assess_protection_level(self, safety_rules: List[SafetyRule]) -> str:
        """Assess overall protection level"""
        critical_count = sum(1 for rule in safety_rules if rule.severity == "critical")
        high_count = sum(1 for rule in safety_rules if rule.severity == "high")
        total_rules = len(safety_rules)
        
        if critical_count >= 3 or high_count >= 5:
            return "maximum"
        elif critical_count >= 1 or high_count >= 3:
            return "high"
        elif total_rules >= 10:
            return "medium"
        else:
            return "basic"
            
    def generate_claude_settings(self, profile: ProjectSafetyProfile) -> Dict[str, Any]:
        """Generate Claude Code settings.json for auto-protection"""
        
        deny_patterns = []
        ask_patterns = []
        
        for rule in profile.safety_rules:
            if rule.rule_type == "deny":
                deny_patterns.append(rule.pattern)
            elif rule.rule_type == "ask":
                ask_patterns.append(rule.pattern)
                
        settings = {
            "enableAllProjectMcpServers": True,
            "permissions": {
                "allow": ["*"],  # Allow everything by default
                "deny": deny_patterns,
                "ask": ask_patterns
            },
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Edit|MultiEdit|Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python .claude/neural-system/safety_checker.py"
                            }
                        ]
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "Edit|MultiEdit|Write",
                        "hooks": [
                            {
                                "type": "command", 
                                "command": "python .claude/neural-system/style_preserver.py"
                            }
                        ]
                    }
                ]
            },
            "env": {
                "NEURAL_L9_MODE": "1",
                "USE_SINGLE_QODO_MODEL": "1",
                "ENABLE_AUTO_SAFETY": "1",
                "L9_PROTECTION_LEVEL": profile.protection_level
            },
            "_metadata": {
                "generated_by": "L9 Auto-Safety System",
                "generated_at": self._get_timestamp(),
                "protection_level": profile.protection_level,
                "total_rules": len(profile.safety_rules),
                "sensitive_files": len(profile.sensitive_files)
            }
        }
        
        return settings
        
    def auto_setup_project_safety(self, project_path: str) -> Dict[str, Any]:
        """Automatically set up safety for a project - ZERO CONFIG"""
        project_path = Path(project_path).resolve()
        
        logger.info(f"🔒 Auto-setting up safety for: {project_path}")
        
        # Scan project for risks
        profile = self.scan_project_for_risks(project_path)
        
        # Generate Claude settings
        settings = self.generate_claude_settings(profile)
        
        # Ensure .claude directory exists
        claude_dir = project_path / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        
        # Write settings.json
        settings_path = claude_dir / "settings.json"
        
        try:
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
                
            logger.info(f"✅ Auto-safety configured: {settings_path}")
            
            # Write safety profile for reference
            profile_path = claude_dir / "safety_profile.json"
            with open(profile_path, 'w') as f:
                json.dump(asdict(profile), f, indent=2)
                
            return {
                "status": "success",
                "message": "✅ L9 Auto-Safety configured successfully",
                "settings_path": str(settings_path),
                "protection_level": profile.protection_level,
                "sensitive_files_protected": len(profile.sensitive_files),
                "dangerous_commands_blocked": len(profile.dangerous_commands),
                "total_safety_rules": len(profile.safety_rules),
                "zero_config": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to write safety configuration: {e}")
            return {
                "status": "error",
                "message": f"Failed to configure auto-safety: {e}",
                "zero_config": False
            }
            
    def _is_actual_file(self, path: str) -> bool:
        """Check if path is an actual file (not directory)"""
        return os.path.isfile(path)
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Global instance for easy access
_l9_safety_system = None

def get_l9_safety_system() -> L9AutoSafetySystem:
    """Get or create L9 safety system instance"""
    global _l9_safety_system
    if _l9_safety_system is None:
        _l9_safety_system = L9AutoSafetySystem()
    return _l9_safety_system

def auto_setup_current_project() -> Dict[str, Any]:
    """Auto-setup safety for current working directory"""
    current_dir = os.getcwd()
    safety_system = get_l9_safety_system()
    return safety_system.auto_setup_project_safety(current_dir)

async def main():
    """Test L9 auto-safety system"""
    safety_system = L9AutoSafetySystem()
    
    # Test on current project
    project_path = "/Users/mkr/local-coding/claude-l9-template"
    
    print(f"🔍 Testing L9 Auto-Safety on: {project_path}")
    
    result = safety_system.auto_setup_project_safety(project_path)
    
    print(f"\n📊 Setup Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
        
    # Show the generated profile
    if project_path in safety_system.project_profiles:
        profile = safety_system.project_profiles[project_path]
        print(f"\n🔒 Safety Profile Summary:")
        print(f"  Protection Level: {profile.protection_level}")
        print(f"  Sensitive Files: {len(profile.sensitive_files)}")
        print(f"  Dangerous Commands: {len(profile.dangerous_commands)}")
        print(f"  Safety Rules: {len(profile.safety_rules)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())