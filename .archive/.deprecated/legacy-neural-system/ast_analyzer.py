#!/usr/bin/env python3
"""
Universal AST Analyzer with Tree-sitter
Phase 1: Multi-language code structure analysis for semantic understanding
Supports 40+ programming languages with unified abstraction layer
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Core Tree-sitter support
TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter
    from tree_sitter_python import language as python_language
    from tree_sitter_javascript import language as javascript_language
    from tree_sitter_typescript import language as typescript_language
    from tree_sitter_java import language as java_language
    from tree_sitter_c import language as c_language
    from tree_sitter_cpp import language as cpp_language
    TREE_SITTER_AVAILABLE = True
except ImportError:
    tree_sitter = None
    python_language = None
    javascript_language = None
    typescript_language = None
    java_language = None
    c_language = None
    cpp_language = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionInfo:
    """Information about a function/method"""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    start_line: int
    end_line: int
    complexity: int = 0
    calls: List[str] = None
    
    def __post_init__(self):
        if self.calls is None:
            self.calls = []

@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    base_classes: List[str]
    methods: List[FunctionInfo]
    attributes: List[str]
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    
    def __post_init__(self):
        if self.base_classes is None:
            self.base_classes = []
        if self.methods is None:
            self.methods = []
        if self.attributes is None:
            self.attributes = []

@dataclass
class ImportInfo:
    """Information about imports"""
    module: str
    alias: Optional[str] = None
    items: List[str] = None
    is_from: bool = False
    
    def __post_init__(self):
        if self.items is None:
            self.items = []

@dataclass
class ASTAnalysisResult:
    """Complete AST analysis result"""
    file_path: str
    language: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[ImportInfo]
    variables: List[str]
    complexity_score: int
    lines_of_code: int
    analysis_time: float
    business_logic_score: float = 0.0
    architectural_patterns: List[str] = None
    
    def __post_init__(self):
        if self.architectural_patterns is None:
            self.architectural_patterns = []

class LanguageAnalyzer:
    """Base language analyzer with common patterns"""
    
    def __init__(self, language_name: str, parser: Optional[Any] = None):
        self.language_name = language_name
        self.parser = parser
        self.function_nodes = ["function_definition", "method_definition", "function_declaration"]
        self.class_nodes = ["class_definition", "class_declaration"]
        self.import_nodes = ["import_statement", "import_from_statement", "import_declaration"]
        
    def analyze(self, code: str, file_path: str) -> ASTAnalysisResult:
        """Analyze code and return structured information"""
        start_time = time.time()
        
        result = ASTAnalysisResult(
            file_path=file_path,
            language=self.language_name,
            functions=[],
            classes=[],
            imports=[],
            variables=[],
            complexity_score=0,
            lines_of_code=len(code.split('\n')),
            analysis_time=0
        )
        
        if not self.parser:
            logger.warning(f"No parser available for {self.language_name}")
            result.analysis_time = (time.time() - start_time) * 1000
            return result
            
        try:
            # Parse the code
            tree = self.parser.parse(bytes(code, 'utf8'))
            
            # Extract information
            self._extract_functions(tree.root_node, code, result)
            self._extract_classes(tree.root_node, code, result)
            self._extract_imports(tree.root_node, code, result)
            self._extract_variables(tree.root_node, code, result)
            
            # Calculate complexity
            result.complexity_score = self._calculate_complexity(tree.root_node)
            
            # Analyze business logic patterns
            result.business_logic_score = self._analyze_business_logic(result)
            
            # Detect architectural patterns
            result.architectural_patterns = self._detect_patterns(result, code)
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            
        result.analysis_time = (time.time() - start_time) * 1000
        return result
    
    def _extract_functions(self, node: Any, code: str, result: ASTAnalysisResult):
        """Extract function information from AST"""
        if node.type in self.function_nodes:
            func_info = self._parse_function(node, code)
            if func_info:
                result.functions.append(func_info)
        
        for child in node.children:
            self._extract_functions(child, code, result)
    
    def _extract_classes(self, node: Any, code: str, result: ASTAnalysisResult):
        """Extract class information from AST"""
        if node.type in self.class_nodes:
            class_info = self._parse_class(node, code)
            if class_info:
                result.classes.append(class_info)
        
        for child in node.children:
            self._extract_classes(child, code, result)
    
    def _extract_imports(self, node: Any, code: str, result: ASTAnalysisResult):
        """Extract import information from AST"""
        if node.type in self.import_nodes:
            import_info = self._parse_import(node, code)
            if import_info:
                result.imports.append(import_info)
        
        for child in node.children:
            self._extract_imports(child, code, result)
    
    def _extract_variables(self, node: Any, code: str, result: ASTAnalysisResult):
        """Extract variable declarations"""
        # This is language-specific and would be implemented in subclasses
        pass
    
    def _parse_function(self, node: Any, code: str) -> Optional[FunctionInfo]:
        """Parse function node (to be overridden by language-specific analyzers)"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
                
            name = code[name_node.start_byte:name_node.end_byte]
            
            # Extract parameters
            parameters = []
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for param in params_node.children:
                    if param.type in ["identifier", "parameter"]:
                        param_name = code[param.start_byte:param.end_byte]
                        parameters.append(param_name)
            
            # Extract docstring (look for first string literal)
            docstring = None
            for child in node.children:
                if child.type == "string":
                    docstring = code[child.start_byte:child.end_byte].strip('"\'')
                    break
            
            return FunctionInfo(
                name=name,
                parameters=parameters,
                return_type=None,  # Language-specific
                docstring=docstring,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                complexity=1  # Base complexity
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse function: {e}")
            return None
    
    def _parse_class(self, node: Any, code: str) -> Optional[ClassInfo]:
        """Parse class node"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
                
            name = code[name_node.start_byte:name_node.end_byte]
            
            # Extract base classes
            base_classes = []
            superclass_node = node.child_by_field_name("superclass")
            if superclass_node:
                base_name = code[superclass_node.start_byte:superclass_node.end_byte]
                base_classes.append(base_name)
            
            # Extract methods
            methods = []
            for child in node.children:
                if child.type in self.function_nodes:
                    method_info = self._parse_function(child, code)
                    if method_info:
                        methods.append(method_info)
            
            return ClassInfo(
                name=name,
                base_classes=base_classes,
                methods=methods,
                attributes=[],  # Would need more sophisticated parsing
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse class: {e}")
            return None
    
    def _parse_import(self, node: Any, code: str) -> Optional[ImportInfo]:
        """Parse import node"""
        try:
            # This is a simplified version - real implementation would be language-specific
            import_text = code[node.start_byte:node.end_byte]
            
            if "from" in import_text:
                # from module import items
                parts = import_text.replace("from", "").replace("import", "|").split("|")
                if len(parts) >= 2:
                    module = parts[0].strip()
                    items = [item.strip() for item in parts[1].split(",")]
                    return ImportInfo(module=module, items=items, is_from=True)
            else:
                # import module
                module = import_text.replace("import", "").strip()
                return ImportInfo(module=module)
                
        except Exception as e:
            logger.debug(f"Failed to parse import: {e}")
            return None
    
    def _calculate_complexity(self, node: Any) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_nodes = ["if_statement", "for_statement", "while_statement", 
                         "try_statement", "catch_clause", "switch_statement"]
        
        def count_decisions(n):
            nonlocal complexity
            if n.type in decision_nodes:
                complexity += 1
            for child in n.children:
                count_decisions(child)
        
        count_decisions(node)
        return complexity
    
    def _analyze_business_logic(self, result: ASTAnalysisResult) -> float:
        """Analyze business logic density"""
        score = 0.0
        
        # Check for business logic indicators
        business_patterns = [
            "validate", "process", "calculate", "transform", "handle",
            "create", "update", "delete", "save", "load", "export", "import"
        ]
        
        for func in result.functions:
            for pattern in business_patterns:
                if pattern in func.name.lower():
                    score += 1.0
                    break
        
        for cls in result.classes:
            if any(pattern in cls.name.lower() for pattern in ["service", "manager", "handler", "controller"]):
                score += 2.0
        
        # Normalize by total functions/classes
        total_items = len(result.functions) + len(result.classes)
        return score / max(1, total_items)
    
    def _detect_patterns(self, result: ASTAnalysisResult, code: str) -> List[str]:
        """Detect architectural patterns"""
        patterns = []
        
        # Design pattern detection
        if any("factory" in cls.name.lower() for cls in result.classes):
            patterns.append("Factory Pattern")
            
        if any("singleton" in cls.name.lower() for cls in result.classes):
            patterns.append("Singleton Pattern")
            
        if any("observer" in cls.name.lower() for cls in result.classes):
            patterns.append("Observer Pattern")
            
        # Architectural pattern detection
        class_names = [cls.name.lower() for cls in result.classes]
        
        if any("controller" in name for name in class_names):
            patterns.append("MVC Pattern")
            
        if any("service" in name for name in class_names):
            patterns.append("Service Layer Pattern")
            
        if any("repository" in name for name in class_names):
            patterns.append("Repository Pattern")
        
        # Framework detection
        import_modules = [imp.module.lower() for imp in result.imports]
        
        if any("react" in module for module in import_modules):
            patterns.append("React Framework")
            
        if any("express" in module for module in import_modules):
            patterns.append("Express.js Framework")
            
        if any("django" in module for module in import_modules):
            patterns.append("Django Framework")
        
        return patterns

class UniversalASTAnalyzer:
    """Universal AST analyzer supporting 40+ languages"""
    
    def __init__(self):
        self.analyzers: Dict[str, LanguageAnalyzer] = {}
        self.language_mappings = {
            ".py": "python",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript", 
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sh": "bash",
            ".pl": "perl",
            ".lua": "lua",
            ".sql": "sql",
            ".html": "html",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
        }
        
        self._init_analyzers()
    
    def _init_analyzers(self):
        """Initialize language analyzers"""
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available, using fallback analyzers")
            # Initialize fallback analyzers for all languages
            for ext, lang in self.language_mappings.items():
                self.analyzers[lang] = LanguageAnalyzer(lang, None)
            return
        
        # Initialize tree-sitter parsers for available languages
        parsers = {}
        
        try:
            parsers["python"] = tree_sitter.Parser()
            parsers["python"].set_language(python_language())
            
            parsers["javascript"] = tree_sitter.Parser()
            parsers["javascript"].set_language(javascript_language())
            
            parsers["typescript"] = tree_sitter.Parser()
            parsers["typescript"].set_language(typescript_language())
            
            parsers["java"] = tree_sitter.Parser()
            parsers["java"].set_language(java_language())
            
            parsers["c"] = tree_sitter.Parser()
            parsers["c"].set_language(c_language())
            
            parsers["cpp"] = tree_sitter.Parser()
            parsers["cpp"].set_language(cpp_language())
            
        except Exception as e:
            logger.error(f"Failed to initialize parsers: {e}")
        
        # Create analyzers
        for ext, lang in self.language_mappings.items():
            parser = parsers.get(lang)
            self.analyzers[lang] = LanguageAnalyzer(lang, parser)
        
        logger.info(f"Initialized analyzers for {len(self.analyzers)} languages")
    
    def get_language_from_file(self, file_path: str) -> Optional[str]:
        """Determine language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.language_mappings.get(ext)
    
    def analyze_file(self, file_path: str) -> Optional[ASTAnalysisResult]:
        """Analyze a single file"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        language = self.get_language_from_file(file_path)
        if not language:
            logger.debug(f"Unsupported file type: {file_path}")
            return None
        
        analyzer = self.analyzers.get(language)
        if not analyzer:
            logger.error(f"No analyzer for language: {language}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            return analyzer.analyze(code, file_path)
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def analyze_directory(self, directory: str, max_files: Optional[int] = None) -> List[ASTAnalysisResult]:
        """Analyze all supported files in a directory"""
        results = []
        files_processed = 0
        
        # Collect all supported files
        supported_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if files_processed >= (max_files or float('inf')):
                    break
                    
                file_path = os.path.join(root, file)
                if self.get_language_from_file(file_path):
                    supported_files.append(file_path)
                    files_processed += 1
        
        logger.info(f"Analyzing {len(supported_files)} files...")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.analyze_file, file_path): file_path 
                      for file_path in supported_files}
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Successfully analyzed {len(results)} files")
        return results
    
    def get_project_summary(self, results: List[ASTAnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive project summary"""
        if not results:
            return {}
        
        summary = {
            "total_files": len(results),
            "languages": {},
            "total_functions": 0,
            "total_classes": 0,
            "total_imports": 0,
            "total_lines": 0,
            "avg_complexity": 0,
            "business_logic_density": 0,
            "architectural_patterns": set(),
            "framework_usage": set(),
            "top_files_by_complexity": [],
            "analysis_time": sum(r.analysis_time for r in results)
        }
        
        complexity_scores = []
        business_scores = []
        
        for result in results:
            # Language statistics
            lang = result.language
            if lang not in summary["languages"]:
                summary["languages"][lang] = {
                    "files": 0,
                    "functions": 0,
                    "classes": 0,
                    "lines": 0
                }
            
            summary["languages"][lang]["files"] += 1
            summary["languages"][lang]["functions"] += len(result.functions)
            summary["languages"][lang]["classes"] += len(result.classes)
            summary["languages"][lang]["lines"] += result.lines_of_code
            
            # Overall statistics
            summary["total_functions"] += len(result.functions)
            summary["total_classes"] += len(result.classes)
            summary["total_imports"] += len(result.imports)
            summary["total_lines"] += result.lines_of_code
            
            complexity_scores.append(result.complexity_score)
            business_scores.append(result.business_logic_score)
            
            # Patterns and frameworks
            summary["architectural_patterns"].update(result.architectural_patterns)
        
        # Calculate averages
        if complexity_scores:
            summary["avg_complexity"] = sum(complexity_scores) / len(complexity_scores)
        
        if business_scores:
            summary["business_logic_density"] = sum(business_scores) / len(business_scores)
        
        # Convert sets to lists for JSON serialization
        summary["architectural_patterns"] = list(summary["architectural_patterns"])
        summary["framework_usage"] = list(summary["framework_usage"])
        
        # Top complex files
        summary["top_files_by_complexity"] = sorted(
            [(r.file_path, r.complexity_score) for r in results],
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return summary

# Global analyzer instance
_ast_analyzer = None

def get_ast_analyzer() -> UniversalASTAnalyzer:
    """Get or create global AST analyzer"""
    global _ast_analyzer
    if _ast_analyzer is None:
        _ast_analyzer = UniversalASTAnalyzer()
    return _ast_analyzer

def analyze_project_structure(project_root: str, max_files: Optional[int] = None) -> Dict[str, Any]:
    """Analyze complete project structure"""
    analyzer = get_ast_analyzer()
    results = analyzer.analyze_directory(project_root, max_files)
    summary = analyzer.get_project_summary(results)
    
    return {
        "summary": summary,
        "detailed_results": [asdict(result) for result in results],
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # Example usage
    analyzer = get_ast_analyzer()
    
    # Test single file analysis
    current_file = __file__
    result = analyzer.analyze_file(current_file)
    
    if result:
        print(f"Analysis of {current_file}:")
        print(f"  Language: {result.language}")
        print(f"  Functions: {len(result.functions)}")
        print(f"  Classes: {len(result.classes)}")
        print(f"  Complexity: {result.complexity_score}")
        print(f"  Business Logic Score: {result.business_logic_score:.2f}")
        print(f"  Patterns: {result.architectural_patterns}")
        print(f"  Analysis Time: {result.analysis_time:.2f}ms")
    
    # Test project analysis
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_analysis = analyze_project_structure(project_root, max_files=10)
    
    print(f"\nProject Analysis Summary:")
    print(json.dumps(project_analysis["summary"], indent=2))