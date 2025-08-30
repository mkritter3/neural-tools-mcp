#!/usr/bin/env python3
"""
L9 Hook Validation Script
Systematically validates all hooks for L9 compliance
"""

from pathlib import Path
import sys
import json

# Add hook_utils to path
sys.path.insert(0, str(Path(__file__).parent / 'hook_utils'))

from hook_utils.validators import validate_all_hooks


def main():
    """Validate all hooks and generate compliance report"""
    
    hooks_dir = Path(__file__).parent / 'hooks'
    
    if not hooks_dir.exists():
        print("❌ Hooks directory not found", file=sys.stderr)
        return 1
    
    print("🔍 L9 Hook Compliance Validation")
    print("=" * 50)
    
    # Run validation
    results = validate_all_hooks(hooks_dir)
    
    # Display results
    print(f"\n📊 SUMMARY:")
    print(f"Total hooks: {results['summary']['total_hooks']}")
    print(f"Compliant hooks: {results['summary']['compliant_hooks']}")
    print(f"Compliance rate: {results['summary']['compliance_rate']:.1%}")
    print(f"Average score: {results['summary']['average_score']:.2f}")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    
    for hook_name, result in results['results'].items():
        status = "✅" if result['compliant'] else "❌"
        score = result['compliance_score']
        
        print(f"\n{status} {hook_name}")
        print(f"   Score: {score:.2f}")
        
        if result['violations']:
            print(f"   Violations ({len(result['violations'])}):")
            for violation in result['violations']:
                severity = violation.get('severity', 'medium')
                message = violation.get('message', '')
                print(f"     • [{severity.upper()}] {message}")
    
    # Generate JSON report
    report_path = hooks_dir.parent / 'hook_compliance_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Return exit code based on compliance
    if results['summary']['compliance_rate'] < 1.0:
        print(f"\n⚠️  Some hooks are not L9 compliant")
        return 1
    else:
        print(f"\n🎉 All hooks are L9 compliant!")
        return 0


if __name__ == "__main__":
    sys.exit(main())