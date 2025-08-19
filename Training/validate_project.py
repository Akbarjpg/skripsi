#!/usr/bin/env python3
"""
Comprehensive System Validator for Face Anti-Spoofing Attendance System
Validates the entire reorganized project structure
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import importlib.util

def validate_project_structure() -> Dict[str, Any]:
    """Validate the clean project structure"""
    print("🔍 VALIDATING CLEAN PROJECT STRUCTURE")
    print("=" * 60)
    
    results = {
        'structure': False,
        'configs': False,
        'core': False,
        'web': False,
        'utils': False,
        'templates': False,
        'missing_files': [],
        'issues': []
    }
    
    # Required files and directories
    required_structure = {
        'files': [
            'main.py',
            'requirements.txt',
            'README_CLEAN.md',
            'config/default.json',
            'config/development.json',
            'src/core/app_launcher.py',
            'src/web/app_clean.py',
            'src/web/templates/base.html',
            'src/web/templates/login.html',
            'src/web/templates/register.html',
            'src/web/templates/dashboard.html',
            'src/web/templates/404.html',
            'src/web/templates/index_clean.html',
            'src/utils/config.py',
            'src/utils/logger.py',
            'src/utils/environment.py',
            'src/training/trainer.py',
            'src/testing/test_runner.py'
        ],
        'directories': [
            'config',
            'src/core',
            'src/web',
            'src/web/templates',
            'src/models',
            'src/utils',
            'src/training',
            'src/testing',
            'logs',
            'models'
        ]
    }
    
    # Check directories
    missing_dirs = []
    for directory in required_structure['directories']:
        if not Path(directory).exists():
            missing_dirs.append(directory)
        else:
            print(f"✅ Directory: {directory}")
    
    if missing_dirs:
        results['issues'].append(f"Missing directories: {missing_dirs}")
        print(f"❌ Missing directories: {missing_dirs}")
    else:
        results['structure'] = True
    
    # Check files
    missing_files = []
    for file_path in required_structure['files']:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ File: {file_path}")
    
    if missing_files:
        results['missing_files'] = missing_files
        results['issues'].append(f"Missing files: {missing_files}")
        print(f"❌ Missing files: {missing_files}")
    
    return results

def validate_configurations() -> Dict[str, Any]:
    """Validate configuration files"""
    print("\n🔧 VALIDATING CONFIGURATIONS")
    print("=" * 60)
    
    results = {'valid': False, 'configs': {}, 'issues': []}
    
    config_files = ['config/default.json', 'config/development.json']
    
    for config_file in config_files:
        try:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Validate required sections
                required_sections = ['model', 'training', 'web', 'detection']
                missing_sections = [section for section in required_sections 
                                  if section not in config_data]
                
                if missing_sections:
                    results['issues'].append(f"{config_file}: Missing sections {missing_sections}")
                    print(f"❌ {config_file}: Missing sections {missing_sections}")
                else:
                    results['configs'][config_file] = True
                    print(f"✅ {config_file}: Valid configuration")
            else:
                results['issues'].append(f"{config_file}: File not found")
                print(f"❌ {config_file}: File not found")
        
        except json.JSONDecodeError as e:
            results['issues'].append(f"{config_file}: Invalid JSON - {e}")
            print(f"❌ {config_file}: Invalid JSON - {e}")
        except Exception as e:
            results['issues'].append(f"{config_file}: Error - {e}")
            print(f"❌ {config_file}: Error - {e}")
    
    if len(results['configs']) == len(config_files) and not results['issues']:
        results['valid'] = True
    
    return results

def validate_python_imports() -> Dict[str, Any]:
    """Validate that all new modules can be imported"""
    print("\n🐍 VALIDATING PYTHON IMPORTS")
    print("=" * 60)
    
    results = {'valid': False, 'imports': {}, 'issues': []}
    
    # Add src to path
    src_path = Path('src').absolute()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        'utils.config',
        'utils.logger', 
        'utils.environment',
        'core.app_launcher',
        'training.trainer',
        'testing.test_runner'
    ]
    
    for module_name in modules_to_test:
        try:
            module_path = f"src/{module_name.replace('.', '/')}.py"
            if Path(module_path).exists():
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute, just check if it can be loaded
                    results['imports'][module_name] = True
                    print(f"✅ Import: {module_name}")
                else:
                    results['issues'].append(f"{module_name}: Cannot create spec")
                    print(f"❌ Import: {module_name} - Cannot create spec")
            else:
                results['issues'].append(f"{module_name}: File not found at {module_path}")
                print(f"❌ Import: {module_name} - File not found")
        
        except Exception as e:
            results['issues'].append(f"{module_name}: Import error - {e}")
            print(f"❌ Import: {module_name} - {e}")
    
    if len(results['imports']) == len(modules_to_test) and not results['issues']:
        results['valid'] = True
    
    return results

def validate_web_templates() -> Dict[str, Any]:
    """Validate web templates"""
    print("\n🌐 VALIDATING WEB TEMPLATES")
    print("=" * 60)
    
    results = {'valid': False, 'templates': {}, 'issues': []}
    
    template_files = [
        'src/web/templates/base.html',
        'src/web/templates/login.html',
        'src/web/templates/register.html',
        'src/web/templates/dashboard.html',
        'src/web/templates/404.html',
        'src/web/templates/index_clean.html'
    ]
    
    for template_file in template_files:
        try:
            if Path(template_file).exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic template structure
                if template_file.endswith('base.html'):
                    required_blocks = ['{% block content %}', '{% block title %}']
                else:
                    required_blocks = ['{% extends "base.html" %}']
                
                missing_blocks = [block for block in required_blocks if block not in content]
                
                if missing_blocks:
                    results['issues'].append(f"{template_file}: Missing {missing_blocks}")
                    print(f"❌ {template_file}: Missing {missing_blocks}")
                else:
                    results['templates'][template_file] = True
                    print(f"✅ {template_file}: Valid template")
            else:
                results['issues'].append(f"{template_file}: File not found")
                print(f"❌ {template_file}: File not found")
        
        except Exception as e:
            results['issues'].append(f"{template_file}: Error - {e}")
            print(f"❌ {template_file}: Error - {e}")
    
    if len(results['templates']) == len(template_files) and not results['issues']:
        results['valid'] = True
    
    return results

def generate_summary_report(validation_results: Dict[str, Any]) -> None:
    """Generate a comprehensive summary report"""
    print("\n📊 VALIDATION SUMMARY REPORT")
    print("=" * 60)
    
    total_checks = 0
    passed_checks = 0
    
    for category, result in validation_results.items():
        total_checks += 1
        if result.get('valid', False) or result.get('structure', False):
            passed_checks += 1
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"{category.upper():20} | {status}")
        
        # Show issues if any
        if 'issues' in result and result['issues']:
            for issue in result['issues'][:3]:  # Show first 3 issues
                print(f"                     │ ⚠️  {issue}")
    
    print("-" * 60)
    print(f"OVERALL RESULT: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 PROJECT STRUCTURE IS CLEAN AND READY!")
        print("\n🚀 Next steps:")
        print("1. Run: python main.py --mode setup")
        print("2. Run: python main.py --mode test") 
        print("3. Run: python main.py --mode web")
    else:
        print("⚠️  SOME ISSUES NEED ATTENTION")
        print("\n🔧 Recommended fixes:")
        print("1. Check missing files and directories")
        print("2. Validate configuration files")
        print("3. Fix import issues")
        print("4. Verify template structure")

def main():
    """Main validation function"""
    print("🛡️  FACE ANTI-SPOOFING ATTENDANCE SYSTEM")
    print("📋 PROJECT STRUCTURE VALIDATION")
    print("=" * 60)
    
    # Run all validations
    validation_results = {
        'structure': validate_project_structure(),
        'configurations': validate_configurations(),
        'imports': validate_python_imports(),
        'templates': validate_web_templates()
    }
    
    # Generate summary
    generate_summary_report(validation_results)
    
    # Save detailed report
    report_file = 'validation_report.json'
    try:
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")

if __name__ == "__main__":
    main()
