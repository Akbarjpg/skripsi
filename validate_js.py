#!/usr/bin/env python3
"""
Validate JavaScript syntax in HTML templates
"""

import os
import re
from pathlib import Path

def validate_html_js(file_path):
    """Validate JavaScript syntax in HTML file"""
    print(f"\n🔍 Validating: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for basic syntax issues
        issues = []
        
        # Check script tags are properly closed  
        # Count all script opening tags
        all_script_opens = len(re.findall(r'<script[^>]*>', content))
        script_closes = content.count('</script>')
        
        # External scripts are self-closing, so don't need </script>
        external_scripts = len(re.findall(r'<script[^>]*src=[^>]*></script>', content))
        
        # Internal scripts need matching </script>
        internal_opens = all_script_opens - external_scripts
        
        if internal_opens != script_closes:
            issues.append(f"Script tag mismatch: {internal_opens} internal opens, {script_closes} closes")
        
        # Check for unclosed brackets in JS
        js_blocks = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        for i, js_block in enumerate(js_blocks):
            if 'src=' in js_block[:50]:  # Skip external scripts
                continue
                
            # Count brackets
            open_braces = js_block.count('{')
            close_braces = js_block.count('}')
            open_parens = js_block.count('(')
            close_parens = js_block.count(')')
            
            if open_braces != close_braces:
                issues.append(f"JS Block {i+1}: Brace mismatch ({open_braces} open, {close_braces} close)")
            
            if open_parens != close_parens:
                issues.append(f"JS Block {i+1}: Parenthesis mismatch ({open_parens} open, {close_parens} close)")
        
        # Check for function definitions
        functions = re.findall(r'function\s+(\w+)\s*\(', content)
        window_functions = re.findall(r'window\.(\w+)\s*=', content)
        
        # Check onclick handlers
        onclick_handlers = re.findall(r'onclick="(\w+)\(\)"', content)
        
        # Validation results
        if not issues:
            print("✅ No syntax issues found")
        else:
            print("❌ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        
        print(f"📊 Statistics:")
        print(f"   - Functions defined: {len(functions)} {functions}")
        print(f"   - Window functions: {len(window_functions)} {window_functions}")
        print(f"   - Onclick handlers: {len(onclick_handlers)} {onclick_handlers}")
        
        # Check if onclick handlers have corresponding functions
        missing_functions = []
        all_functions = functions + window_functions
        for handler in onclick_handlers:
            if handler not in all_functions:
                missing_functions.append(handler)
        
        if missing_functions:
            print(f"⚠️ Missing functions for onclick: {missing_functions}")
        else:
            print("✅ All onclick handlers have corresponding functions")
        
        return len(issues) == 0 and len(missing_functions) == 0
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """Main validation function"""
    print("🧪 JavaScript Validation Tool")
    print("=" * 50)
    
    templates_dir = Path("src/web/templates")
    
    html_files = [
        "face_detection_test.html",
        "face_detection_test_fixed.html", 
        "face_detection_clean.html",
        "simple_camera_test.html",
        "attendance.html"
    ]
    
    results = {}
    
    for html_file in html_files:
        file_path = templates_dir / html_file
        if file_path.exists():
            results[html_file] = validate_html_js(file_path)
        else:
            print(f"\n❌ File not found: {file_path}")
            results[html_file] = False
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    for file_name, is_valid in results.items():
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"{file_name:<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All files passed validation!")
        print("✅ Ready for production use")
    else:
        print("\n⚠️ Some files have issues")
        print("🔧 Please check the detailed output above")
    
    return all_passed

if __name__ == "__main__":
    main()
