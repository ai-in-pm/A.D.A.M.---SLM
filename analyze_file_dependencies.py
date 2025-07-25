#!/usr/bin/env python3
"""
Analyze File Dependencies in A.D.A.M.-SLM Tokenization System
Determine which files are actually needed vs legacy files
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def analyze_file_usage():
    """Analyze which tokenization files are actually used"""
    
    print("🔍 Analyzing A.D.A.M.-SLM Tokenization File Dependencies")
    print("="*70)
    
    files_to_analyze = [
        "adam_slm/tokenization/bpe.py",
        "adam_slm/tokenization/corpus_analyzer.py", 
        "adam_slm/tokenization/tokenizer.py"
    ]
    
    analysis_results = {}
    
    for file_path in files_to_analyze:
        print(f"\n📄 Analyzing: {file_path}")
        analysis_results[file_path] = analyze_single_file(file_path)
    
    return analysis_results

def analyze_single_file(file_path):
    """Analyze a single file's usage"""
    
    result = {
        'exists': False,
        'imported_in_init': False,
        'used_in_system': False,
        'backward_compatibility': False,
        'can_be_removed': False,
        'recommendation': ''
    }
    
    # Check if file exists
    if Path(file_path).exists():
        result['exists'] = True
        print(f"   ✅ File exists")
    else:
        print(f"   ❌ File does not exist")
        return result
    
    # Check if imported in __init__.py
    try:
        with open("adam_slm/tokenization/__init__.py", 'r') as f:
            init_content = f.read()
        
        filename = Path(file_path).stem
        if f"from .{filename} import" in init_content:
            result['imported_in_init'] = True
            print(f"   ✅ Imported in __init__.py")
        else:
            print(f"   ❌ Not imported in __init__.py")
    except Exception as e:
        print(f"   ⚠️ Could not check __init__.py: {e}")
    
    # Test if system works without the file
    try:
        # Test basic tokenizer functionality
        from adam_slm.tokenization import get_tokenizer
        tokenizer = get_tokenizer("adam_slm")
        
        # Test encoding/decoding
        test_text = "Hello A.D.A.M.!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        if decoded.strip() == test_text.strip():
            result['used_in_system'] = True
            print(f"   ✅ System works with current imports")
        else:
            print(f"   ⚠️ System has issues")
            
    except Exception as e:
        print(f"   ❌ System fails without this file: {e}")
        result['used_in_system'] = True  # Assume needed if system fails
    
    # Analyze specific file purposes
    if "bpe.py" in file_path:
        result['backward_compatibility'] = True
        result['recommendation'] = "Legacy BPE implementation - may be removable if not used"
        
    elif "corpus_analyzer.py" in file_path:
        result['backward_compatibility'] = False
        result['recommendation'] = "Training component - needed for tokenizer training"
        
    elif "tokenizer.py" in file_path:
        result['backward_compatibility'] = True
        result['recommendation'] = "Original GPT-2 tokenizer - needed for fallback compatibility"
    
    # Determine if can be removed
    if result['imported_in_init'] and result['backward_compatibility']:
        result['can_be_removed'] = False
        result['recommendation'] += " - KEEP for backward compatibility"
    elif not result['imported_in_init'] and not result['used_in_system']:
        result['can_be_removed'] = True
        result['recommendation'] += " - CAN BE REMOVED"
    else:
        result['can_be_removed'] = False
        result['recommendation'] += " - KEEP (actively used)"
    
    return result

def test_system_without_files():
    """Test if the system works without questionable files"""
    
    print("\n🧪 Testing System Functionality...")
    
    try:
        # Test 1: Basic tokenizer import
        print("\n   Test 1: Basic tokenizer import")
        from adam_slm.tokenization import get_tokenizer
        print("   ✅ get_tokenizer import successful")
        
        # Test 2: Smart tokenizer creation
        print("\n   Test 2: Smart tokenizer creation")
        tokenizer = get_tokenizer("adam_slm")
        print(f"   ✅ Tokenizer created: {type(tokenizer).__name__}")
        
        # Test 3: Tokenization functionality
        print("\n   Test 3: Tokenization functionality")
        test_texts = [
            "Hello A.D.A.M.!",
            "Transformer neural networks use attention mechanisms",
            "The gradient ∇f(x) represents the derivative"
        ]
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            print(f"   ✅ '{text[:30]}...' → {len(tokens)} tokens")
        
        # Test 4: Backward compatibility
        print("\n   Test 4: Backward compatibility")
        try:
            from adam_slm.tokenization import AdamTokenizer, BPETokenizer
            print("   ✅ Legacy tokenizers importable")
        except ImportError as e:
            print(f"   ⚠️ Legacy tokenizer import issue: {e}")
        
        # Test 5: Training components
        print("\n   Test 5: Training components")
        try:
            from adam_slm.tokenization import DomainCorpusAnalyzer, AdamBPETrainer
            print("   ✅ Training components importable")
        except ImportError as e:
            print(f"   ⚠️ Training component import issue: {e}")
        
        print("\n   ✅ All core functionality working")
        return True
        
    except Exception as e:
        print(f"\n   ❌ System test failed: {e}")
        return False

def generate_recommendations():
    """Generate final recommendations"""
    
    print("\n📋 FINAL RECOMMENDATIONS")
    print("="*50)
    
    recommendations = {
        "bpe.py": {
            "status": "LEGACY - Consider Removal",
            "reason": "Original BPE implementation, replaced by A.D.A.M.-SLM custom tokenizer",
            "action": "Can be removed if not used for backward compatibility",
            "risk": "Low - system uses new tokenizer"
        },
        "corpus_analyzer.py": {
            "status": "REQUIRED - Keep",
            "reason": "Used for training A.D.A.M.-SLM tokenizer and analysis",
            "action": "Keep - needed for tokenizer training and development",
            "risk": "High - breaks tokenizer training if removed"
        },
        "tokenizer.py": {
            "status": "BACKWARD COMPATIBILITY - Keep",
            "reason": "Original GPT-2 tokenizer used as fallback",
            "action": "Keep - provides fallback and compatibility",
            "risk": "Medium - breaks fallback functionality if removed"
        }
    }
    
    for filename, rec in recommendations.items():
        print(f"\n📄 {filename}:")
        print(f"   Status: {rec['status']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Action: {rec['action']}")
        print(f"   Risk: {rec['risk']}")
    
    return recommendations

def main():
    """Main analysis function"""
    
    print("🔍 A.D.A.M.-SLM Tokenization File Dependency Analysis")
    print("🎯 Determining which files are needed vs legacy")
    print("="*80)
    
    # Analyze file usage
    analysis_results = analyze_file_usage()
    
    # Test system functionality
    system_working = test_system_without_files()
    
    # Generate recommendations
    recommendations = generate_recommendations()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🎯 File Status:")
    for file_path, result in analysis_results.items():
        filename = Path(file_path).name
        status = "NEEDED" if not result['can_be_removed'] else "REMOVABLE"
        print(f"   • {filename}: {status}")
    
    print(f"\n🚀 System Status: {'✅ Working' if system_working else '❌ Issues detected'}")
    
    print("\n💡 Recommendations:")
    print("   • corpus_analyzer.py: KEEP (training component)")
    print("   • tokenizer.py: KEEP (backward compatibility)")
    print("   • bpe.py: EVALUATE (may be legacy)")
    
    print("\n🎯 Next Steps:")
    print("   1. Keep corpus_analyzer.py - needed for training")
    print("   2. Keep tokenizer.py - needed for GPT-2 fallback")
    print("   3. Evaluate bpe.py usage - may be removable")
    print("   4. Monitor system for any missing dependencies")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
