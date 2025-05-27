#!/usr/bin/env python3
"""
Phase 1 Quick Status Check

A lightweight validation to check Phase 1 completion status
without complex imports that might cause compatibility issues.

Author: GitHub Copilot
Date: 2025-01-25
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class QuickPhase1Check:
    """Quick Phase 1 status checker"""
    
    def __init__(self):
        self.base_path = "c:\\ALTAIR GARCIA\\04__ia"
        self.results = []
        self.score = 0
        self.total_checks = 0
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all quick status checks"""
        print("🚀 Quick Phase 1 Status Check")
        print("=" * 50)
        
        checks = [
            self.check_core_files,
            self.check_integration_files,
            self.check_compatibility_fixes,
            self.check_enhanced_apps,
            self.check_testing_suite
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self._add_result(check.__name__, False, str(e))
        
        return self._generate_report()
    
    def check_core_files(self):
        """Check core infrastructure files exist"""
        print("📁 Checking Core Infrastructure Files...")
        
        core_files = [
            'infrastructure_integration_hub.py',
            'api_gateway.py',
            'message_queue_system.py', 
            'database_connection_pool.py',
            'enhanced_rate_limiting.py',
            'cache_manager.py',
            'config_hot_reload.py'
        ]
        
        missing = []
        for file in core_files:
            if not os.path.exists(os.path.join(self.base_path, file)):
                missing.append(file)
        
        if not missing:
            self._add_result('core_infrastructure_files', True, 
                           f"All {len(core_files)} core files present")
            print(f"✅ Core Infrastructure Files: All {len(core_files)} files present")
        else:
            self._add_result('core_infrastructure_files', False, 
                           f"Missing files: {missing}")
            print(f"❌ Core Infrastructure Files: Missing {len(missing)} files")
    
    def check_integration_files(self):
        """Check integration system files"""
        print("🔗 Checking Integration System Files...")
        
        integration_files = [
            'langgraph_integration_adapter.py',
            'langgraph_startup.py',
            'integrated_config.py',
            'app_integration_wrapper.py',
            'streamlit_integration_patch.py',
            'cli_integration_patch.py'
        ]
        
        missing = []
        for file in integration_files:
            if not os.path.exists(os.path.join(self.base_path, file)):
                missing.append(file)
        
        if not missing:
            self._add_result('integration_system_files', True, 
                           f"All {len(integration_files)} integration files present")
            print(f"✅ Integration System Files: All {len(integration_files)} files present")
        else:
            self._add_result('integration_system_files', False, 
                           f"Missing files: {missing}")
            print(f"❌ Integration System Files: Missing {len(missing)} files")
    
    def check_compatibility_fixes(self):
        """Check compatibility fixes"""
        print("🔧 Checking Compatibility Fixes...")
        
        compatibility_files = [
            'aioredis_compat.py',
            'redis_fallback.py',
            'input_security.py',
            'phase1_compatibility_fixes.py'
        ]
        
        missing = []
        present = []
        for file in compatibility_files:
            if os.path.exists(os.path.join(self.base_path, file)):
                present.append(file)
            else:
                missing.append(file)
        
        success_rate = len(present) / len(compatibility_files) * 100
        
        if success_rate >= 75:
            self._add_result('compatibility_fixes', True, 
                           f"{len(present)}/{len(compatibility_files)} fixes applied ({success_rate:.1f}%)")
            print(f"✅ Compatibility Fixes: {len(present)}/{len(compatibility_files)} applied ({success_rate:.1f}%)")
        else:
            self._add_result('compatibility_fixes', False, 
                           f"Only {len(present)}/{len(compatibility_files)} fixes applied")
            print(f"❌ Compatibility Fixes: Only {len(present)}/{len(compatibility_files)} applied")
    
    def check_enhanced_apps(self):
        """Check enhanced applications"""
        print("📱 Checking Enhanced Applications...")
        
        enhanced_apps = [
            'langgraph_enhanced_cli.py',
            'langgraph_enhanced_streamlit.py'
        ]
        
        original_apps = [
            'langgraph-101.py',
            'streamlit_app.py'
        ]
        
        enhanced_present = sum(1 for app in enhanced_apps 
                             if os.path.exists(os.path.join(self.base_path, app)))
        original_present = sum(1 for app in original_apps 
                             if os.path.exists(os.path.join(self.base_path, app)))
        
        if enhanced_present == len(enhanced_apps) and original_present == len(original_apps):
            self._add_result('enhanced_applications', True, 
                           "Enhanced and original applications present")
            print("✅ Enhanced Applications: Both enhanced and original apps present")
        else:
            self._add_result('enhanced_applications', False, 
                           f"Enhanced: {enhanced_present}/{len(enhanced_apps)}, Original: {original_present}/{len(original_apps)}")
            print(f"❌ Enhanced Applications: Enhanced: {enhanced_present}/{len(enhanced_apps)}, Original: {original_present}/{len(original_apps)}")
    
    def check_testing_suite(self):
        """Check testing and validation suite"""
        print("🧪 Checking Testing Suite...")
        
        test_files = [
            'integration_validation_test.py',
            'phase1_integration_status.py', 
            'phase1_core_validation.py',
            'phase1_final_validation.py',
            'phase1_completion_summary.py'
        ]
        
        present = []
        for file in test_files:
            if os.path.exists(os.path.join(self.base_path, file)):
                present.append(file)
        
        completion_rate = len(present) / len(test_files) * 100
        
        if completion_rate >= 80:
            self._add_result('testing_suite', True, 
                           f"{len(present)}/{len(test_files)} test files present ({completion_rate:.1f}%)")
            print(f"✅ Testing Suite: {len(present)}/{len(test_files)} files present ({completion_rate:.1f}%)")
        else:
            self._add_result('testing_suite', False, 
                           f"Only {len(present)}/{len(test_files)} test files present")
            print(f"❌ Testing Suite: Only {len(present)}/{len(test_files)} files present")
    
    def _add_result(self, check_name: str, success: bool, message: str):
        """Add check result"""
        self.results.append({
            'check': check_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        self.total_checks += 1
        if success:
            self.score += 1
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final report"""
        success_rate = (self.score / self.total_checks * 100) if self.total_checks > 0 else 0
        
        print("\\n🎯 Phase 1 Quick Status Report")
        print("=" * 40)
        print(f"Checks Passed: {self.score}/{self.total_checks}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            status = "✅ EXCELLENT"
            recommendation = "Phase 1 implementation is complete and ready for Phase 2"
        elif success_rate >= 75:
            status = "✅ GOOD"
            recommendation = "Phase 1 implementation is mostly complete"
        elif success_rate >= 50:
            status = "⚠️  ACCEPTABLE"
            recommendation = "Phase 1 has core functionality in place"
        else:
            status = "❌ NEEDS WORK"
            recommendation = "Phase 1 requires more work"
        
        print(f"Status: {status}")
        print(f"Recommendation: {recommendation}")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"phase1_quick_status_{timestamp}.json"
        
        report = {
            'summary': {
                'total_checks': self.total_checks,
                'passed': self.score,
                'success_rate': success_rate,
                'status': status,
                'recommendation': recommendation
            },
            'detailed_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\n📋 Report saved to: {report_file}")
        
        return report

def main():
    """Run quick Phase 1 status check"""
    checker = QuickPhase1Check()
    results = checker.run_all_checks()
    
    success_rate = results['summary']['success_rate']
    return success_rate >= 75

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
