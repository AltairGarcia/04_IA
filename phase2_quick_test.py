#!/usr/bin/env python3
"""
Phase 2 Security Component Quick Test
====================================

Quick validation test for Phase 2 security components to verify they can be
imported and instantiated without errors.

Usage:
    python phase2_quick_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all security components can be imported."""
    try:
        from phase2_advanced_security import (
            AdvancedSecurityManager,
            JWTAuthenticationManager,
            EncryptionManager,
            SecurityAuditLogger,
            IntrusionDetectionSystem,
            SecureSessionManager,
            SecurityVulnerabilityScanner,
            SECURITY_CONFIG,
            SecurityUser,
            SecurityThreat
        )
        print("✅ All security components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_instantiation():
    """Test that all security components can be instantiated."""
    try:
        from phase2_advanced_security import (
            AdvancedSecurityManager,
            JWTAuthenticationManager,
            EncryptionManager,
            SecurityAuditLogger,
            IntrusionDetectionSystem,
            SecureSessionManager,
            SecurityVulnerabilityScanner
        )
        
        # Test instantiation
        auth_manager = JWTAuthenticationManager()
        print("✅ JWTAuthenticationManager instantiated")
        
        encryption_manager = EncryptionManager()
        print("✅ EncryptionManager instantiated")
        
        audit_logger = SecurityAuditLogger("test_audit.db")
        print("✅ SecurityAuditLogger instantiated")
        
        ids = IntrusionDetectionSystem()
        print("✅ IntrusionDetectionSystem instantiated")
        
        session_manager = SecureSessionManager()
        print("✅ SecureSessionManager instantiated")
        
        scanner = SecurityVulnerabilityScanner()
        print("✅ SecurityVulnerabilityScanner instantiated")
        
        security_manager = AdvancedSecurityManager()
        print("✅ AdvancedSecurityManager instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

async def test_basic_functionality():
    """Test basic functionality of key components."""
    try:
        from phase2_advanced_security import (
            JWTAuthenticationManager,
            EncryptionManager,
            SecurityUser
        )
        
        # Test authentication manager
        auth_manager = JWTAuthenticationManager()
        test_user = await auth_manager.create_user("testuser", "test@example.com", "SecurePass123!")
        if test_user and test_user.username == "testuser":
            print("✅ User creation works")
        else:
            print("❌ User creation failed")
            return False
        
        # Test token creation
        token = await auth_manager.create_access_token(test_user)
        if token:
            print("✅ JWT token creation works")
        else:
            print("❌ JWT token creation failed")
            return False
        
        # Test token verification
        verified_user = await auth_manager.verify_token(token)
        if verified_user and verified_user.user_id == test_user.user_id:
            print("✅ JWT token verification works")
        else:
            print("❌ JWT token verification failed")
            return False
        
        # Test encryption manager
        encryption_manager = EncryptionManager()
        test_data = "This is test data for encryption"
        encrypted = await encryption_manager.encrypt(test_data)
        if encrypted and encrypted != test_data:
            print("✅ Data encryption works")
        else:
            print("❌ Data encryption failed")
            return False
        
        decrypted = await encryption_manager.decrypt(encrypted)
        if decrypted == test_data:
            print("✅ Data decryption works")
        else:
            print("❌ Data decryption failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
        return False

def main():
    """Main test function."""
    print("🔐 Phase 2 Security Component Quick Test")
    print("=" * 45)
    
    # Test imports
    print("\n📦 Testing Imports...")
    if not test_imports():
        return 1
    
    # Test instantiation
    print("\n🏗️  Testing Instantiation...")
    if not test_instantiation():
        return 1
    
    # Test basic functionality
    print("\n⚙️  Testing Basic Functionality...")
    try:
        if not asyncio.run(test_basic_functionality()):
            return 1
    except Exception as e:
        print(f"❌ Async test error: {e}")
        return 1
    
    print("\n🎉 All quick tests passed! Phase 2 security components are functional.")
    print("\n🚀 Ready for integration with existing applications.")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
