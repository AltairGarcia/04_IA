#!/usr/bin/env python3
"""
Integration Compatibility Wrapper for Python 3.13

This module provides compatibility wrappers for integration components
that experience issues with Python 3.13.
"""

import sys
import warnings
import logging
from typing import Any, Dict, Optional
import traceback

logger = logging.getLogger(__name__)

# Apply suppression
if sys.version_info >= (3, 13):
    warnings.filterwarnings("ignore", category=RuntimeWarning, 
                          message=".*TimeoutError.*duplicate base class.*")

class CompatibleIntegrationAdapter:
    """Compatibility wrapper for integration adapter"""
    
    def __init__(self):
        self.original_adapter = None
        self.fallback_mode = False
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize adapter with fallback"""
        try:
            from langgraph_integration_adapter import LangGraphIntegrationAdapter
            self.original_adapter = LangGraphIntegrationAdapter()
            logger.info("✅ Original integration adapter loaded")
        except Exception as e:
            logger.warning(f"⚠️ Using fallback adapter: {e}")
            self.fallback_mode = True
    
    def start(self):
        """Start adapter with error handling"""
        if self.original_adapter and not self.fallback_mode:
            try:
                return self.original_adapter.start()
            except Exception as e:
                logger.error(f"❌ Adapter start failed: {e}")
                self.fallback_mode = True
        
        logger.info("🔄 Running in fallback mode")
        return {"status": "fallback", "success": True}
    
    def stop(self):
        """Stop adapter safely"""
        if self.original_adapter and not self.fallback_mode:
            try:
                return self.original_adapter.stop()
            except Exception:
                pass
        return {"status": "stopped", "success": True}

class CompatibleCLIIntegration:
    """Compatibility wrapper for CLI integration"""
    
    def __init__(self):
        self.fallback_mode = False
        self._initialize()
    
    def _initialize(self):
        """Initialize with fallback"""
        try:
            from cli_integration_patch import CLIIntegrationPatch
            self.patch = CLIIntegrationPatch()
            logger.info("✅ CLI integration patch loaded")
        except Exception as e:
            logger.warning(f"⚠️ CLI integration fallback: {e}")
            self.fallback_mode = True
    
    def apply_patches(self):
        """Apply patches with error handling"""
        if not self.fallback_mode:
            try:
                return self.patch.apply_patches()
            except Exception:
                self.fallback_mode = True
        
        return {"status": "fallback", "patches_applied": 0}

class CompatibleStreamlitIntegration:
    """Compatibility wrapper for Streamlit integration"""
    
    def __init__(self):
        self.fallback_mode = False
        self._initialize()
    
    def _initialize(self):
        """Initialize with fallback"""
        try:
            from streamlit_integration_patch import StreamlitIntegrationPatch
            self.patch = StreamlitIntegrationPatch()
            logger.info("✅ Streamlit integration patch loaded")
        except Exception as e:
            logger.warning(f"⚠️ Streamlit integration fallback: {e}")
            self.fallback_mode = True
    
    def enhance_app(self):
        """Enhance app with error handling"""
        if not self.fallback_mode:
            try:
                return self.patch.enhance_app()
            except Exception:
                self.fallback_mode = True
        
        return {"status": "fallback", "enhancements": 0}

# Convenience factory functions
def get_integration_adapter():
    """Get compatible integration adapter"""
    return CompatibleIntegrationAdapter()

def get_cli_integration():
    """Get compatible CLI integration"""
    return CompatibleCLIIntegration()

def get_streamlit_integration():
    """Get compatible Streamlit integration"""
    return CompatibleStreamlitIntegration()
