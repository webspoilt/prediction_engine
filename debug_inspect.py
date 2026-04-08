import sys
import os

# Ensure Python can find the local modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))
try:
    from ml_engine.hybrid_model import HybridEnsemble
    print(f"Module file: {sys.modules['ml_engine.hybrid_model'].__file__}")
    ensemble = HybridEnsemble()
    print(f"Attributes: {dir(ensemble)}")
    if hasattr(ensemble, 'load_models'):
        print("✅ load_models found")
    else:
        print("❌ load_models NOT found")
except Exception as e:
    print(f"Error: {e}")
