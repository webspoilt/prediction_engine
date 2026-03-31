import os
from huggingface_hub import HfApi
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)
try:
    files = api.list_repo_files("zeroday01/predictionsingle")
    print(f"✅ Files in Hub: {len(files)}")
    for f in files[:5]:
        print(f" - {f}")
except Exception as e:
    print(f"❌ Error: {e}")
