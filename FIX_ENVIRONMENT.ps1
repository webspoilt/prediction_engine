# 🛠️ Fix Environment & Sync Model (Windows PowerShell)

# 1. Set the Hugging Face Token for this session
$env:HF_TOKEN = "your_write_token_here"
Write-Host "✅ HF_TOKEN set successfully." -ForegroundColor Cyan

# 2. Activate the correct Virtual Environment
if (Test-Path ".venv312\Scripts\Activate.ps1") {
    .venv312\Scripts\Activate.ps1
    Write-Host "✅ Virtual Environment (.venv312) activated." -ForegroundColor Cyan
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    .venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual Environment (.venv) activated." -ForegroundColor Cyan
} else {
    Write-Host "⚠️ Virtual Environment not found. Installing dependencies in global Python..." -ForegroundColor Yellow
}

# 3. Ensure dependencies are installed
Write-Host "📦 Verifying dependencies..." -ForegroundColor Gray
pip install -r backend/requirements.txt numpy pandas xgboost torch scrapling huggingface_hub asyncpg --quiet

# 4. Run the Hugging Face Initializer
Write-Host "🚀 Running Model Sync to Hugging Face..." -ForegroundColor Green
python init_huggingface.py

Write-Host "🏁 Setup complete. You can now run 'python backend/api_server.py' to start the live discovery engine." -ForegroundColor Green
