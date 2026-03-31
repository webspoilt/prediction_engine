# 🚀 PUSH TO HUGGING FACE HUB (Automated Setup)

# 1. Install the Hugging Face CLI (If not already present)
Write-Host "📦 Installing Hugging Face CLI..." -ForegroundColor Cyan
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# 2. Login (Requires Interaction)
Write-Host "🔑 Authenticating with your Hugging Face credentials..." -ForegroundColor Yellow
Write-Host "Please follow the browser prompt or enter your Write Token." -ForegroundColor Gray
hf auth login

# 3. Push EVERYTHING to your Repo
# This includes the api_server.py, models/, and all necessary pipeline code.
Write-Host "🚀 Uploading Engine & Models to zeroday01/predictionsingle..." -ForegroundColor Green
hf upload zeroday01/predictionsingle . --exclude ".venv/*" ".venv312/*" ".pytest_cache/*" ".git/*" "__pycache__/*" "*.log" "pytest_*.txt"
Write-Host "✅ Success! Your engine is now synced to the Hub." -ForegroundColor Green

# 🏁 Reminder:
# If you are hosting on Hugging Face Spaces (Docker), 
# make sure your Dockerfile points to 'backend/api_server.py'.
