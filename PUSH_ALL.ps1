# Unified Push Script for IPL Prediction Engine v2.0
# Pushes to GitHub (Code & History) and Hugging Face (Live Space)

Write-Host "🚀 Starting Unified v2.0 Push & Sync..." -ForegroundColor Cyan

# 1. GitHub
Write-Host "📦 Pushing to GitHub..." -ForegroundColor Yellow
git add .
git commit -m "Production Polish: Fix 500 error at /matches, implement NumPy serialization, and rename archive routes."
git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ GitHub Sync Failed. Check your git status." -ForegroundColor Red
} else {
    Write-Host "✅ GitHub Sync Complete!" -ForegroundColor Green
}

# 2. Hugging Face
Write-Host "🌌 Syncing to Hugging Face Spaces (direct API upload)..." -ForegroundColor Yellow
python sync_to_hf.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Hugging Face Sync Failed. Check sync_to_hf.py logs." -ForegroundColor Red
} else {
    Write-Host "🎉 ALL SYSTEMS SYNCED! Match Prediction Engine is LIVE." -ForegroundColor Green
}
