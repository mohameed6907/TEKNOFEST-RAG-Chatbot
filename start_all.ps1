# start_all.ps1
# PowerShell script to run both LangGraph Dev server and FastAPI Web App

Write-Host "Checking if Docker is running..." -ForegroundColor Cyan
docker info >$null 2>&1
if ($LastExitCode -ne 0) {
    Write-Warning "Docker is not running! Please start Docker Desktop first. 'langgraph dev' requires Docker."
    Exit 1
}

Write-Host "Starting LangGraph Dev Server (LangGraph Studio)..." -ForegroundColor Green
$langgraphProcess = Start-Process langgraph -ArgumentList "dev" -PassThru -NoNewWindow

Write-Host "Waiting 5 seconds for LangGraph server to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

Write-Host "Starting FastAPI Chat Server..." -ForegroundColor Green
$fastapiProcess = Start-Process uvicorn -ArgumentList "app.main:app --port 8010 --reload" -PassThru -NoNewWindow

Write-Host "Both servers are running successfully!" -ForegroundColor Yellow
Write-Host "- LangGraph Studio Local URL: http://localhost:2024 (or as printed by langgraph dev)" -ForegroundColor Cyan
Write-Host "- Web Chat UI: http://localhost:8010" -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this window to stop both servers." -ForegroundColor Yellow

# Wait for user key press to exit and clean up processes
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "`nStopping servers..." -ForegroundColor Red
    if ($langgraphProcess) {
        Stop-Process -Id $langgraphProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($fastapiProcess) {
        Stop-Process -Id $fastapiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopped all services." -ForegroundColor Green
}
