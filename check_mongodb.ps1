# MongoDB Service Management Script for Windows
# This script helps you check and manage MongoDB service

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MongoDB Service Manager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if MongoDB service exists
$mongoService = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue

if ($mongoService) {
    Write-Host "MongoDB Service Found!" -ForegroundColor Green
    Write-Host "Status: $($mongoService.Status)" -ForegroundColor Yellow
    Write-Host ""
    
    if ($mongoService.Status -eq "Running") {
        Write-Host "✓ MongoDB is currently running" -ForegroundColor Green
        Write-Host "  Connection: mongodb://localhost:27017/" -ForegroundColor Cyan
    } else {
        Write-Host "✗ MongoDB is not running" -ForegroundColor Red
        Write-Host ""
        $start = Read-Host "Do you want to start MongoDB? (y/n)"
        
        if ($start -eq "y" -or $start -eq "Y") {
            try {
                Start-Service -Name "MongoDB"
                Write-Host "✓ MongoDB service started successfully!" -ForegroundColor Green
            } catch {
                Write-Host "✗ Failed to start MongoDB. Try running as Administrator." -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "MongoDB Service Not Found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "MongoDB might not be installed or not installed as a service." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "1. Install MongoDB from: https://www.mongodb.com/try/download/community" -ForegroundColor White
    Write-Host "2. Use MongoDB Atlas (cloud): https://www.mongodb.com/cloud/atlas" -ForegroundColor White
    Write-Host "3. Check if mongod.exe is in your PATH and run: mongod" -ForegroundColor White
    Write-Host ""
    
    # Check if mongod.exe exists in PATH
    $mongod = Get-Command mongod -ErrorAction SilentlyContinue
    if ($mongod) {
        Write-Host "✓ mongod.exe found in PATH: $($mongod.Source)" -ForegroundColor Green
        Write-Host ""
        $startManual = Read-Host "Start MongoDB manually? (y/n)"
        if ($startManual -eq "y" -or $startManual -eq "Y") {
            Write-Host "Starting MongoDB..." -ForegroundColor Yellow
            Write-Host "Press Ctrl+C to stop MongoDB when done" -ForegroundColor Yellow
            Write-Host ""
            mongod
        }
    } else {
        Write-Host "✗ mongod.exe not found in PATH" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Ensure MongoDB is running" -ForegroundColor White
Write-Host "2. Run: python test_mongodb.py" -ForegroundColor White
Write-Host "3. Run: streamlit run app.py" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
