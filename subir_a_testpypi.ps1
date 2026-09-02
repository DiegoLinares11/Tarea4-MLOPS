# Publica el paquete en TestPyPI leyendo el token del archivo .env
#
#   Uso:   .\subir_a_testpypi.ps1
#
# El .env debe tener estas dos lineas (sin comillas):
#   TWINE_USERNAME=__token__
#   TWINE_PASSWORD=pypi-AgENdGVzdC5weXBpLm9yZw...
#
# El token nunca se escribe en la terminal ni queda en el historial de comandos.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Write-Host "No encuentro .env en esta carpeta." -ForegroundColor Red
    Write-Host "Copia .env.example a .env y pone tu token de TestPyPI adentro."
    exit 1
}

# --- Cargar el .env en variables de entorno de esta sesion ---
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $nombre = $matches[1]
        $valor = $matches[2].Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($nombre, $valor, "Process")
    }
}

if (-not $env:TWINE_PASSWORD) {
    Write-Host "El .env no define TWINE_PASSWORD." -ForegroundColor Red
    exit 1
}
Write-Host "Token cargado desde .env (usuario: $env:TWINE_USERNAME)" -ForegroundColor Green

# --- 1. Construir ---
Write-Host "`n[1/3] Construyendo el paquete..." -ForegroundColor Cyan
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force }
python -m build

# --- 2. Validar la metadata antes de subir ---
Write-Host "`n[2/3] Validando la metadata..." -ForegroundColor Cyan
python -m twine check dist/*

# --- 3. Subir a TestPyPI ---
Write-Host "`n[3/3] Subiendo a TestPyPI..." -ForegroundColor Cyan
python -m twine upload --repository testpypi dist/*

Write-Host "`nListo. El paquete deberia estar en:" -ForegroundColor Green
Write-Host "  https://test.pypi.org/project/act3-pipeline-mlops/"
Write-Host "`nPara comprobar que se instala:"
Write-Host "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ act3-pipeline-mlops"
