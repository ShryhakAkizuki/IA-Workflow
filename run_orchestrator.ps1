#Requires -Version 5.0
<#
run_orchestrator.ps1
====================
Ejecuta el orquestador (src/main.py) con el venv del PROYECTO.

El STT (ScribeCrate Live) y el LLM (Docker Model Runner) corren como
servicios Docker, fuera del proceso de Python; el venv solo necesita
numpy, sounddevice, openai y websocket-client (requirements.txt en la raiz).

Dónde busca/crea el venv (en este orden):
  1. $env:ORCH_VENV   (ruta a un .venv existente)
  2. .\.venv          (en la raiz del repo; se crea solo si no existe)

Uso:
  powershell -ExecutionPolicy Bypass -File .\run_orchestrator.ps1

La configuracion (variables de entorno) se lee de .env en la raiz del repo.
No hay valores por defecto: si falta alguna variable requerida, el arranque
falla indicando cuales faltan (ver .env.example).

Salidas:
  0 = OK · 1 = no se pudo usar/crear el venv · 2 = dependencias faltantes
  3 = no existe .env · resto = código de salida de src.main
#>
$ErrorActionPreference = 'Stop'

# --- 1. Localizar o crear el venv del proyecto ------------------------------
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = $env:ORCH_VENV
if (-not $VenvDir) {
    $VenvDir = Join-Path $Root '.venv'
}

$Py = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $Py)) {
    Write-Host "[run] venv no encontrado en $VenvDir ; creándolo..."
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pythonExe) {
        Write-Host 'ERROR: no hay un python en el PATH para crear el venv.'
        exit 1
    }
    & $pythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: fallo al crear el venv con $pythonExe"
        exit 1
    }
    # Actualiza pip dentro del venv.
    & $Py -m pip install --upgrade pip --quiet
}
if (-not (Test-Path $Py)) {
    Write-Host "ERROR: tras el intento, no existe $Py"
    exit 1
}
Write-Host "[run] venv: $VenvDir"

# --- 2. Verificar dependencias críticas -------------------------------------
$Check = @'
import importlib.util, sys
mods = ["sounddevice", "numpy", "openai", "websocket"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
if missing:
    print("FALTAN: " + ", ".join(missing))
    sys.exit(1)
'@
$tmp = Join-Path $env:TEMP ("check_orch_deps_" + [guid]::NewGuid().ToString('N') + ".py")
Set-Content -Path $tmp -Value $Check -Encoding utf8
try {
    & $Py $tmp
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}
if ($LASTEXITCODE -ne 0) {
    Write-Host 'ERROR: faltan dependencias en el venv. Instalar con:'
    Write-Host "  & '$Py' -m pip install -r requirements.txt"
    exit 2
}

# --- 3. Verificar la configuracion (.env) ------------------------------------
$EnvFile = Join-Path $Root '.env'
if (-not (Test-Path $EnvFile)) {
    Write-Host 'ERROR: no existe .env en la raiz del repo. Copia .env.example a .env.'
    exit 3
}

# --- 4. Ejecutar el orquestador ----------------------------------------------
$env:PYTHONUNBUFFERED = '1'
Push-Location $Root
try {
    & $Py -u -m src.main
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
