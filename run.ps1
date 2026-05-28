# 必ずプロジェクト venv の Python で起動（Store 版 python との混在を防ぐ）
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "venv がありません: $Py`npython -m venv .venv のあと依存関係をインストールしてください。"
}
& $Py -m app.main @args
