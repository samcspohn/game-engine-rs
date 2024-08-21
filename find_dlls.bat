FOR /F %%i IN ('where.exe /r .\target\%1 assimp.dll') DO (copy %%i .\target\%1\deps)
@REM FOR /F \"tokens=*\" %%g IN ('where.exe /r .\target\release assimp.dll') do (SET PATH=%%g;${env:PATH})