@echo off
setlocal

set CMAKE="C:\Program Files\CMake\bin\cmake.exe"
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set BUILD=%SRC%build
set CONFIG=Release

if "%1"=="clean" (
    echo Cleaning build directory...
    if exist "%BUILD%" rmdir /s /q "%BUILD%"
    echo Done.
    exit /b 0
)

if "%1"=="debug" set CONFIG=Debug

echo Configuring (%CONFIG%)...
%CMAKE% -B "%BUILD%" -G "Visual Studio 17 2022" -S "%SRC%" || exit /b 1

echo Building...
%CMAKE% --build "%BUILD%" --config %CONFIG% || exit /b 1

echo.
echo Build complete: %BUILD%\%CONFIG%\grrt-cli.exe
