::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
setlocal EnableDelayedExpansion
REM === End-to-End custom-kernel test pipeline ===
REM Runs every kernel suite under lib\Runtime\Kernels\test\example that ships a
REM Makefile, on one arch, and prints a single pass/fail summary.
REM
REM Each suite is driven through its own Makefile rather than by re-issuing
REM hipcc commands here, so the build recipes stay in one place. They do drift:
REM gemm_fp16u2 and gemm_fp16i8 now link an extra autotune_stub.cpp that
REM gemm_fp16u3 does not, and duplicating that here would rot silently.
REM
REM Deliberately does NOT call env.bat: that configures MLIR, hipblaslt and
REM MIOpen for the hip-compiler pipelines and hardcodes one developer's paths.
REM These suites need only a HIP SDK and make.
REM
REM Usage:
REM   run_full_pipeline_kernel_tests.bat --arch gfx1151
REM   run_full_pipeline_kernel_tests.bat --suite matmul --quick
REM   run_full_pipeline_kernel_tests.bat --hip-sdk D:\rocm --gs 32

REM Captured before the parse loop: "shift" also shifts %0, so %~dp0 is only
REM valid here.
set "SCRIPT_DIR=%~dp0"

if not defined OFFLOAD_ARCH set "OFFLOAD_ARCH=gfx1150"
set "SUITE=all"
set "GS=128"
set "QUICK=0"
set "MAKE_EXE="
set "USER_HIP_SDK=%HIP_SDK%"
set "USER_PYTHON="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--arch"     (set "OFFLOAD_ARCH=%~2"  & shift & shift & goto parse)
if /i "%~1"=="--hip-sdk"  (set "USER_HIP_SDK=%~2"  & shift & shift & goto parse)
if /i "%~1"=="--python"   (set "USER_PYTHON=%~2"   & shift & shift & goto parse)
if /i "%~1"=="--make"     (set "MAKE_EXE=%~2"      & shift & shift & goto parse)
if /i "%~1"=="--suite"    (set "SUITE=%~2"         & shift & shift & goto parse)
if /i "%~1"=="--gs"       (set "GS=%~2"            & shift & shift & goto parse)
if /i "%~1"=="--quick"    (set "QUICK=1" & shift & goto parse)
if /i "%~1"=="--help"     goto usage
if /i "%~1"=="-h"         goto usage
echo Unknown option: %~1
goto usage
:parsed

if /i not "%SUITE%"=="all" if /i not "%SUITE%"=="matmul" if /i not "%SUITE%"=="gqa" (
  echo ERROR: --suite must be all, matmul or gqa ^(got "%SUITE%"^)
  exit /b 2
)

REM ---- locate make ----
if not defined MAKE_EXE (
  for %%M in (make.exe mingw32-make.exe) do (
    if not defined MAKE_EXE (
      for /f "delims=" %%P in ('where %%M 2^>nul') do if not defined MAKE_EXE set "MAKE_EXE=%%P"
    )
  )
)
if not defined MAKE_EXE (
  echo ERROR: no make found on PATH.
  echo        The per-suite Makefiles are the build recipe; install make
  echo        ^(MSYS2, Git-for-Windows, or WSL^) or pass --make ^<path^>.
  exit /b 1
)

REM ---- optional overrides forwarded to make ----
REM Left unset by default so each Makefile keeps its own OS-aware default
REM (the HIP SDK path differs between Windows and WSL).
REM Flat control flow on purpose: "exit /b" does not reliably propagate its
REM code out of nested parenthesised blocks.
set "MK_ARGS=OFFLOAD=--offload-arch=%OFFLOAD_ARCH%"
if not defined USER_HIP_SDK goto sdk_done
if exist "%USER_HIP_SDK%\bin\hipcc.exe" goto sdk_ok
echo ERROR: hipcc not found at "%USER_HIP_SDK%\bin\hipcc.exe"
exit /b 1
:sdk_ok
set "MK_ARGS=%MK_ARGS% HIP_SDK=%USER_HIP_SDK%"
:sdk_done
if defined USER_PYTHON set "MK_ARGS=%MK_ARGS% PYTHON=%USER_PYTHON%"

set "EXAMPLE_DIR=%SCRIPT_DIR%..\lib\Runtime\Kernels\test\example"
if not exist "%EXAMPLE_DIR%" (
  echo ERROR: example tree not found at "%EXAMPLE_DIR%"
  exit /b 1
)

set "PASSED=0"
set "FAILED=0"
set "SKIPPED=0"
set "SUMMARY="

echo.
echo ============================================================
echo  custom_kernels test pipeline
echo ============================================================
echo   make      : %MAKE_EXE%
echo   arch      : %OFFLOAD_ARCH%
echo   suite     : %SUITE%
echo   group size: %GS%
if "%QUICK%"=="1" echo   mode      : quick ^(decode shape only^)

if /i "%SUITE%"=="gqa" goto gqa_suite

REM ============================================================
REM  MatMulNBits: one suite per weight format, same shapes
REM ============================================================
call :matmul u2 gemm_fp16u2
call :matmul u3 gemm_fp16u3
call :matmul i8 gemm_fp16i8

REM No Makefile, so not runnable from a clean checkout (see docs).
call :skip "matmul u4" "gemm_fp16u4 has no Makefile"

if /i "%SUITE%"=="matmul" goto summary

:gqa_suite
REM ============================================================
REM  GQA
REM ============================================================
call :gqa_autotune
call :skip "gqa decode"    "gqa\decode has no Makefile"
call :skip "gqa prefill"   "gqa\prefill has no Makefile"

goto summary

REM ============================================================
REM  :matmul <tag> <dir>
REM ============================================================
:matmul
set "TAG=%~1"
set "DIR=%EXAMPLE_DIR%\%~2"

echo.
echo ------------------------------------------------------------
echo  MatMulNBits %TAG%
echo ------------------------------------------------------------
if not exist "%DIR%\Makefile" (
  call :skip "matmul %TAG%" "%~2 has no Makefile"
  goto :eof
)
pushd "%DIR%"

REM Decode (M=1) takes the GEMV path; small prefill (M=128) takes WMMA.
call :make_test "matmul %TAG%" 1x2880x5120
if "%QUICK%"=="0" call :make_test "matmul %TAG%" 128x2880x5120
if "%QUICK%"=="0" call :make_test "matmul %TAG%" 1x4096x2880

popd
goto :eof

REM  :make_test <label> <MxKxN>
:make_test
echo   [make] test SIZE=%~2 GS=%GS%
"%MAKE_EXE%" test SIZE=%~2 GS=%GS% %MK_ARGS%
if errorlevel 1 (call :fail "%~1 %~2" "make test" & goto :eof)
call :ok "%~1 %~2"
goto :eof

REM ============================================================
:gqa_autotune
set "DIR=%EXAMPLE_DIR%\gqa\autotune"
echo.
echo ------------------------------------------------------------
echo  GQA autotune sweep
echo ------------------------------------------------------------
if not exist "%DIR%\Makefile" (
  call :skip "gqa autotune" "gqa\autotune has no Makefile"
  goto :eof
)
REM A shape sweep, not a pass/fail test: minutes, not seconds, and it pipes
REM through tee. Not what --quick is for.
if "%QUICK%"=="1" (
  call :skip "gqa autotune" "sweep skipped under --quick"
  goto :eof
)
pushd "%DIR%"
echo   [make] run
"%MAKE_EXE%" run %MK_ARGS%
if errorlevel 1 (call :fail "gqa autotune" "make run" & popd & goto :eof)
call :ok "gqa autotune"
popd
goto :eof

REM ============================================================
:ok
set /a PASSED+=1
set "SUMMARY=!SUMMARY!  PASS  %~1;"
goto :eof

:fail
set /a FAILED+=1
echo   FAILED: %~1 at step '%~2'
set "SUMMARY=!SUMMARY!  FAIL  %~1 [%~2];"
goto :eof

:skip
set /a SKIPPED+=1
set "SUMMARY=!SUMMARY!  SKIP  %~1 [%~2];"
goto :eof

REM ============================================================
:summary
echo.
echo ============================================================
echo  Summary  ^(arch %OFFLOAD_ARCH%^)
echo ============================================================
if defined SUMMARY for %%L in ("!SUMMARY:;=" "!") do if not "%%~L"=="" echo %%~L
if not defined SUMMARY echo   ^(nothing ran^)
echo ------------------------------------------------------------
echo   passed: %PASSED%    failed: %FAILED%    skipped: %SKIPPED%
echo ============================================================
if %FAILED% GTR 0 exit /b 1
exit /b 0

:usage
echo.
echo Usage: run_full_pipeline_kernel_tests.bat [options]
echo.
echo   --arch ^<gfxNNNN^>   offload arch           (default gfx1150)
echo   --hip-sdk ^<path^>   HIP SDK root           (default: per-Makefile)
echo   --python ^<exe^>     python interpreter     (default: per-Makefile)
echo   --make ^<path^>      make executable        (default: found on PATH)
echo   --suite ^<name^>     all ^| matmul ^| gqa     (default all)
echo   --gs ^<n^>           MatMulNBits group size (default 128)
echo   --quick             decode shape only
echo.
exit /b 2
