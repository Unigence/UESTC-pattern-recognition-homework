<#
.SYNOPSIS
Replace unique equation markers in a DOCX with editable MathType OLE equations.
.DESCRIPTION
Requires Windows, desktop Word and desktop MathType (same bitness as PowerShell).
The JSON file is an array of objects with id and tex fields. The DOCX is updated
only after all conversions succeed. A PDF is exported for visual verification.
Uses a new Word instance, the vendor BlankEqn.doc bookmark and MathPage.wll API;
it does not use the clipboard or close pre-existing Word/MathType windows.
.EXAMPLE
.\tools\homework2_mathtype.ps1 -DocumentPath .\report\report.docx -EquationsPath .\report\equations.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$EquationsPath,
    [string]$PdfPath,
    [string]$MathTypeRoot = 'C:\Program Files (x86)\MathType',
    [string]$AuditPath
)
$ErrorActionPreference = 'Stop'
$documentFullPath = (Resolve-Path -LiteralPath $DocumentPath).Path
$equationsFullPath = (Resolve-Path -LiteralPath $EquationsPath).Path
if ([IO.Path]::GetExtension($documentFullPath) -ne '.docx') { throw 'DocumentPath must be a DOCX file.' }
if (-not $PdfPath) { $PdfPath = [IO.Path]::ChangeExtension($documentFullPath, '.pdf') }
$pdfFullPath = [IO.Path]::GetFullPath($PdfPath)
$architecture = if ([Environment]::Is64BitProcess) { '64' } else { '32' }
$mathPagePath = Join-Path $MathTypeRoot "MathPage\$architecture\MathPage.wll"
$blankEquationPath = Join-Path $MathTypeRoot 'Office Support\BlankEqn.doc'
foreach ($requiredPath in @($mathPagePath, $blankEquationPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) { throw "Required MathType file not found: $requiredPath" }
}
$equations = @(Get-Content -LiteralPath $equationsFullPath -Raw -Encoding UTF8 | ConvertFrom-Json)
if ($equations.Count -eq 0) { throw 'Equation manifest is empty.' }
$seenIds = @{}
foreach ($equation in $equations) {
    if (-not $equation.id -or -not $equation.tex) { throw 'Every equation requires nonempty id and tex fields.' }
    if ($seenIds.ContainsKey([string]$equation.id)) { throw "Duplicate equation id: $($equation.id)" }
    $seenIds[[string]$equation.id] = $true
}
if (-not ('Homework2MathTypeNative' -as [type])) {
    $csharpPath = $mathPagePath.Replace('"', '""')
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Homework2MathTypeNative {
    [DllImport(@"$csharpPath", CallingConvention=CallingConvention.StdCall)]
    public static extern int MTInitAPI(short options, short timeout);
    [DllImport(@"$csharpPath", CallingConvention=CallingConvention.StdCall)]
    public static extern int MTTermAPI();
    [DllImport(@"$csharpPath", CallingConvention=CallingConvention.StdCall)]
    public static extern int MTSetEqnFromLangStr(IntPtr equation, short language, IntPtr text, int length);
    [DllImport(@"$csharpPath", CallingConvention=CallingConvention.StdCall)]
    public static extern int MTGetLangStrFromEqn(IntPtr equation, short language, IntPtr text, ref int length);
    [DllImport(@"$csharpPath", CallingConvention=CallingConvention.StdCall)]
    public static extern int MTCloseOleObject(int saveOptions, IntPtr equation);
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
}
"@
}
$word = $null
$document = $null
$ownsWordInstance = $false
$apiInitialized = $false
$conversionAudit = @()
$existingWordProcessIds = @(Get-Process -Name WINWORD -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
try {
    $initializationStatus = [Homework2MathTypeNative]::MTInitAPI(0, 30)
    if ($initializationStatus -le 0) { throw "MathType initialization failed: $initializationStatus" }
    $apiInitialized = $true
    $word = New-Object -ComObject Word.Application
    [void]$word.Version
    $newWordProcessIds = @(Get-Process -Name WINWORD -ErrorAction SilentlyContinue | Where-Object { $_.Id -notin $existingWordProcessIds } | Select-Object -ExpandProperty Id)
    if ($newWordProcessIds.Count -ne 1) {
        throw 'Could not verify a separate Word process; no document was opened.'
    }
    $ownsWordInstance = $true
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3  # Disable document macros in this new instance.
    $document = $word.Documents.Open($documentFullPath, $false, $false, $false)
    if ($document.ReadOnly) { throw 'The DOCX is locked/read-only; close that generated report before conversion.' }
    $documentText = $document.Content.Text
    foreach ($equation in $equations) {
        $count = [regex]::Matches($documentText, [regex]::Escape([string]$equation.id)).Count
        if ($count -ne 1) { throw "Marker $($equation.id) occurs $count times; exactly one occurrence is required." }
    }
    $initialShapeCount = $document.InlineShapes.Count
    $index = 0
    foreach ($equation in $equations) {
        $index++
        $marker = [string]$equation.id
        $tex = [string]$equation.tex
        if ($tex -notmatch '[\\^]') { $tex = '\mathit{' + $tex + '}' }
        Write-Output "[$index/$($equations.Count)] Converting $marker"
        $markerRange = $document.Content.Duplicate
        $find = $markerRange.Find
        $find.ClearFormatting()
        $find.Text = $marker
        $find.MatchCase = $true
        $find.MatchWholeWord = $false
        $find.MatchWildcards = $false
        $find.Forward = $true
        $find.Wrap = 0
        if (-not $find.Execute()) { throw "Marker disappeared: $marker" }
        $start = $markerRange.Start
        $markerRange.Text = ''
        $markerRange.Collapse(1)
        $markerRange.InsertFile($blankEquationPath, 'MTBlankEqn')
        $shape = $null
        foreach ($candidateShape in $document.InlineShapes) {
            if ($candidateShape.Range.Start -eq $start) { $shape = $candidateShape; break }
        }
        if ($null -eq $shape) { throw "Expected one MathType inline shape at $marker." }
        if ($shape.OLEFormat.ProgID -ne 'Equation.DSMT4') { throw "Unexpected equation ProgID at $marker." }
        $dispatchPointer = [IntPtr]::Zero
        $textPointer = [IntPtr]::Zero
        $closeStatus = $null
        $roundTripTex = ''
        try {
            $shape.OLEFormat.DoVerb(0)
            Start-Sleep -Milliseconds 650
            $equationObject = $shape.OLEFormat.Object
            $dispatchPointer = [Runtime.InteropServices.Marshal]::GetIDispatchForObject($equationObject)
            $textPointer = [Runtime.InteropServices.Marshal]::StringToHGlobalUni($tex)
            $writeStatus = [Homework2MathTypeNative]::MTSetEqnFromLangStr($dispatchPointer, 1, $textPointer, $tex.Length)
            # MathType can return success while inserting "Tex translation failed".
            # Read the interpreted equation back to reject that silent failure.
            [int]$readLength = 0
            $readStatus = [Homework2MathTypeNative]::MTGetLangStrFromEqn($dispatchPointer, 1, [IntPtr]::Zero, [ref]$readLength)
            if ($readStatus -eq 0 -and $readLength -gt 0) {
                $readBuffer = [Runtime.InteropServices.Marshal]::AllocHGlobal($readLength + 2)
                try {
                    $readStatus = [Homework2MathTypeNative]::MTGetLangStrFromEqn($dispatchPointer, 1, $readBuffer, [ref]$readLength)
                    $roundTripTex = [Runtime.InteropServices.Marshal]::PtrToStringAnsi($readBuffer)
                }
                finally { [Runtime.InteropServices.Marshal]::FreeHGlobal($readBuffer) }
            }
            $closeStatus = [Homework2MathTypeNative]::MTCloseOleObject(1, $dispatchPointer)
            if ($writeStatus -ne 0 -or $closeStatus -ne 0) {
                throw "MathType conversion failed for ${marker}: write=$writeStatus, close=$closeStatus"
            }
            if ($readStatus -ne 0 -or -not $roundTripTex -or $roundTripTex -match 'translation failed') {
                throw "MathType could not interpret ${marker}: $roundTripTex (read status $readStatus)"
            }
        }
        finally {
            if ($textPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::FreeHGlobal($textPointer) }
            if ($dispatchPointer -ne [IntPtr]::Zero) {
                if ($null -eq $closeStatus) { [void][Homework2MathTypeNative]::MTCloseOleObject(0, $dispatchPointer) }
                [void][Runtime.InteropServices.Marshal]::Release($dispatchPointer)
            }
        }
        Start-Sleep -Milliseconds 150
        if ($shape.Width -le 1 -or $shape.Height -le 1) { throw "MathType produced an empty equation for $marker." }
        # Equation paragraphs must expand vertically to include fractions/subscripts.
        $paragraphFormat = $shape.Range.ParagraphFormat
        $paragraphFormat.LineSpacingRule = 3  # wdLineSpaceAtLeast
        $paragraphFormat.LineSpacing = 24
        $paragraphFormat.FirstLineIndent = 0
        $availableWidth = $document.PageSetup.PageWidth - $document.PageSetup.LeftMargin - $document.PageSetup.RightMargin
        if ($shape.Width -gt $availableWidth) {
            throw "Equation $marker is too wide ($($shape.Width) pt > $availableWidth pt); split its TeX across equations."
        }
        $conversionAudit += [pscustomobject]@{
            id = $marker; prog_id = [string]$shape.OLEFormat.ProgID
            width_pt = [double]$shape.Width; height_pt = [double]$shape.Height
            write_status = $writeStatus; close_status = $closeStatus
            round_trip_tex = $roundTripTex
        }
        Write-Output "  Equation.DSMT4, $($shape.Width) x $($shape.Height) pt"
    }
    if ($document.InlineShapes.Count -ne $initialShapeCount + $equations.Count) { throw 'Final inline-shape count is inconsistent.' }
    foreach ($equation in $equations) {
        if ($document.Content.Text.Contains([string]$equation.id)) { throw "Unreplaced marker: $($equation.id)" }
    }
    [void]$document.Fields.Update()
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $document.Save()
    $document.ExportAsFixedFormat($pdfFullPath, 17)
    if ($AuditPath) {
        $conversionAudit | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $AuditPath -Encoding UTF8
    }
    Write-Output "Saved $documentFullPath"
    Write-Output "Exported $pdfFullPath ($pageCount pages; $($equations.Count) editable MathType equations)"
}
finally {
    if ($null -ne $document) {
        try { $document.Close(0) } catch { Write-Warning $_.Exception.Message }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        if ($ownsWordInstance) { try { $word.Quit(0) } catch { Write-Warning $_.Exception.Message } }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    if ($apiInitialized) { [void][Homework2MathTypeNative]::MTTermAPI() }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
