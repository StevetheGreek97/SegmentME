; Inno Setup script that wraps the PyInstaller onedir build (dist/SegmentME/
; or dist/SegmentME-cuda/) into a single Windows installer: installs to
; Program Files (or a per-user folder if the installer is run without admin
; rights), adds a Start Menu entry and optional Desktop shortcut, registers
; the .SEproj file association, and adds a normal Add/Remove Programs entry
; with an uninstaller.
;
; Built by `python build.py --installer` (which passes AppVersion, Flavor,
; Arch and SourceDir via /D defines -- see make_installer() in build.py) or
; directly, after a normal build.py run has produced dist/SegmentME/:
;
;   iscc /DAppVersion=2.0.0 /DFlavor=cpu /DArch=x64 /DSourceDir=..\dist\SegmentME installer\windows.iss
;
; Arch is this project's own tag (x64 or arm64), not Inno Setup's -- see
; ArchId below.
;
; Needs the Inno Setup 6 command-line compiler (ISCC.exe) --
; https://jrsoftware.org/isinfo.php.
;
; Like the Linux .deb (see BUILDING.md), the cpu and cuda flavours install
; to the same path and are not meant to be installed side by side.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Flavor
  #define Flavor "cpu"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\SegmentME"
#endif
#ifndef Arch
  #define Arch "x64"
#endif
#define ExeName (Flavor == "cuda" ? "SegmentME-cuda.exe" : "SegmentME.exe")
#define DisplayName (Flavor == "cuda" ? "SegmentME (CUDA)" : "SegmentME")
; Inno Setup's own architecture identifier, not this project's "x64"/"arm64"
; tag: x64compatible also covers an ARM64 Windows machine running the
; installer under x64 emulation, which is correct for an x64-flavour build,
; but a native ARM64 build should say so explicitly so it installs as a true
; 64-bit ARM64 app (64-bit registry/Program Files) rather than emulated.
#define ArchId (Arch == "arm64" ? "arm64" : "x64compatible")

[Setup]
AppId={{9C3B29FD-0EC5-42F3-AA4C-47073695A81F}
AppName={#DisplayName}
AppVersion={#AppVersion}
AppPublisher=Steve Inc
AppSupportURL=https://github.com/StevetheGreek97/SegmentME
DefaultDirName={autopf}\SegmentME
DefaultGroupName=SegmentME
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=SegmentME-Setup-{#AppVersion}-windows-{#Flavor}-{#Arch}
SetupIconFile=..\resources\icons\icon.ico
WizardImageFile=..\resources\icons\installer_banner.bmp
WizardSmallImageFile=..\resources\icons\installer_small.bmp
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed={#ArchId}
ArchitecturesInstallIn64BitMode={#ArchId}
UninstallDisplayIcon={app}\{#ExeName}
; No admin rights required: installs per-user under AppData if the user
; declines elevation, or to Program Files if they accept/run as admin.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Everything PyInstaller collected -- the exe, _internal/, and models/
; (SAM2 Tiny, bundled by build.py after the PyInstaller step).
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SegmentME"; Filename: "{app}\{#ExeName}"
Name: "{group}\Uninstall SegmentME"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SegmentME"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Registry]
; Register the .SEproj project file extension, matching install-desktop.sh's
; MIME registration on Linux, so double-clicking a project opens it.
Root: HKA; Subkey: "Software\Classes\.SEproj"; ValueType: string; ValueName: ""; ValueData: "SegmentME.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\SegmentME.Project"; ValueType: string; ValueName: ""; ValueData: "SegmentME Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SegmentME.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#ExeName},0"
Root: HKA; Subkey: "Software\Classes\SegmentME.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#ExeName}"" ""%1"""

[Run]
Filename: "{app}\{#ExeName}"; Description: "Launch SegmentME"; Flags: nowait postinstall skipifsilent
