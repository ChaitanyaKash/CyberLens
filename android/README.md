# CyberLens Android

Phase 0 scaffold: Kotlin, Compose, Hilt, encrypted Room, appearance preferences, an adaptive navigation shell and a debug component catalogue. Call detection, recording access, analysis and message protection are not enabled in this scaffold.

## Local build

Use JDK 17 or newer, Android SDK Platform 36 and Python with the dependency in `tools/requirements.txt`. Set `sdk.dir` in an untracked `local.properties`. The build script uses Android Studio's bundled JDK when `JAVA_HOME` is unset.

Project JVM options include `-XX:ActiveProcessorCount=2` so internal compiler and garbage-collection pools match the two-CPU build budget. This complements Gradle's two-worker limit. The Gradle JVM also uses `-XX:TieredStopAtLevel=1` to limit cold JIT overhead; this does not change the Android app runtime or system settings.

The build script defaults to one Gradle worker to leave memory for other applications; `-Workers 2` is the supported upper limit. Run lint separately from tests on a memory-constrained laptop.

Before building, run LibreHardwareMonitor as administrator. Enable its web server on **127.0.0.1:8085 only**. The guard requires an Intel CPU Package temperature sensor and NVIDIA GPU temperature sensors from `/data.json`; it fails closed if either is unavailable. Both AC and battery maximum processor states must be 99%, and processor boost (`PERFBOOSTMODE`) must be Disabled (0) on both.

```powershell
python tools/thermal_guard.py --probe
.\tools\build.ps1
.\tools\build.ps1 -Tasks ':app:testDebugUnitTest'
.\tools\build.ps1 -Tasks ':app:testDebugUnitTest' -RecordScreenshots
.\tools\build.ps1 -Tasks ':app:testDebugUnitTest' -RecordScreenshots -ScreenshotSmoke
```

The guard restricts its process tree to two logical processors and idle priority. Short work bursts use a bounded 10–40% duty cycle according to temperature. It holds at CPU 65 °C or GPU 75 °C until cooling below CPU 60 °C and GPU 70 °C, pauses at 80 °C and aborts its own job at 85 °C. On a transient unavailable reading it keeps its job paused; readings must recover within 10 seconds and cool before work resumes. Persistent sensor loss aborts the job. It resets the monitored sensors' min/max at job start and checks their recorded peaks as well as current temperatures, so a spike between polls is not discarded. Detected external compute jobs cause it to wait; it never suspends them. Sampling still depends on the monitor's update rate, so retain hardware thermal protection and supervise first-time workloads.

## Catalogue and review

In debug builds, open **Components** from the shell. Review typography, controls, cards, glass and questions in light/dark, at 100%/200% font size and with reduced motion/transparency enabled. Settings are in Profile. No real content is analysed and no sensitive Android permission is requested by this phase.

Room uses SQLCipher with a random database passphrase wrapped by an Android Keystore AES-GCM key. The wrapped passphrase lives in `noBackupFilesDir`. Backup is disabled, and explicit legacy backup and modern cloud/device-transfer rules exclude all supported app-storage domains. These rules keep a device transfer from restoring an encrypted database without its device-bound key.

The current device scope is Samsung phones and tablets, prioritizing Galaxy S23 FE and the Galaxy Ultra series. The screenshot matrix covers small and large phone windows, portrait/landscape, tablets and narrow multi-window, both themes and 100%/200% text, plus 130% phone text and Hindi at 200%. The connected S23 FE reports 1080×2340 pixels at 450 dpi, corresponding to a 384×832 dp full window. Wider phone cases cover different display zoom and resolution configurations without assuming a particular Ultra model. `-ScreenshotSmoke` selects one English light-theme phone screenshot while retaining the interaction tests. A defined case is not a passed check: inspect test results and actual PNG output. Review changed baselines before committing them. Device installation and catalogue review are mandatory before Phase 0 is complete.

Navigation adapts to the current window width, keeping phone, tablet and multi-window layouts responsive. Existing hinge-aware panes and their regression tests are retained; additional foldable design and device validation are deferred while Samsung phone/tablet coverage takes priority.

Compact-height windows use the bottom bar. When scaled labels cannot fit beneath five icons, the bar shows the full selected or keyboard-focused name on a shared line; each tab retains its own accessible name. Compact rails grow with their labels rather than breaking words. Question sheets inherit the caller's font scaling, including catalogue previews. Reduced transparency uses opaque surfaces with defined card and sheet borders.

For a change confined to one window, record its subset with `-PscreenshotWindow=tablet_portrait` on `:app:testDebugUnitTest`, then run `:app:verifyRoborazziDebug` without that property to verify the complete matrix.

Keep local notes, SDK paths, keystores, model weights and private recordings outside version control. Public releases require later phase gates and separate approval.

## Launcher artwork

The full-colour transparent master is `artwork/launcher_master.png`; `artwork/launcher_512.png` is the square export. Run `python tools/package_icon.py` to recreate the five density variants. The adaptive launcher uses the full-colour artwork over a navy background with safe-zone padding. A separate vector mask supports Android's optional themed-icon mode.
