Embedded Machine Learning for Real-Time Fall Detection: An Edge-Computing TinyML Paradigm
-----------------------------------------------------------------------------------------

### 1. Project Overview & Architectural Evolution

This Capstone project delivers a completely offline, real-time fall detection
wearable device executing TinyML inference directly on a resource-constrained
edge microcontroller.

-   **Architectural Pivot:** The project initially utilized a cross-platform
    Flutter application architecture. However, to eliminate deployment overhead
    and strict CI/CD dependencies (e.g., `codemagic.yaml`), the system was
    successfully refactored into a pure HTML5 Web BLE architecture. This ensures
    cross-platform telemetry via standard Web APIs (e.g., Bluefy browser on iOS)
    with zero app-installation friction.

### 2. Hardware Specifications

The physical hardware relies on ultra-low-power edge components:

-   **MCU:** Arduino Nano 33 BLE Sense Rev2 (nRF52840 ARM Cortex-M4F), strictly
    constrained to 256 KB SRAM and 1 MB Flash.

-   **Sensors:** Onboard Bosch BMI270 6-axis IMU.

-   **Actuators & I/O:** Active Piezo-Buzzer connected to D3, Red LED on D5,
    Blue LED on D6, and a manual hardware reset button on D10 configured with an
    internal pull-down resistor.

-   **Power:** External 3.7V Li-Po battery managed by a TP4056/PM11 power
    module.

### 3. Data Engineering & MLOps Pipeline

To address real-world Covariate Shifts and prevent overfitting, we constructed
an iterative human-in-the-loop MLOps pipeline:

-   **Base Dataset Initialization:** Initial 3-axis IMU data was extracted from
    the public SisFall dataset. The continuous time-series data was segmented
    using a 2-second sliding window with a 50% overlap to isolate distinct
    kinematic features.

-   **Iterative Human-in-the-Loop:** The base INT8 model was flashed to the MCU.
    The researcher performed simulated falls and activities of daily living
    (ADL) while tethered via a USB cable to a PC. False positives were manually
    captured via the data forwarder, re-labeled, and fed back into the Edge
    Impulse platform for continuous hyperparameter optimization.

-   **Model Compression:** The final 1D-CNN model was highly compressed using
    INT8 post-training quantization. This allows the neural network to execute
    within the strict 256 KB SRAM limit.

### 4. Firmware Architecture & DSP (C++ Device-Side)

The Arduino C++ firmware (v15) operates an industrial-grade non-blocking Finite
State Machine (FSM).

-   **Always-On Continuous Inference:** Due to the extreme efficiency of the
    INT8 quantized model, the system bypasses traditional physical wake-up
    thresholds. It executes continuous active inference to proactively evaluate
    human kinematics.

-   **Advanced AVM Temporal Low-G Guard:** To eliminate false positives from
    daily activities (e.g., wrist waving, walking), a temporal DSP filter is
    applied. The system requires the Acceleration Vector Magnitude (AVM) to
    remain strictly below 4.5 m/s² for at least 4 consecutive samples (160ms) to
    unlock the pre-fall state. A severe hard impact threshold is set at 24.0
    m/s².

-   **Telemetry Selectivity:** The firmware supports dynamic monitoring modes
    via a macro selector: Mode 0 (Standard Diagnostic Monitor), Mode 1 (Arduino
    Serial Plotter format at 25Hz), and Mode 2 (Serial Studio Telemetry CSV
    wrapping).

-   **BLE Telemetry Integration:** Utilizes `<ArduinoBLE.h>` exposing the custom
    `FFF0` service. Characteristics include `FFF1` (Fall State), `FFF2`
    (Real-time AVM), `FFF3/4` (Active Class & Confidence), and `FFF5` (Raw
    6-axis IMU data).

### 5. Web BLE Telemetry & CSV Dataset Generation

The frontend telemetry dashboard (`index.html`) is built purely with HTML5, CSS,
and JavaScript, establishing a GATT client connection to the MCU.

-   **Real-time Rendering Fix:** Implemented robust JS data parsing to resolve
    the `[object Object]` rendering bug during active countdown alerts.

-   **Cross-Platform CSV Export:** The interface acts as a standalone data
    logger. It captures the 25Hz telemetry stream and exports it using a
    fallback mechanism: utilizing `navigator.share` (Web Share API) to invoke
    native iOS/Android share sheets (e.g., AirDrop, Save to Files), or falling
    back to standard Blob downloads for PC environments.
