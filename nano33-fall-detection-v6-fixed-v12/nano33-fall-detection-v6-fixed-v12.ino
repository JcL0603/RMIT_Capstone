// ==================================================================== //
// EEET2450 Capstone Project - TinyML Active 5-Class Fall Monitor (v6.0 - FIXED v12)
// Target Hardware: Arduino Nano 33 BLE Sense Rev2 (ABX00070 SKU)
// Student: Jacky LIN (S4143087 / 247024328)
// Supervisor: Dr. Scottie Man
//
// Description: Polished, professional-grade wearable active fall safety firmware.
//              Optimized and Upgraded in v6.0-Fixed-v12 (BLE Telemetry Integration & Internal GPIO Fix):
//              1. [PM11 POWER & BUTTON SIMPLIFICATION]:
//                 - Updated power architecture docs to PM11 USB charging & distribution module.
//                 - D10 Manual Reset Button configured with nRF52840 internal pull-down resistor (pinMode INPUT_PULLDOWN).
//                 - Eliminates the need for any physical external pull-down hardware resistors!
//              2. [INTEGRATED BLE WIRELESS ALERT SYSTEM]:
//                 - Employs <ArduinoBLE.h> to advertise "Jacky_FallMonitor" service.
//                 - Exposes FFF0 Service, FFF1 (fall_state) and FFF2 (avm) Characteristics.
//                 - Broadcasts SOS state changes (0: Standby, 1: Prefall Warning, 2: Fall Alarm) to caregivers' phones.
//                 - Fully non-blocking BLE.poll() loop integration that strictly avoids any timing conflicts with 25Hz IMU sampling.
//              3. [MONITOR_MODE SELECTOR]:
//                 - Mode 0: Standard Diagnostic Monitor (Human-readable text, 2.5Hz refresh)
//                 - Mode 1: Arduino Serial Plotter (AVM & Flags, 25Hz Tab-delimited \t format)
//                 - Mode 2: Serial Studio Telemetry (25Hz Comma-Separated Values wrapped in /* ... */)
//              4. [PREFALL WARNING RE-TRIGGER LOCK]:
//                 - Prefall hold-on duration set to 2.0 seconds (2000ms).
//                 - If a new prefall classification is detected WHILE already in STATE_FALL_RISK,
//                   the 2.0-second timer is refreshed (renewed) dynamically.
//              5. Compile-time Dynamic Memory Alignment Guard (Fixes the -1004 & -5 matrix crash).
//              6. Re-triggerable Monostable 1-second LED controller (Uses ONLY Blue LED, green disabled).
//              7. Advanced Temporal Low-G Filter to eliminate hand-shaking and walking false alarms (>=4 consecutive samples <4.5 m/s²).
//              8. Active Buzzer Ambulance Simulation (100% tone-free, temporal-modulation on-off rhythm).
// ==================================================================== //

#include <Arduino_BMI270_BMM150.h> // Onboard IMU for Rev2
#include <ArduinoBLE.h>            // Onboard Bluetooth Low Energy Library
#include <Jacky_TinyML_Fall_Detection_Waist_v4_inferencing.h> // Target V4 exported C++ Library

// ==================== 0. Debug & Plotter Mode Selector ====================
// 💡 [MONITOR MODE SELECTOR]
// 0: Standard Diagnostic Monitor - human-readable text blocks at comfortable 2.5Hz (RMIT Standard)
// 1: Arduino Serial Plotter - AVM and Flags printed at 25Hz using Tab-delimiters (\t)
// 2: Serial Studio Telemetry - /*[payload]*/ high-speed comma-separated frame at 25Hz (For Serial Studio Dashboard!)
#define MONITOR_MODE   2

// ==================== 1. Hardware Pin Configurations ====================
const int LED_R = 5;      // Red LED (D5) - Prefall / Fall danger warnings
const int LED_B = 6;      // Blue LED (D6) - Standby / Walk status indication
const int LED_G = 9;      // Green LED (D9) - Disused/Abandoned to prevent voltage sag
const int BUZZER = 3;     // Active Buzzer (D3)
const int BUTTON = 10;    // Manual reset button (D10) configured with internal pull-down resistor

const int LED_ON = HIGH;
const int LED_OFF = LOW;
const int BUZZ_ON = HIGH;
const int BUZZ_OFF = LOW;
const int BUTTON_PRESSED = HIGH;

// ==================== 2. Sampling & Buffer Configurations ====================
const float SAMPLE_RATE_HZ = 25.0;                      // 25Hz Target sampling rate
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ; // 40ms interval
const float G_TO_MS2 = 9.80665;                         // Conversion constant

// 💡 Compile-time Dynamic Memory Alignment Guard (Fixes the -1004 & -5 matrix crash!)
const int WINDOW_SIZE_SAMPLES = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / 6;

// Sliding Circular Ring Buffer
struct IMUSample {
    float ax; // m/s²
    float ay; // m/s²
    float az; // m/s²
    float gx; // dps
    float gy; // dps
    float gz; // dps
};
IMUSample ring_buffer[WINDOW_SIZE_SAMPLES];
int buffer_head = 0;
int buffer_count = 0;

// Continuous Inference Scheduling
int sample_counter = 0;
const int INFERENCE_STRIDE_SAMPLES = 2; // Run inference every 2 samples (80ms / 12.5Hz) for zero-latency

// Global flags for 5 states (normally 0, becomes 1 when active)
int flag_static = 0;
int flag_walk = 0;
int flag_stairs = 0;
int flag_prefall = 0;
int flag_fall = 0;

// Global AI probability states (Updated at 12.5Hz, streamed at 25Hz for Serial Studio)
float prob_static = 0.0;
float prob_walk = 0.0;
float prob_stairs = 0.0;
float prob_prefall = 0.0;
float prob_fall = 0.0;
float winner_confidence = 0.0;
int winner_class_idx = 0; // 0:Static, 1:Walk, 2:Stairs, 3:Prefall, 4:Fall

// Non-blocking monostable 1-second LED timers
unsigned long blueLedOnUntil = 0;

// ==================== 3. Advanced AVM Temporal Low-G Guard ====================
// AVM = sqrt(ax^2 + ay^2 + az^2)
// Daily wrist waving or walking ADL only has transient low-G dips (<40ms / 1 sample).
// A real falling body experiences sustained free-fall (<0.45g / 4.5 m/s²) lasting > 150ms.
// This filter requires AVM to be continuously < 4.5 m/s² for at least 4 samples (160ms) to unlock Prefall.
const float PREFALL_PHYSICAL_THRES = 4.5;    // Severe weightlessness threshold (m/s²)
const float FALL_PHYSICAL_THRES = 24.0;      // Hard impact threshold (m/s²)
const int REQUIRED_CONSECUTIVE_LOW_G = 4;    // Must be in freefall for 4 consecutive samples (160ms)

// ==================== 4. Finite State Machine (FSM) ====================
enum SystemState {
    STATE_STANDBY,        // Standard monitoring (Dynamic monostable lights, quiet)
    STATE_FALL_RISK,      // Pre-Impact Alert: Impending fall forecasted (Solid Red LED, direct HIGH buzzer scream)
    STATE_SEVERE_FALL     // Post-Impact Alarm: Hard crash confirmed (Red/Blue strobe, active buzzer ambulance siren)
};
SystemState currentState = STATE_STANDBY;

// Non-blocking Timing Control
unsigned long lastSampleTime = 0;
unsigned long lastBlinkTime = 0;
unsigned long lastSirenTime = 0;
unsigned long prefallStartTime = 0; // Records when prefall alert starts
bool blinkState = false;
bool sirenToggle = false;

// ==================== 5. BLE Wireless Configurations ====================
// Custom BLE Service & Characteristics UUIDs (FFF0 Profile)
BLEService fallService("FFF0");
BLEUnsignedCharCharacteristic fallStateChar("FFF1", BLERead | BLENotify); // 0:Standby, 1:Prefall, 2:Fall
BLEFloatCharacteristic bleAvmChar("FFF2", BLERead | BLENotify);          // Wireless real-time AVM transmission

// ==================== 6. Function Prototypes ====================
void readSensorsAndBuffer();
void runActiveTinyMLInference();
void updateFSM();
void handleStandby();
void handleFallRisk();
void handleSevereFall();
void resetToStandby();
void updateBLEState(uint8_t stateVal);

// ==================== 7. Setup ====================
void setup() {
    Serial.begin(115200);
    
    pinMode(LED_R, OUTPUT);
    pinMode(LED_B, OUTPUT);
    pinMode(LED_G, OUTPUT);
    pinMode(BUZZER, OUTPUT);
    
    // 💡 Rigorous Fact-Check: Configured internal pull-down resistor on nRF52840.
    // Removes the need for physical external resistors on D10!
    pinMode(BUTTON, INPUT_PULLDOWN); 
    
    // Set initial quiet state
    digitalWrite(LED_R, LED_OFF);
    digitalWrite(LED_B, LED_OFF);
    digitalWrite(LED_G, LED_OFF);
    digitalWrite(BUZZER, BUZZ_OFF);
    
    unsigned long waitStart = millis();
    while (!Serial && (millis() - waitStart < 4000)) {
        // Wait up to 4 seconds for a virtual serial port host connection
    }
    delay(500);

    #if MONITOR_MODE == 0
    if (Serial) {
        Serial.println("=================================================");
        Serial.println("  EEET2450 Capstone Fall Monitor - TinyML v6.0-Fixed-v12 ");
        Serial.println("  Inference Strategy: 12.5Hz Continuous Active CNN  ");
        Serial.println("  Filters: Advanced Temporal Low-G Safeguards       ");
        Serial.println("  Hardware: Arduino Nano 33 BLE Sense Rev2 + PM11   ");
        Serial.println("  Supervisor: Dr. Scottie Man                       ");
        Serial.println("=================================================");
    }
    #endif

    // Initialize Onboard IMU
    if (!IMU.begin()) {
        #if MONITOR_MODE == 0
        if (Serial) {
            Serial.println("❌ Error: Onboard BMI270 IMU failed to initialize!");
        }
        #endif
        while (1) {
            digitalWrite(LED_R, HIGH); delay(100);
            digitalWrite(LED_R, LOW); delay(100);
        }
    }
    
    // Initialize Bluetooth Low Energy (BLE)
    if (!BLE.begin()) {
        #if MONITOR_MODE == 0
        if (Serial) {
            Serial.println("❌ Error: Onboard nRF52840 BLE failed to initialize!");
        }
        #endif
        // Visual warning for BLE failure (Flash Red/Blue simultaneously)
        for (int i=0; i<10; i++) {
            digitalWrite(LED_R, HIGH); digitalWrite(LED_B, HIGH); delay(50);
            digitalWrite(LED_R, LOW);  digitalWrite(LED_B, LOW);  delay(50);
        }
    } else {
        // Config advertisement payload
        BLE.setLocalName("Jacky_FallMonitor");
        BLE.setAdvertisedService(fallService);
        
        // Bind characteristics
        fallService.addCharacteristic(fallStateChar);
        fallService.addCharacteristic(bleAvmChar);
        BLE.addService(fallService);
        
        // Initial values
        fallStateChar.writeValue(0); // Standby
        bleAvmChar.writeValue(0.0);
        
        // Begin advertising
        BLE.advertise();
        
        #if MONITOR_MODE == 0
        if (Serial) {
            Serial.println("📡 BLE Onboard Server Active: Advertising 'Jacky_FallMonitor'...");
        }
        #endif
    }
    
    #if MONITOR_MODE == 0
    if (Serial) {
        Serial.println("✅ Onboard BMI270 IMU active (25Hz sampling rate)");
        Serial.print("🛡️ Compile-Time Buffer Size: "); Serial.print(WINDOW_SIZE_SAMPLES); Serial.println(" samples (100% stack-aligned)");
        Serial.println("🛡️ Active Pre-Impact Guardian: Online.");
        Serial.println("=================================================");
    }
    #endif
}

// ==================== 8. Main Loop ====================
void loop() {
    // 💡 Non-blocking BLE events polling.
    // Handles peer handshake, subscription request, and telemetry synchronization
    // in microseconds without blocking the strict 25Hz (40ms) IMU sampling loop!
    BLE.poll();

    unsigned long currentTime = millis();
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = currentTime;
        readSensorsAndBuffer();
    }
    updateFSM();
}

// ==================== 9. Sensor Sampling & Continuous Stride Control ====================
void readSensorsAndBuffer() {
    float raw_ax, raw_ay, raw_az;
    float raw_gx, raw_gy, raw_gz;
    
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(raw_ax, raw_ay, raw_az);
        IMU.readGyroscope(raw_gx, raw_gy, raw_gz);
        
        // Convert to m/s² for acc, dps for gyro
        float ax = raw_ax * G_TO_MS2;
        float ay = raw_ay * G_TO_MS2;
        float az = raw_az * G_TO_MS2;
        float gx = raw_gx;
        float gy = raw_gy;
        float gz = raw_gz;
        
        // Write to circular ring buffer
        int write_idx = (buffer_head + buffer_count) % WINDOW_SIZE_SAMPLES;
        ring_buffer[write_idx] = { ax, ay, az, gx, gy, gz };
        
        if (buffer_count < WINDOW_SIZE_SAMPLES) {
            buffer_count++;
        } else {
            buffer_head = (buffer_head + 1) % WINDOW_SIZE_SAMPLES;
        }

        // Calculate current instantaneous AVM
        float current_avm = sqrt(ax * ax + ay * ay + az * az);

        // 📡 Transmit AVM periodically to subscribed BLE clients (Decimated to 5Hz to save RF power)
        static int ble_telemetry_decimator = 0;
        ble_telemetry_decimator++;
        if (ble_telemetry_decimator >= 5) {
            ble_telemetry_decimator = 0;
            bleAvmChar.writeValue(current_avm);
        }

        // 🛡️ CDC Guard: Only output if the port has been actively opened by PC.
        if (Serial) {
            #if MONITOR_MODE == 1
            // 📊 25Hz Real-Time Data Streaming for BetterSerialPlotter (Tab-delimited)
            Serial.print("AVM:");     Serial.print(current_avm, 2); Serial.print("\t");
            Serial.print("Static:");  Serial.print(flag_static);    Serial.print("\t");
            Serial.print("Walk:");    Serial.print(flag_walk);      Serial.print("\t");
            Serial.print("Stairs:");  Serial.print(flag_stairs);    Serial.print("\t");
            Serial.print("Prefall:"); Serial.print(flag_prefall);   Serial.print("\t");
            Serial.print("Fall:");    Serial.print(flag_fall);
            Serial.println(); 
            #elif MONITOR_MODE == 2
            // 📡 25Hz Real-Time Packet Streaming for Serial Studio (/* ... */ Comma Separated)
            // Sequence: ax, ay, az, gx, gy, gz, avm, f_static, f_walk, f_stairs, f_prefall, f_fall, p_static, p_walk, p_stairs, p_prefall, p_fall, winner_idx, winner_conf
            Serial.print("/*");
            Serial.print(ax, 3); Serial.print(",");
            Serial.print(ay, 3); Serial.print(",");
            Serial.print(az, 3); Serial.print(",");
            Serial.print(gx, 2); Serial.print(",");
            Serial.print(gy, 2); Serial.print(",");
            Serial.print(gz, 2); Serial.print(",");
            Serial.print(current_avm, 3); Serial.print(",");
            Serial.print(flag_static); Serial.print(",");
            Serial.print(flag_walk); Serial.print(",");
            Serial.print(flag_stairs); Serial.print(",");
            Serial.print(flag_prefall); Serial.print(",");
            Serial.print(flag_fall); Serial.print(",");
            Serial.print(prob_static * 100.0, 1); Serial.print(",");
            Serial.print(prob_walk * 100.0, 1); Serial.print(",");
            Serial.print(prob_stairs * 100.0, 1); Serial.print(",");
            Serial.print(prob_prefall * 100.0, 1); Serial.print(",");
            Serial.print(prob_fall * 100.0, 1); Serial.print(",");
            Serial.print(winner_class_idx); Serial.print(",");
            Serial.print(winner_confidence * 100.0, 1);
            Serial.println("*/");
            #endif
        }
        
        // 💡 Continuous Inference Scheduler (Every 2 samples = 80ms / 12.5Hz)
        if (buffer_count == WINDOW_SIZE_SAMPLES) {
            sample_counter++;
            if (sample_counter >= INFERENCE_STRIDE_SAMPLES) {
                sample_counter = 0; // Reset stride counter
                
                // Trigger 12.5Hz continuous active inference
                runActiveTinyMLInference();
            }
        }
    }
}

// ==================== 10. 12.5Hz Continuous Active Inference Engine ====================
void runActiveTinyMLInference() {
    // Flatten ring buffer to 1D feature array (BSS static memory to prevent Stack Overflow)
    static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE]; 
    for (int i = 0; i < WINDOW_SIZE_SAMPLES; i++) {
        int ring_idx = (buffer_head + i) % WINDOW_SIZE_SAMPLES;
        features[i * 6 + 0] = ring_buffer[ring_idx].ax;
        features[i * 6 + 1] = ring_buffer[ring_idx].ay;
        features[i * 6 + 2] = ring_buffer[ring_idx].az;
        features[i * 6 + 3] = ring_buffer[ring_idx].gx;
        features[i * 6 + 4] = ring_buffer[ring_idx].gy;
        features[i * 6 + 5] = ring_buffer[ring_idx].gz;
    }
    
    // Wrap buffer into Edge Impulse signal structure
    signal_t signal;
    numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    
    // Invoke our quantized 1D-CNN model
    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
    if (res != EI_IMPULSE_OK) {
        #if MONITOR_MODE == 0
        if (Serial) {
            Serial.print("❌ Edge Impulse run_classifier failed: ");
            Serial.println(res);
        }
        #endif
        return;
    }
    
    // Extract class probabilities dynamically via sub-string matching
    // Modify global probability states
    prob_static = 0.0;
    prob_walk = 0.0;
    prob_stairs = 0.0;
    prob_prefall = 0.0;
    prob_fall = 0.0;
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        String lbl = String(result.classification[ix].label);
        lbl.toLowerCase();
        float val = result.classification[ix].value;
        
        if (lbl.indexOf("static") >= 0 || lbl.indexOf("stat") >= 0) {
            prob_static = val;
        } else if (lbl.indexOf("walk") >= 0) {
            prob_walk = val;
        } else if (lbl.indexOf("stair") >= 0) {
            prob_stairs = val;
        } else if (lbl.indexOf("prefall") >= 0 || lbl.indexOf("pre_fall") >= 0 || lbl.indexOf("alert") >= 0) {
            prob_prefall = val;
        } else if (lbl.indexOf("fall") >= 0) {
            prob_fall = val;
        } else if (lbl.indexOf("adl") >= 0) {
            prob_static = val; // Backward compatibility fallback
        }
    }
    
    // Save raw neural network values before applying physical filtering
    float raw_nn_prefall = prob_prefall;
    float raw_nn_fall = prob_fall;
    
    // ==================== 💡 AVM Temporal Low-G Filter ====================
    float max_avm_in_window = 0.0;
    float min_avm_in_window = 999.0;
    int max_consecutive_low_g = 0;
    int current_consecutive_low_g = 0;
    
    for (int i = 0; i < WINDOW_SIZE_SAMPLES; i++) {
        float sample_avm = sqrt(ring_buffer[i].ax * ring_buffer[i].ax + 
                                ring_buffer[i].ay * ring_buffer[i].ay + 
                                ring_buffer[i].az * ring_buffer[i].az);
        if (sample_avm > max_avm_in_window) max_avm_in_window = sample_avm;
        if (sample_avm < min_avm_in_window) min_avm_in_window = sample_avm;
        
        // Temporal consecutive low-G count
        if (sample_avm < PREFALL_PHYSICAL_THRES) {
            current_consecutive_low_g++;
            if (current_consecutive_low_g > max_consecutive_low_g) {
                max_consecutive_low_g = current_consecutive_low_g;
            }
        } else {
            current_consecutive_low_g = 0; // Break, reset counter
        }
    }
    
    // Physical Gate Decisions:
    bool is_prefall_physically_allowed = (max_consecutive_low_g >= REQUIRED_CONSECUTIVE_LOW_G);
    bool is_fall_physically_allowed = (max_avm_in_window > FALL_PHYSICAL_THRES);
    
    if (!is_prefall_physically_allowed) {
        prob_prefall = 0.0; // Physical veto
    }
    if (!is_fall_physically_allowed) {
        prob_fall = 0.0;   // Physical veto
    }
    
    // Update State Flags: Find the class with the highest probability
    flag_static = 0;
    flag_walk = 0;
    flag_stairs = 0;
    flag_prefall = 0;
    flag_fall = 0;

    float max_val = 0.0;
    String max_label = "";
    float probs[5] = {prob_static, prob_walk, prob_stairs, prob_prefall, prob_fall};
    String labels[5] = {"class_static", "class_walk", "class_stairs", "class_prefall_alert", "class_fall"};
    
    for (int i = 0; i < 5; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
            max_label = labels[i];
        }
    }
    
    // Update global AI Winner index and confidence level
    winner_confidence = max_val;
    if (max_label == "class_static") winner_class_idx = 0;
    else if (max_label == "class_walk") winner_class_idx = 1;
    else if (max_label == "class_stairs") winner_class_idx = 2;
    else if (max_label == "class_prefall_alert") winner_class_idx = 3;
    else if (max_label == "class_fall") winner_class_idx = 4;

    // Set active flag to 1 if confidence threshold is met
    if (max_val > 0.50) {
        if (max_label == "class_static") {
            flag_static = 1;
            blueLedOnUntil = 0; // Clear walk monostable instantly
        }
        else if (max_label == "class_walk") {
            flag_walk = 1;
            blueLedOnUntil = millis() + 1000; // Re-triggerable Blue ON for 1s
        }
        else if (max_label == "class_stairs") {
            flag_stairs = 1;
            blueLedOnUntil = millis() + 1000; // Stairs also triggers Blue ON for 1s
        }
        else if (max_label == "class_prefall_alert") {
            flag_prefall = 1;
        }
        else if (max_label == "class_fall") {
            flag_fall = 1;
        }
    }

    // 🖥️ 12.5Hz Diagnostic Monitor (Visual Comfort Monitor: 2.5Hz)
    #if MONITOR_MODE == 0
    static int diag_decimation = 0;
    diag_decimation++;
    if (diag_decimation >= 5) { // 5 * 80ms = 400ms refresh rate!
        diag_decimation = 0;
        if (Serial) {
            Serial.println();
            Serial.println("################################################################");
            Serial.println("🧠 [AI Evaluation Scores] (Visual Comfort Monitor: 2.5Hz)");
            Serial.print("   • class_static:  "); Serial.print(prob_static * 100.0, 1); Serial.println("%");
            Serial.print("   • class_walk:    "); Serial.print(prob_walk * 100.0, 1); Serial.println("%");
            Serial.print("   • class_stairs:  "); Serial.print(prob_stairs * 100.0, 1); Serial.println("%");
            Serial.print("   • class_prefall: "); Serial.print(prob_prefall * 100.0, 1); Serial.println("%");
            Serial.print("   • class_fall:    "); Serial.print(prob_fall * 100.0, 1); Serial.println("%");
            Serial.println("----------------------------------------------------------------");
            Serial.print("📊 [Filter Monitor] Window_minAVM: "); Serial.print(min_avm_in_window, 2);
            Serial.print(" m/s² | Consecutive_Low_G: "); Serial.print(max_consecutive_low_g);
            Serial.print(" (Req: "); Serial.print(REQUIRED_CONSECUTIVE_LOW_G); Serial.println(")");
            Serial.print("   • PrefallAllowed: "); Serial.print(is_prefall_physically_allowed ? "YES" : "NO");
            Serial.print(" | FallAllowed: "); Serial.println(is_fall_physically_allowed ? "YES" : "NO");
            Serial.println("----------------------------------------------------------------");
            Serial.print("🎯 [AI WINNER CLASS] >>> "); Serial.print(max_label); Serial.print(" <<< with CONFIDENCE "); Serial.print(max_val * 100.0, 1); Serial.println("%");
            Serial.println("################################################################");
            Serial.println();
        }
    }
    #endif

    // ==================== FSM STATE TRANSITIONS ====================
    if (currentState == STATE_STANDBY) {
        if (prob_prefall > 0.82) {
            #if MONITOR_MODE == 0
            if (Serial) {
                Serial.println("\n⚠️ [ACTIVE WARNING] Impending fall forecasted! Warning user...");
            }
            #endif
            currentState = STATE_FALL_RISK;
            prefallStartTime = millis(); // Record start time of the prefall
            
            // Turn off standby monostables immediately to stabilize voltage
            blueLedOnUntil = 0;
            digitalWrite(LED_B, LED_OFF);
            
            // Direct HIGH GPIO constant alarm screech
            digitalWrite(LED_R, LED_ON);
            digitalWrite(BUZZER, BUZZ_ON);

            // Update Wireless BLE Characteristics
            updateBLEState(1); // 1: Prefall Warning
        }
        else if (prob_fall > 0.85) {
            #if MONITOR_MODE == 0
            if (Serial) {
                Serial.println("\n🚨 [CRITICAL ALERT] Severe fall detected by 1D-CNN!");
            }
            #endif
            currentState = STATE_SEVERE_FALL;
            
            // Update Wireless BLE Characteristics
            updateBLEState(2); // 2: Fall Alarm
        }
    }
    else if (currentState == STATE_FALL_RISK) {
        // 💡 [PREFALL RE-TRIGGER PROTECTION]
        // If the AI model continuously outputs prefall danger alert while in FALL_RISK state,
        // dynamically refresh (renew) the 2.0-second hold timer so that warning signal persists.
        if (prob_prefall > 0.82) {
            prefallStartTime = millis(); // Refresh start timer
            #if MONITOR_MODE == 0
            if (Serial) {
                Serial.println("\n🔄 [PREFALL RENEWED] Impending fall danger persists. 2.0s hold refreshed.");
            }
            #endif
        }

        unsigned long elapsedPrefall = millis() - prefallStartTime;
        
        if (prob_fall > 0.85) {
            #if MONITOR_MODE == 0
            if (Serial) {
                Serial.println("\n🚨 [CRITICAL ALERT] Fall confirmed! Siren engaged!");
            }
            #endif
            currentState = STATE_SEVERE_FALL;
            
            // Update Wireless BLE Characteristics
            updateBLEState(2); // 2: Fall Alarm
        }
        else if (elapsedPrefall >= 2000) {
            // Rigid Locking: Must hold prefall alarm for at least 2.0 seconds.
            // 2.0s later, we ONLY exit back to Standby if AI confirms return to class_static.
            if (prob_static > 0.80) {
                #if MONITOR_MODE == 0
                if (Serial) {
                    Serial.println("\n🟢 [AUTO-RECOVERY] User safely stabilized back to Static. Standby restored.");
                }
                #endif
                resetToStandby();
            }
        }
    }
}

// ==================== 11. Finite State Machine Drivers ====================
void updateFSM() {
    switch (currentState) {
        case STATE_STANDBY:
            handleStandby();
            break;
        case STATE_FALL_RISK:
            handleFallRisk();
            break;
        case STATE_SEVERE_FALL:
            handleSevereFall();
            break;
    }
}

void handleStandby() {
    digitalWrite(LED_R, LED_OFF);
    digitalWrite(LED_G, LED_OFF); // Green completely abandoned
    
    // Completely disable Buzzer to prevent bleed-through
    digitalWrite(BUZZER, BUZZ_OFF);

    // Non-blocking Monostable 1-second LED controller
    unsigned long currentMillis = millis();
    
    if (currentMillis < blueLedOnUntil) {
        digitalWrite(LED_B, LED_ON);
    } else {
        digitalWrite(LED_B, LED_OFF);
    }
}

void handleFallRisk() {
    digitalWrite(LED_R, LED_ON);
    digitalWrite(LED_B, LED_OFF);
    digitalWrite(LED_G, LED_OFF);
    digitalWrite(BUZZER, BUZZ_ON); // Pure high GPIO constant alarm
}

void handleSevereFall() {
    unsigned long currentMillis = millis();
    digitalWrite(LED_G, LED_OFF); // Green completely off
    
    // Active Buzzer ambulance siren simulation:
    unsigned long cycleTime = currentMillis % 1200;
    if (cycleTime < 600) {
        // Phase A: Rapid warning chirps (100ms cycle: 50ms ON, 50ms OFF)
        if ((cycleTime / 75) % 2 == 0) {
            digitalWrite(BUZZER, BUZZ_ON);
        } else {
            digitalWrite(BUZZER, BUZZ_OFF);
        }
    } else {
        // Phase B: Long piercing alarm (300ms ON, 100ms OFF)
        unsigned long phaseB_time = cycleTime - 600;
        if (phaseB_time < 450) {
            digitalWrite(BUZZER, BUZZ_ON);
        } else {
            digitalWrite(BUZZER, BUZZ_OFF);
        }
    }
    
    // Ambulance strobe light: alternating red and blue LEDs every 150ms
    if (currentMillis - lastBlinkTime >= 150) {
        lastBlinkTime = currentMillis;
        blinkState = !blinkState;
        
        if (blinkState) {
            digitalWrite(LED_R, LED_ON);
            digitalWrite(LED_B, LED_OFF);
        } else {
            digitalWrite(LED_R, LED_OFF);
            digitalWrite(LED_B, LED_ON);
        }
    }
    
    // Physical button reset check (Crucial for overriding persistent sirens)
    if (digitalRead(BUTTON) == BUTTON_PRESSED) {
        delay(50); // Debounce delay
        if (digitalRead(BUTTON) == BUTTON_PRESSED) {
            #if MONITOR_MODE == 0
            if (Serial) {
                Serial.println("\n🔘 RESET: Emergency alarm manual release triggered by user.");
            }
            #endif
            resetToStandby();
            while(digitalRead(BUTTON) == BUTTON_PRESSED); // Wait for physical release
        }
    }
}

void resetToStandby() {
    digitalWrite(BUZZER, BUZZ_OFF);
    
    digitalWrite(LED_R, LED_OFF);
    digitalWrite(LED_B, LED_OFF);
    digitalWrite(LED_G, LED_OFF);
    
    // Reset pipeline variables
    buffer_head = 0;
    buffer_count = 0;
    sample_counter = 0;
    
    blueLedOnUntil = 0;
    
    flag_static = 0;
    flag_walk = 0;
    flag_stairs = 0;
    flag_prefall = 0;
    flag_fall = 0;
    
    currentState = STATE_STANDBY;

    // Reset BLE State to Standby
    updateBLEState(0); // 0: Standby

    #if MONITOR_MODE == 0
    if (Serial) {
        Serial.println("🏠 System reset back to STANDBY safely.");
        Serial.println("=================================================");
    }
    #endif
}

// ==================== 12. Wireless BLE Helper ====================
void updateBLEState(uint8_t stateVal) {
    if (BLE.connected()) {
        fallStateChar.writeValue(stateVal);
    }
}
