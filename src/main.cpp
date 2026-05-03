// TB6600 common-anode control from an ESP32 using open-drain GPIOs.
// PUL+ and DIR+ are pulled up externally; the ESP32 only sinks current on PUL-/DIR-.

#include <Arduino.h>
#include <Wire.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "driver/gpio.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp32-hal-timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "soc/gpio_struct.h"

namespace {

constexpr gpio_num_t kStepPin = GPIO_NUM_25;
constexpr gpio_num_t kDirPin = GPIO_NUM_26;
constexpr gpio_num_t kDirAltPin = GPIO_NUM_33;
constexpr gpio_num_t kEnablePin = GPIO_NUM_27;
constexpr gpio_num_t kHomeSwitchPin = GPIO_NUM_32;
constexpr gpio_num_t kAs5600SdaPin = GPIO_NUM_21;
constexpr gpio_num_t kAs5600SclPin = GPIO_NUM_22;

constexpr uint32_t kStepsPerRevolution = 3200;
constexpr float kMillimetersPerStep = 0.0125f;
constexpr float kMillimetersPerMeter = 1000.0f;
constexpr float kDegreesPerRevolution = 360.0f;
constexpr float kTwoPi = 6.28318530718f;
constexpr float kRadiansPerCount = kTwoPi / 4096.0f;
constexpr int32_t kHalfTurnCounts = 2048;
constexpr uint32_t kPulseActiveUs = 20;
constexpr uint32_t kPulseInactiveMinUs = 20;
constexpr uint32_t kDirectionSetupUs = 2000;
constexpr uint32_t kTimerTickUs = 20;
constexpr uint32_t kConsoleBaudRate = 460800;
constexpr uint8_t kAs5600I2cAddress = 0x36;
constexpr uint8_t kAs5600RawAngleHighRegister = 0x0C;
constexpr uint32_t kAs5600I2cClockHz = 100000;
constexpr uint32_t kAs5600I2cTimeoutMs = 2;

constexpr int32_t kDefaultSignedStepRate = 0;
constexpr uint32_t kMinStepRateHz = 10;
constexpr uint32_t kMaxStepRateHz = 20000;
constexpr uint32_t kLimitRecoveryRateHz = 300;
constexpr uint32_t kTeacherControlRateHz = 500;
constexpr uint32_t kTelemetryRateHz = 200;
constexpr int32_t kLimitPaddingSteps = 50;
constexpr int32_t kCalibrationMaxTravelSteps = 20000;
constexpr TickType_t kMotionPollTicks = pdMS_TO_TICKS(2);
constexpr TickType_t kSwitchDebounceTicks = pdMS_TO_TICKS(15);
constexpr TickType_t kTeacherControlPeriodTicks = pdMS_TO_TICKS(1000 / kTeacherControlRateHz);
constexpr TickType_t kTelemetryPeriodTicks = pdMS_TO_TICKS(1000 / kTelemetryRateHz);
constexpr size_t kStateDimension = 4U;
constexpr float kVelocityFilterAlpha = 0.2f;

constexpr size_t kCommandBufferSize = 64;
constexpr UBaseType_t kMotionCommandQueueLength = 1;
constexpr BaseType_t kMotionTaskCore = 1;
constexpr BaseType_t kSerialTaskCore = 0;
constexpr BaseType_t kTeacherTaskCore = 1;
constexpr BaseType_t kTelemetryTaskCore = 0;
constexpr uint32_t kTaskStackBytes = 6144;

constexpr uint32_t kPulseActiveTicks = (kPulseActiveUs + kTimerTickUs - 1U) / kTimerTickUs;
constexpr uint32_t kStepPinMask = (1UL << static_cast<uint32_t>(kStepPin));

// If your mechanical setup spins the opposite direction, change this to 1.
constexpr int kDirLevelClockwise = 0;
// Set to true when positive signed step rate drives the cart toward the home switch.
constexpr bool kHomeDirectionClockwise = false;
// Wiring for the NC switch is GPIO -> switch -> GND with the ESP32 internal pull-up enabled.
// Off the switch the input reads LOW, and at the left stop it opens and reads HIGH.
constexpr int kHomeSwitchActiveLevel = 1;
constexpr int32_t kCartPositionSign = kHomeDirectionClockwise ? -1 : 1;
// For many TB6600 modules with common-anode wiring, ENA- LOW disables current
// and ENA- HIGH/released enables the outputs. If yours is opposite, swap these.
constexpr int kEnableLevelMotorEnabled = 1;
constexpr int kEnableLevelMotorDisabled = 0;

enum class MotionCommandKind : uint8_t {
    SetSpeed,
    Stop,
    HomeAndCenter,
    MoveToCenter,
};

struct MotionCommand {
    MotionCommandKind kind;
    int32_t signedStepRate;
};

enum class ControllerMode : uint8_t {
    Manual,
    TeacherLqr,
};

struct SensorSnapshot {
    uint64_t timestampUs = 0;
    int32_t cartHomeSteps = 0;
    int32_t cartCenteredSteps = 0;
    float cartHomeMm = 0.0f;
    float cartCenteredMm = 0.0f;
    float cartVelocityMmPerSec = 0.0f;
    uint16_t angleRawCounts = 0U;
    float angleDegrees = 0.0f;
    float angleRadians = 0.0f;
    float angleVelocityRadPerSec = 0.0f;
    int32_t commandStepsPerSecond = 0;
    bool sampleValid = false;
    bool sensorOnline = false;
    bool angleZeroValid = false;
    bool axisHomed = false;
    ControllerMode controllerMode = ControllerMode::Manual;
};

char g_commandBuffer[kCommandBufferSize] = {};
size_t g_commandLength = 0U;
bool g_commandOverflow = false;

hw_timer_t* g_stepTimer = nullptr;
QueueHandle_t g_motionCommandQueue = nullptr;
TaskHandle_t g_motionTaskHandle = nullptr;
TaskHandle_t g_serialTaskHandle = nullptr;
TaskHandle_t g_teacherTaskHandle = nullptr;
TaskHandle_t g_telemetryTaskHandle = nullptr;
portMUX_TYPE g_motionMux = portMUX_INITIALIZER_UNLOCKED;

volatile int32_t g_commandedSignedStepRate = kDefaultSignedStepRate;
volatile uint32_t g_activeStepRateHz = 0U;
volatile int32_t g_positionSteps = 0;
volatile uint32_t g_phaseTicksRemaining = 0U;
volatile uint32_t g_inactiveTicks = 1U;
volatile bool g_motionActive = false;
volatile bool g_directionClockwise = true;
volatile bool g_stepSignalActive = false;
volatile bool g_driverEnabled = false;
volatile bool g_axisHomed = false;
volatile bool g_homingInProgress = false;
volatile bool g_abortMotionSequence = false;
volatile int32_t g_axisTravelSteps = 0;
volatile int32_t g_axisCenterSteps = 0;
volatile int32_t g_softLimitMinSteps = 0;
volatile int32_t g_softLimitMaxSteps = 0;
volatile bool g_sensorOnline = false;
volatile bool g_sensorSampleValid = false;
volatile bool g_angleZeroValid = false;
volatile bool g_telemetryEnabled = false;
volatile bool g_as5600AckSeen = false;
volatile bool g_as5600RawReadOk = false;
volatile ControllerMode g_controllerMode = ControllerMode::Manual;
volatile uint16_t g_lastAngleRawCounts = 0U;
volatile int32_t g_lastContinuousAngleCounts = 0;
volatile int32_t g_angleZeroContinuousCounts = 0;
volatile float g_lastCartVelocityMmPerSec = 0.0f;
volatile float g_lastAngleVelocityRadPerSec = 0.0f;
volatile uint64_t g_lastSensorTimestampUs = 0U;
volatile uint8_t g_lastAs5600I2cStatus = 0xFFU;
SensorSnapshot g_latestSensorSnapshot = {};
float g_teacherLqrGain[kStateDimension] = {0.0f, 0.0f, 0.0f, 0.0f};
bool g_telemetryHeaderPrinted = false;

uint32_t max_u32(uint32_t lhs, uint32_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

uint32_t min_u32(uint32_t lhs, uint32_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

uint32_t clamp_u32(uint32_t value, uint32_t lower, uint32_t upper)
{
    return min_u32(max_u32(value, lower), upper);
}

int32_t abs_i32(int32_t value)
{
    return (value < 0) ? -value : value;
}

double steps_to_mm(int32_t steps)
{
    return static_cast<double>(steps) * static_cast<double>(kMillimetersPerStep);
}

double steps_to_meters(int32_t steps)
{
    return steps_to_mm(steps) / static_cast<double>(kMillimetersPerMeter);
}

int32_t wrap_angle_count_delta(int32_t deltaCounts)
{
    if (deltaCounts > 2048) {
        deltaCounts -= 4096;
    } else if (deltaCounts < -2048) {
        deltaCounts += 4096;
    }
    return deltaCounts;
}

bool read_home_switch_active()
{
    return gpio_get_level(kHomeSwitchPin) == kHomeSwitchActiveLevel;
}

uint32_t magnitude_from_signed_rate(long signedStepRate)
{
    if (signedStepRate == 0L) {
        return 0U;
    }

    const long magnitude = (signedStepRate < 0L) ? -signedStepRate : signedStepRate;
    return clamp_u32(static_cast<uint32_t>(magnitude), kMinStepRateHz, kMaxStepRateHz);
}

int32_t signed_rate_from_direction(bool clockwise, uint32_t magnitude);
bool queue_motion_command(MotionCommandKind kind);
bool sample_sensor_snapshot(bool forceRead);
void set_signed_step_rate(int32_t requestedSignedStepRate, bool enforceSafety, bool emitSerial);

const char* reset_reason_name(esp_reset_reason_t resetReason)
{
    switch (resetReason) {
        case ESP_RST_UNKNOWN:
            return "UNKNOWN";
        case ESP_RST_POWERON:
            return "POWERON";
        case ESP_RST_EXT:
            return "EXTERNAL";
        case ESP_RST_SW:
            return "SOFTWARE";
        case ESP_RST_PANIC:
            return "PANIC";
        case ESP_RST_INT_WDT:
            return "INT_WDT";
        case ESP_RST_TASK_WDT:
            return "TASK_WDT";
        case ESP_RST_WDT:
            return "OTHER_WDT";
        case ESP_RST_DEEPSLEEP:
            return "DEEPSLEEP";
        case ESP_RST_BROWNOUT:
            return "BROWNOUT";
        case ESP_RST_SDIO:
            return "SDIO";
        default:
            return "UNMAPPED";
    }
}

int32_t clamp_signed_step_rate(int32_t requestedSignedStepRate)
{
    const uint32_t magnitude = magnitude_from_signed_rate(requestedSignedStepRate);
    if (requestedSignedStepRate == 0) {
        return 0;
    }
    return signed_rate_from_direction(requestedSignedStepRate > 0, magnitude);
}

int32_t signed_rate_from_direction(bool clockwise, uint32_t magnitude)
{
    return clockwise ? static_cast<int32_t>(magnitude) : -static_cast<int32_t>(magnitude);
}

int32_t home_signed_step_rate(uint32_t magnitude)
{
    return signed_rate_from_direction(kHomeDirectionClockwise, magnitude);
}

int32_t away_from_home_signed_step_rate(uint32_t magnitude)
{
    return signed_rate_from_direction(!kHomeDirectionClockwise, magnitude);
}

bool signed_rate_moves_toward_home(int32_t signedStepRate)
{
    return (signedStepRate != 0) && ((signedStepRate > 0) == kHomeDirectionClockwise);
}

bool signed_rate_moves_away_from_home(int32_t signedStepRate)
{
    return (signedStepRate != 0) && !signed_rate_moves_toward_home(signedStepRate);
}

int32_t raw_position_steps()
{
    portENTER_CRITICAL(&g_motionMux);
    const int32_t positionSteps = g_positionSteps;
    portEXIT_CRITICAL(&g_motionMux);
    return positionSteps;
}

void set_raw_position_steps(int32_t rawSteps)
{
    portENTER_CRITICAL(&g_motionMux);
    g_positionSteps = rawSteps;
    portEXIT_CRITICAL(&g_motionMux);
}

int32_t cart_position_steps()
{
    return raw_position_steps() * kCartPositionSign;
}

int32_t axis_center_steps()
{
    portENTER_CRITICAL(&g_motionMux);
    const int32_t axisCenterSteps = g_axisCenterSteps;
    portEXIT_CRITICAL(&g_motionMux);
    return axisCenterSteps;
}

int32_t cart_centered_position_steps()
{
    return cart_position_steps() - axis_center_steps();
}

int32_t hard_limit_min_steps()
{
    portENTER_CRITICAL(&g_motionMux);
    const int32_t hardLimitMinSteps = g_softLimitMinSteps;
    portEXIT_CRITICAL(&g_motionMux);
    return hardLimitMinSteps;
}

int32_t hard_limit_max_steps()
{
    portENTER_CRITICAL(&g_motionMux);
    const int32_t hardLimitMaxSteps = g_softLimitMaxSteps;
    portEXIT_CRITICAL(&g_motionMux);
    return hardLimitMaxSteps;
}

void clear_axis_calibration()
{
    portENTER_CRITICAL(&g_motionMux);
    g_axisHomed = false;
    g_axisTravelSteps = 0;
    g_axisCenterSteps = 0;
    g_softLimitMinSteps = 0;
    g_softLimitMaxSteps = 0;
    portEXIT_CRITICAL(&g_motionMux);
}

void set_as5600_diagnostics(bool ackSeen, bool rawReadOk, uint8_t i2cStatus)
{
    portENTER_CRITICAL(&g_motionMux);
    g_as5600AckSeen = ackSeen;
    g_as5600RawReadOk = rawReadOk;
    g_lastAs5600I2cStatus = i2cStatus;
    portEXIT_CRITICAL(&g_motionMux);
}

void get_as5600_diagnostics(bool* ackSeen, bool* rawReadOk, uint8_t* i2cStatus)
{
    portENTER_CRITICAL(&g_motionMux);
    *ackSeen = g_as5600AckSeen;
    *rawReadOk = g_as5600RawReadOk;
    *i2cStatus = g_lastAs5600I2cStatus;
    portEXIT_CRITICAL(&g_motionMux);
}

void reset_sensor_sample_state()
{
    portENTER_CRITICAL(&g_motionMux);
    g_sensorSampleValid = false;
    g_lastSensorTimestampUs = 0U;
    g_lastCartVelocityMmPerSec = 0.0f;
    g_lastAngleVelocityRadPerSec = 0.0f;
    g_latestSensorSnapshot.sampleValid = false;
    portEXIT_CRITICAL(&g_motionMux);

    set_as5600_diagnostics(false, false, 0xFFU);
}

void update_axis_calibration(int32_t travelSteps)
{
    const int32_t clampedTravelSteps = abs_i32(travelSteps);
    const int32_t axisCenterSteps = clampedTravelSteps / 2;
    const int32_t centeredLimitExtent = max(0, axisCenterSteps - kLimitPaddingSteps);
    const int32_t softLimitMin = -centeredLimitExtent;
    const int32_t softLimitMax = centeredLimitExtent;

    portENTER_CRITICAL(&g_motionMux);
    g_axisHomed = true;
    g_axisTravelSteps = clampedTravelSteps;
    g_axisCenterSteps = axisCenterSteps;
    g_softLimitMinSteps = softLimitMin;
    g_softLimitMaxSteps = softLimitMax;
    portEXIT_CRITICAL(&g_motionMux);
}

void set_homing_in_progress(bool active)
{
    portENTER_CRITICAL(&g_motionMux);
    g_homingInProgress = active;
    portEXIT_CRITICAL(&g_motionMux);
}

void set_controller_mode(ControllerMode controllerMode)
{
    portENTER_CRITICAL(&g_motionMux);
    g_controllerMode = controllerMode;
    portEXIT_CRITICAL(&g_motionMux);
}

ControllerMode current_controller_mode()
{
    portENTER_CRITICAL(&g_motionMux);
    const ControllerMode controllerMode = g_controllerMode;
    portEXIT_CRITICAL(&g_motionMux);
    return controllerMode;
}

void set_telemetry_enabled(bool enabled)
{
    portENTER_CRITICAL(&g_motionMux);
    g_telemetryEnabled = enabled;
    portEXIT_CRITICAL(&g_motionMux);

    if (!enabled) {
        g_telemetryHeaderPrinted = false;
    }
}

bool telemetry_enabled()
{
    portENTER_CRITICAL(&g_motionMux);
    const bool enabled = g_telemetryEnabled;
    portEXIT_CRITICAL(&g_motionMux);
    return enabled;
}

const char* controller_mode_name(ControllerMode controllerMode)
{
    switch (controllerMode) {
        case ControllerMode::Manual:
            return "MANUAL";
        case ControllerMode::TeacherLqr:
            return "TEACHER";
    }

    return "UNKNOWN";
}

void copy_teacher_lqr_gains(float gainsOut[kStateDimension])
{
    portENTER_CRITICAL(&g_motionMux);
    for (size_t index = 0; index < kStateDimension; ++index) {
        gainsOut[index] = g_teacherLqrGain[index];
    }
    portEXIT_CRITICAL(&g_motionMux);
}

void set_teacher_lqr_gains(float positionGain, float velocityGain, float angleGain, float angleRateGain)
{
    portENTER_CRITICAL(&g_motionMux);
    g_teacherLqrGain[0] = positionGain;
    g_teacherLqrGain[1] = velocityGain;
    g_teacherLqrGain[2] = angleGain;
    g_teacherLqrGain[3] = angleRateGain;
    portEXIT_CRITICAL(&g_motionMux);
}

bool teacher_lqr_gains_are_zero()
{
    float gains[kStateDimension] = {};
    copy_teacher_lqr_gains(gains);
    return (fabsf(gains[0]) < 1e-6f) && (fabsf(gains[1]) < 1e-6f) && (fabsf(gains[2]) < 1e-6f) &&
           (fabsf(gains[3]) < 1e-6f);
}

SensorSnapshot latest_sensor_snapshot()
{
    portENTER_CRITICAL(&g_motionMux);
    const SensorSnapshot snapshot = g_latestSensorSnapshot;
    portEXIT_CRITICAL(&g_motionMux);
    return snapshot;
}

void request_motion_sequence_abort()
{
    portENTER_CRITICAL(&g_motionMux);
    g_abortMotionSequence = true;
    portEXIT_CRITICAL(&g_motionMux);
}

void clear_motion_sequence_abort()
{
    portENTER_CRITICAL(&g_motionMux);
    g_abortMotionSequence = false;
    portEXIT_CRITICAL(&g_motionMux);
}

bool motion_sequence_abort_requested()
{
    portENTER_CRITICAL(&g_motionMux);
    const bool abortRequested = g_abortMotionSequence;
    portEXIT_CRITICAL(&g_motionMux);
    return abortRequested;
}

int32_t apply_motion_safety(int32_t requestedSignedStepRate)
{
    const int32_t clampedRequestedRate = clamp_signed_step_rate(requestedSignedStepRate);
    if (clampedRequestedRate == 0) {
        return 0;
    }

    if (read_home_switch_active() && signed_rate_moves_toward_home(clampedRequestedRate)) {
        return 0;
    }

    bool axisHomed = false;
    int32_t hardLimitMinSteps = 0;
    int32_t hardLimitMaxSteps = 0;

    portENTER_CRITICAL(&g_motionMux);
    axisHomed = g_axisHomed;
    hardLimitMinSteps = g_softLimitMinSteps;
    hardLimitMaxSteps = g_softLimitMaxSteps;
    portEXIT_CRITICAL(&g_motionMux);

    if (!axisHomed) {
        return clampedRequestedRate;
    }

    const int32_t currentCartPositionSteps = cart_centered_position_steps();
    if ((currentCartPositionSteps <= hardLimitMinSteps) &&
        signed_rate_moves_toward_home(clampedRequestedRate)) {
        return 0;
    }

    if ((currentCartPositionSteps >= hardLimitMaxSteps) &&
        signed_rate_moves_away_from_home(clampedRequestedRate)) {
        return 0;
    }

    return clampedRequestedRate;
}

void enforce_runtime_motion_safety()
{
    bool motionActive = false;
    bool axisHomed = false;
    bool directionClockwise = false;
    uint32_t activeStepRateHz = 0U;
    int32_t positionSteps = 0;
    int32_t axisCenterSteps = 0;
    int32_t hardLimitMinSteps = 0;
    int32_t hardLimitMaxSteps = 0;

    portENTER_CRITICAL(&g_motionMux);
    motionActive = g_motionActive;
    axisHomed = g_axisHomed;
    directionClockwise = g_directionClockwise;
    activeStepRateHz = g_activeStepRateHz;
    positionSteps = g_positionSteps;
    axisCenterSteps = g_axisCenterSteps;
    hardLimitMinSteps = g_softLimitMinSteps;
    hardLimitMaxSteps = g_softLimitMaxSteps;
    portEXIT_CRITICAL(&g_motionMux);

    if (!motionActive) {
        return;
    }

    const int32_t activeSignedStepRate =
        directionClockwise ? static_cast<int32_t>(activeStepRateHz) : -static_cast<int32_t>(activeStepRateHz);
    const int32_t currentCartSteps = (positionSteps * kCartPositionSign) - axisCenterSteps;
    const bool hitHomeLimit = axisHomed && (currentCartSteps <= hardLimitMinSteps) &&
                              signed_rate_moves_toward_home(activeSignedStepRate);
    const bool hitFarLimit = axisHomed && (currentCartSteps >= hardLimitMaxSteps) &&
                             signed_rate_moves_away_from_home(activeSignedStepRate);
    const bool hitSwitch = read_home_switch_active() && signed_rate_moves_toward_home(activeSignedStepRate);

    if (!(hitHomeLimit || hitFarLimit || hitSwitch)) {
        return;
    }

    set_signed_step_rate(0, false, true);
    Serial.printf(
        "Hard limit stop: cart=%ld steps hard=[%ld,%ld]\n",
        static_cast<long>(currentCartSteps),
        static_cast<long>(hardLimitMinSteps),
        static_cast<long>(hardLimitMaxSteps));
}

bool as5600_read_raw_angle(uint16_t* rawAngleCounts)
{
    Wire.beginTransmission(kAs5600I2cAddress);
    const uint8_t probeStatus = Wire.endTransmission(true);
    if (probeStatus != 0U) {
        set_as5600_diagnostics(false, false, probeStatus);
        return false;
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool useRepeatedStart = attempt == 0;

        Wire.beginTransmission(kAs5600I2cAddress);
        Wire.write(kAs5600RawAngleHighRegister);
        const uint8_t txStatus = Wire.endTransmission(!useRepeatedStart);
        if (txStatus != 0U) {
            set_as5600_diagnostics(true, false, txStatus);
            continue;
        }

        const uint8_t bytesRequested = 2;
        const size_t bytesRead = Wire.requestFrom(static_cast<uint8_t>(kAs5600I2cAddress), bytesRequested);
        if (bytesRead != bytesRequested) {
            while (Wire.available() > 0) {
                (void)Wire.read();
            }
            set_as5600_diagnostics(true, false, 0U);
            continue;
        }

        const uint8_t highByte = static_cast<uint8_t>(Wire.read());
        const uint8_t lowByte = static_cast<uint8_t>(Wire.read());
        *rawAngleCounts = static_cast<uint16_t>(((highByte & 0x0FU) << 8U) | lowByte);
        set_as5600_diagnostics(true, true, 0U);
        return true;
    }

    return false;
}

void print_sensor_diagnostics()
{
    const bool sampleValid = sample_sensor_snapshot(true);
    const SensorSnapshot snapshot = latest_sensor_snapshot();
    bool ackSeen = false;
    bool rawReadOk = false;
    uint8_t i2cStatus = 0xFFU;
    get_as5600_diagnostics(&ackSeen, &rawReadOk, &i2cStatus);

    Serial.printf(
        "AS5600 addr=0x%02X sda=%d scl=%d bus=%lu ack=%s rawRead=%s i2cStatus=%u sample=%s raw=%u angleDeg=%.2f angleRad=%.4f\n",
        static_cast<unsigned int>(kAs5600I2cAddress),
        static_cast<int>(kAs5600SdaPin),
        static_cast<int>(kAs5600SclPin),
        static_cast<unsigned long>(kAs5600I2cClockHz),
        ackSeen ? "yes" : "no",
        rawReadOk ? "yes" : "no",
        static_cast<unsigned int>(i2cStatus),
        sampleValid ? "yes" : "no",
        static_cast<unsigned int>(snapshot.angleRawCounts),
        snapshot.angleDegrees,
        snapshot.angleRadians);

    if (!ackSeen) {
        Serial.println(
            "AS5600 not acknowledging. Check 3.3V power, common GND, SDA on GPIO21, SCL on GPIO22, and I2C pull-ups.");
    } else if (!rawReadOk) {
        Serial.println(
            "AS5600 acknowledged but the raw-angle read still failed. Recheck wiring quality, pull-ups, and magnet placement.");
    }
}

bool sample_sensor_snapshot(bool forceRead = false)
{
    const ControllerMode controllerMode = current_controller_mode();
    const bool shouldRead = forceRead || telemetry_enabled() || (controllerMode == ControllerMode::TeacherLqr);
    if (!shouldRead) {
        return false;
    }

    uint16_t rawAngleCounts = 0U;
    if (!as5600_read_raw_angle(&rawAngleCounts)) {
        portENTER_CRITICAL(&g_motionMux);
        g_sensorOnline = false;
        g_sensorSampleValid = false;
        g_latestSensorSnapshot.sensorOnline = false;
        g_latestSensorSnapshot.sampleValid = false;
        portEXIT_CRITICAL(&g_motionMux);
        return false;
    }

    const uint64_t timestampUs = static_cast<uint64_t>(esp_timer_get_time());
    const int32_t cartHomeSteps = cart_position_steps();
    const int32_t cartCenteredSteps = cart_centered_position_steps();
    const float cartHomeMm = static_cast<float>(steps_to_mm(cartHomeSteps));
    const float cartCenteredMm = static_cast<float>(steps_to_mm(cartCenteredSteps));

    bool previousSampleValid = false;
    bool angleZeroValid = false;
    bool axisHomed = false;
    int32_t angleZeroContinuousCounts = 0;
    int32_t previousContinuousAngleCounts = 0;
    uint16_t previousRawAngleCounts = 0U;
    uint64_t previousTimestampUs = 0U;
    float previousCartVelocityMmPerSec = 0.0f;
    float previousAngleVelocityRadPerSec = 0.0f;
    float previousCartCenteredMm = 0.0f;
    float previousAngleRadians = 0.0f;
    int32_t commandStepsPerSecond = 0;

    portENTER_CRITICAL(&g_motionMux);
    previousSampleValid = g_sensorSampleValid;
    angleZeroValid = g_angleZeroValid;
    axisHomed = g_axisHomed;
    angleZeroContinuousCounts = g_angleZeroContinuousCounts;
    previousContinuousAngleCounts = g_lastContinuousAngleCounts;
    previousRawAngleCounts = g_lastAngleRawCounts;
    previousTimestampUs = g_lastSensorTimestampUs;
    previousCartVelocityMmPerSec = g_lastCartVelocityMmPerSec;
    previousAngleVelocityRadPerSec = g_lastAngleVelocityRadPerSec;
    previousCartCenteredMm = g_latestSensorSnapshot.cartCenteredMm;
    previousAngleRadians = g_latestSensorSnapshot.angleRadians;
    commandStepsPerSecond = g_commandedSignedStepRate;
    portEXIT_CRITICAL(&g_motionMux);

    int32_t continuousAngleCounts = static_cast<int32_t>(rawAngleCounts);
    if (previousSampleValid) {
        continuousAngleCounts = previousContinuousAngleCounts +
                                wrap_angle_count_delta(static_cast<int32_t>(rawAngleCounts) -
                                                       static_cast<int32_t>(previousRawAngleCounts));
    }

        const float relativeAngleRadians = angleZeroValid
                                 ? static_cast<float>(continuousAngleCounts - angleZeroContinuousCounts) *
                                     kRadiansPerCount
                                 : 0.0f;
    const float relativeAngleDegrees = angleZeroValid ? (relativeAngleRadians * 180.0f / PI) : 0.0f;

    float cartVelocityMmPerSec = 0.0f;
    float angleVelocityRadPerSec = 0.0f;
    if (previousSampleValid && (timestampUs > previousTimestampUs)) {
        const float dtSeconds = static_cast<float>(timestampUs - previousTimestampUs) * 1e-6f;
        const float rawCartVelocityMmPerSec = (cartCenteredMm - previousCartCenteredMm) / dtSeconds;
        cartVelocityMmPerSec = previousCartVelocityMmPerSec +
                               (kVelocityFilterAlpha * (rawCartVelocityMmPerSec - previousCartVelocityMmPerSec));

        if (angleZeroValid) {
            const float rawAngleVelocityRadPerSec = (relativeAngleRadians - previousAngleRadians) / dtSeconds;
            angleVelocityRadPerSec = previousAngleVelocityRadPerSec +
                                     (kVelocityFilterAlpha *
                                      (rawAngleVelocityRadPerSec - previousAngleVelocityRadPerSec));
        }
    }

    SensorSnapshot snapshot = {};
    snapshot.timestampUs = timestampUs;
    snapshot.cartHomeSteps = cartHomeSteps;
    snapshot.cartCenteredSteps = cartCenteredSteps;
    snapshot.cartHomeMm = cartHomeMm;
    snapshot.cartCenteredMm = cartCenteredMm;
    snapshot.cartVelocityMmPerSec = cartVelocityMmPerSec;
    snapshot.angleRawCounts = rawAngleCounts;
    snapshot.angleDegrees = relativeAngleDegrees;
    snapshot.angleRadians = relativeAngleRadians;
    snapshot.angleVelocityRadPerSec = angleVelocityRadPerSec;
    snapshot.commandStepsPerSecond = commandStepsPerSecond;
    snapshot.sampleValid = true;
    snapshot.sensorOnline = true;
    snapshot.angleZeroValid = angleZeroValid;
    snapshot.axisHomed = axisHomed;
    snapshot.controllerMode = controllerMode;

    portENTER_CRITICAL(&g_motionMux);
    g_sensorOnline = true;
    g_sensorSampleValid = true;
    g_lastAngleRawCounts = rawAngleCounts;
    g_lastContinuousAngleCounts = continuousAngleCounts;
    g_lastCartVelocityMmPerSec = cartVelocityMmPerSec;
    g_lastAngleVelocityRadPerSec = angleVelocityRadPerSec;
    g_lastSensorTimestampUs = timestampUs;
    g_latestSensorSnapshot = snapshot;
    portEXIT_CRITICAL(&g_motionMux);

    return true;
}

bool capture_rest_angle_reference()
{
    if (!sample_sensor_snapshot(true)) {
        Serial.println("Rest-angle capture unavailable: AS5600 not detected on I2C");
        return false;
    }

    uint16_t rawAngleCounts = 0U;
    int32_t continuousAngleCounts = 0;

    portENTER_CRITICAL(&g_motionMux);
    rawAngleCounts = g_lastAngleRawCounts;
    continuousAngleCounts = g_lastContinuousAngleCounts;
    g_angleZeroContinuousCounts = continuousAngleCounts - kHalfTurnCounts;
    g_angleZeroValid = true;
    portEXIT_CRITICAL(&g_motionMux);

    reset_sensor_sample_state();
    (void)sample_sensor_snapshot(true);
    Serial.printf(
        "Rest angle captured: raw=%u. This hanging position is +180 deg; upright is 0 deg.\n",
        static_cast<unsigned int>(rawAngleCounts));
    return true;
}

void print_teacher_gains()
{
    float gains[kStateDimension] = {};
    copy_teacher_lqr_gains(gains);
    Serial.printf(
        "Teacher LQR gains [cart_m cart_mps angle_rad angle_rate_radps] -> steps/s: [%.6f %.6f %.6f %.6f]\n",
        gains[0],
        gains[1],
        gains[2],
        gains[3]);
}

bool enable_teacher_mode()
{
    if (!sample_sensor_snapshot(true)) {
        Serial.println("Teacher mode unavailable: AS5600 read failed");
        return false;
    }

    bool axisHomed = false;
    bool angleZeroValid = false;
    portENTER_CRITICAL(&g_motionMux);
    axisHomed = g_axisHomed;
    angleZeroValid = g_angleZeroValid;
    portEXIT_CRITICAL(&g_motionMux);

    if (!axisHomed) {
        Serial.println("Teacher mode unavailable: run HOME first so cart position is centered");
        return false;
    }

    if (!angleZeroValid) {
        Serial.println("Teacher mode unavailable: let the pole hang and run RESTANGLE first");
        return false;
    }

    if (teacher_lqr_gains_are_zero()) {
        Serial.println("Teacher mode warning: LQR gains are all zero; use SETK before expecting control action");
    }

    set_controller_mode(ControllerMode::TeacherLqr);
    Serial.printf("Teacher LQR enabled at %lu Hz\n", static_cast<unsigned long>(kTeacherControlRateHz));
    return true;
}

void disable_teacher_mode(bool stopMotion)
{
    set_controller_mode(ControllerMode::Manual);
    if (stopMotion) {
        (void)queue_motion_command(MotionCommandKind::Stop);
    }
    Serial.println("Teacher LQR disabled");
}

int32_t compute_teacher_command_steps_per_second(const SensorSnapshot& snapshot)
{
    float gains[kStateDimension] = {};
    copy_teacher_lqr_gains(gains);

    const float state[kStateDimension] = {
        static_cast<float>(steps_to_meters(snapshot.cartCenteredSteps)),
        snapshot.cartVelocityMmPerSec / kMillimetersPerMeter,
        snapshot.angleRadians,
        snapshot.angleVelocityRadPerSec,
    };

    float requestedStepsPerSecond = 0.0f;
    for (size_t index = 0; index < kStateDimension; ++index) {
        requestedStepsPerSecond -= gains[index] * state[index];
    }

    return static_cast<int32_t>(lroundf(requestedStepsPerSecond));
}

int32_t current_signed_step_rate()
{
    portENTER_CRITICAL(&g_motionMux);
    const int32_t signedRate = g_motionActive
                                   ? signed_rate_from_direction(g_directionClockwise, g_activeStepRateHz)
                                   : 0;
    portEXIT_CRITICAL(&g_motionMux);
    return signedRate;
}

void signal_inactive(gpio_num_t pin)
{
    (void)gpio_set_level(pin, 1);
}

void signal_active(gpio_num_t pin)
{
    (void)gpio_set_level(pin, 0);
}

void IRAM_ATTR step_signal_inactive_isr()
{
    GPIO.out_w1ts = kStepPinMask;
}

void IRAM_ATTR step_signal_active_isr()
{
    GPIO.out_w1tc = kStepPinMask;
}

void configure_open_drain_output(gpio_num_t pin)
{
    gpio_config_t config = {};
    config.pin_bit_mask = (1ULL << static_cast<uint32_t>(pin));
    config.mode = GPIO_MODE_OUTPUT_OD;
    config.pull_up_en = GPIO_PULLUP_DISABLE;
    config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    config.intr_type = GPIO_INTR_DISABLE;

    (void)gpio_config(&config);
    signal_inactive(pin);
}

void configure_input_pullup(gpio_num_t pin)
{
    gpio_config_t config = {};
    config.pin_bit_mask = (1ULL << static_cast<uint32_t>(pin));
    config.mode = GPIO_MODE_INPUT;
    config.pull_up_en = GPIO_PULLUP_ENABLE;
    config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    config.intr_type = GPIO_INTR_DISABLE;

    (void)gpio_config(&config);
}

void angle_sensor_init()
{
    pinMode(static_cast<uint8_t>(kAs5600SdaPin), INPUT_PULLUP);
    pinMode(static_cast<uint8_t>(kAs5600SclPin), INPUT_PULLUP);
    Wire.begin(static_cast<int>(kAs5600SdaPin), static_cast<int>(kAs5600SclPin), kAs5600I2cClockHz);
    Wire.setTimeOut(kAs5600I2cTimeoutMs);
}

void release_direction_signal()
{
    (void)gpio_set_direction(kDirPin, GPIO_MODE_INPUT);
    (void)gpio_set_direction(kDirAltPin, GPIO_MODE_INPUT);
}

void assert_direction_signal()
{
    (void)gpio_set_direction(kDirPin, GPIO_MODE_OUTPUT_OD);
    (void)gpio_set_direction(kDirAltPin, GPIO_MODE_OUTPUT_OD);
    signal_active(kDirPin);
    signal_active(kDirAltPin);
}

void set_driver_enabled(bool enabled)
{
    const int enableLevel = enabled ? kEnableLevelMotorEnabled : kEnableLevelMotorDisabled;
    (void)gpio_set_level(kEnablePin, enableLevel);

    portENTER_CRITICAL(&g_motionMux);
    g_driverEnabled = enabled;
    portEXIT_CRITICAL(&g_motionMux);
}

uint32_t pulse_period_us_from_rate(uint32_t stepRateHz)
{
    const uint32_t clampedRateHz = clamp_u32(stepRateHz, kMinStepRateHz, kMaxStepRateHz);
    const uint32_t minPeriodUs = kPulseActiveUs + kPulseInactiveMinUs;
    return max_u32(1000000UL / clampedRateHz, minPeriodUs);
}

uint32_t period_ticks_from_rate(uint32_t stepRateHz)
{
    const uint32_t periodUs = pulse_period_us_from_rate(stepRateHz);
    const uint32_t roundedTicks = (periodUs + kTimerTickUs - 1U) / kTimerTickUs;
    return max_u32(roundedTicks, kPulseActiveTicks + 1U);
}

double speed_rpm_from_rate(uint32_t stepRateHz)
{
    return (60.0 * static_cast<double>(stepRateHz)) / static_cast<double>(kStepsPerRevolution);
}

double speed_rpm_from_signed_rate(int32_t signedStepRate)
{
    return speed_rpm_from_rate(magnitude_from_signed_rate(signedStepRate));
}

char* trim_in_place(char* text)
{
    while ((*text != '\0') && isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }

    char* end = text + strlen(text);
    while ((end > text) && isspace(static_cast<unsigned char>(end[-1]))) {
        --end;
    }

    *end = '\0';
    return text;
}

void print_help()
{
    Serial.println("Commands:");
    Serial.println("  SENSOR");
    Serial.println("  RESTANGLE");
    Serial.println("  ANGLEZERO   (alias for RESTANGLE)");
    Serial.println("  SETK <cart_m> <cart_mps> <angle_rad> <angle_rate_radps>");
    Serial.println("  GAINS");
    Serial.println("  TEACHER ON");
    Serial.println("  TEACHER OFF");
    Serial.println("  LOG ON");
    Serial.println("  LOG OFF");
    Serial.println("  HOME");
    Serial.println("  CENTER");
    Serial.println("  MOVE <signed_steps_per_sec>");
    Serial.println("  SPEED <signed_steps_per_sec>");
    Serial.println("  <signed_steps_per_sec>");
    Serial.println("  STOP");
    Serial.println("  STATUS");
    Serial.println("  HELP");
    Serial.printf(
        "Commanded speed: %+ld steps/s (%.2f rpm)\n",
        static_cast<long>(g_commandedSignedStepRate),
        speed_rpm_from_signed_rate(g_commandedSignedStepRate));
}

void print_status()
{
    int32_t commandedSignedStepRate = 0;
    int32_t activeSignedStepRate = 0;
    int32_t positionSteps = 0;
    int32_t axisTravelSteps = 0;
    int32_t axisCenterSteps = 0;
    int32_t softLimitMinSteps = 0;
    int32_t softLimitMaxSteps = 0;
    bool active = false;
    bool clockwise = true;
    bool driverEnabled = false;
    bool axisHomed = false;
    bool homingInProgress = false;
    bool telemetryEnabled = false;
    bool as5600AckSeen = false;
    bool as5600RawReadOk = false;
    uint8_t lastAs5600I2cStatus = 0xFFU;
    int dirPinLevel = 0;
    int dirAltPinLevel = 0;
    int enablePinLevel = 0;
    const bool homeSwitchActive = read_home_switch_active();
    const SensorSnapshot sensorSnapshot = latest_sensor_snapshot();
    const ControllerMode controllerMode = current_controller_mode();
    float gains[kStateDimension] = {};
    copy_teacher_lqr_gains(gains);
    get_as5600_diagnostics(&as5600AckSeen, &as5600RawReadOk, &lastAs5600I2cStatus);

    portENTER_CRITICAL(&g_motionMux);
    commandedSignedStepRate = g_commandedSignedStepRate;
    activeSignedStepRate = g_motionActive
                               ? signed_rate_from_direction(g_directionClockwise, g_activeStepRateHz)
                               : 0;
    positionSteps = g_positionSteps;
    axisTravelSteps = g_axisTravelSteps;
    axisCenterSteps = g_axisCenterSteps;
    softLimitMinSteps = g_softLimitMinSteps;
    softLimitMaxSteps = g_softLimitMaxSteps;
    active = g_motionActive;
    clockwise = g_directionClockwise;
    driverEnabled = g_driverEnabled;
    axisHomed = g_axisHomed;
    homingInProgress = g_homingInProgress;
    telemetryEnabled = g_telemetryEnabled;
    portEXIT_CRITICAL(&g_motionMux);

    dirPinLevel = gpio_get_level(kDirPin);
    dirAltPinLevel = gpio_get_level(kDirAltPin);
    enablePinLevel = gpio_get_level(kEnablePin);

    const int32_t currentHomeSteps = positionSteps * kCartPositionSign;
    const int32_t currentCartSteps = currentHomeSteps - axisCenterSteps;

    Serial.printf(
        "cart=%ld steps (%.2f mm) home=%ld steps raw=%ld active=%s ena=%s enaPin=%s dir=%s dir26=%s dir33=%s commanded=%+ld steps/s current=%+ld steps/s homeStop=%s homed=%s homing=%s travel=%ld steps centerHome=%ld steps limits=[%ld,%ld]\n",
        static_cast<long>(currentCartSteps),
        steps_to_mm(currentCartSteps),
        static_cast<long>(currentHomeSteps),
        static_cast<long>(positionSteps),
        active ? "yes" : "no",
        driverEnabled ? "ON" : "OFF",
        (enablePinLevel == 0) ? "LOW" : "HIGH",
        clockwise ? "CW" : "CCW",
        (dirPinLevel == 0) ? "LOW" : "HIGH",
        (dirAltPinLevel == 0) ? "LOW" : "HIGH",
        static_cast<long>(commandedSignedStepRate),
        static_cast<long>(activeSignedStepRate),
        homeSwitchActive ? "TRIGGERED" : "clear",
        axisHomed ? "yes" : "no",
        homingInProgress ? "yes" : "no",
        static_cast<long>(axisTravelSteps),
        static_cast<long>(axisCenterSteps),
        static_cast<long>(hard_limit_min_steps()),
        static_cast<long>(hard_limit_max_steps()));

    Serial.printf(
        "mode=%s telemetry=%s sensor=%s ack=%s rawRead=%s i2c=%u sample=%s uprightRef=%s cartCenter=%ld steps (%.2f mm) angle=%.4f rad (%.2f deg) cartVel=%.2f mm/s angleVel=%.4f rad/s sampleTs=%llu K=[%.6f %.6f %.6f %.6f]\n",
        controller_mode_name(controllerMode),
        telemetryEnabled ? "ON" : "OFF",
        sensorSnapshot.sensorOnline ? "online" : "offline",
        as5600AckSeen ? "yes" : "no",
        as5600RawReadOk ? "yes" : "no",
        static_cast<unsigned int>(lastAs5600I2cStatus),
        sensorSnapshot.sampleValid ? "yes" : "no",
        sensorSnapshot.angleZeroValid ? "yes" : "no",
        static_cast<long>(sensorSnapshot.cartCenteredSteps),
        sensorSnapshot.cartCenteredMm,
        sensorSnapshot.angleRadians,
        sensorSnapshot.angleDegrees,
        sensorSnapshot.cartVelocityMmPerSec,
        sensorSnapshot.angleVelocityRadPerSec,
        static_cast<unsigned long long>(sensorSnapshot.timestampUs),
        gains[0],
        gains[1],
        gains[2],
        gains[3]);
}

void IRAM_ATTR on_step_timer()
{
    portENTER_CRITICAL_ISR(&g_motionMux);

    if (!g_motionActive) {
        step_signal_inactive_isr();
        g_stepSignalActive = false;
        g_phaseTicksRemaining = 0U;
        portEXIT_CRITICAL_ISR(&g_motionMux);
        return;
    }

    if (g_phaseTicksRemaining > 0U) {
        --g_phaseTicksRemaining;
        portEXIT_CRITICAL_ISR(&g_motionMux);
        return;
    }

    if (g_stepSignalActive) {
        step_signal_inactive_isr();
        g_stepSignalActive = false;
        g_phaseTicksRemaining = g_inactiveTicks - 1U;
        portEXIT_CRITICAL_ISR(&g_motionMux);
        return;
    }

    step_signal_active_isr();
    g_stepSignalActive = true;

    if (g_directionClockwise) {
        ++g_positionSteps;
    } else {
        --g_positionSteps;
    }

    g_phaseTicksRemaining = kPulseActiveTicks - 1U;
    portEXIT_CRITICAL_ISR(&g_motionMux);
}

void stepper_init()
{
    configure_open_drain_output(kStepPin);
    configure_open_drain_output(kDirAltPin);
    configure_open_drain_output(kEnablePin);
    configure_input_pullup(kHomeSwitchPin);

    // Open-drain logic for the TB6600 common-anode wiring:
    // level 0 sinks current through the optocoupler input and activates the signal,
    // level 1 releases the pin so the external pull-up makes it inactive.
    signal_inactive(kStepPin);
    release_direction_signal();
    set_driver_enabled(false);
}

void stepper_timer_init()
{
    g_stepTimer = timerBegin(0, 80, true);
    timerAttachInterrupt(g_stepTimer, &on_step_timer, true);
    timerAlarmWrite(g_stepTimer, kTimerTickUs, true);
    timerAlarmDisable(g_stepTimer);
}

void stepper_set_direction(bool clockwise)
{
    const int directionLevel = clockwise ? kDirLevelClockwise : (1 - kDirLevelClockwise);
    if (directionLevel == 0) {
        assert_direction_signal();
    } else {
        release_direction_signal();
    }

    portENTER_CRITICAL(&g_motionMux);
    g_directionClockwise = clockwise;
    portEXIT_CRITICAL(&g_motionMux);

    delayMicroseconds(kDirectionSetupUs);
}

void stop_motion()
{
    timerAlarmDisable(g_stepTimer);

    portENTER_CRITICAL(&g_motionMux);
    g_motionActive = false;
    g_activeStepRateHz = 0U;
    g_phaseTicksRemaining = 0U;
    g_stepSignalActive = false;
    portEXIT_CRITICAL(&g_motionMux);

    signal_inactive(kStepPin);
    set_driver_enabled(false);
}

void pause_motion_pulses()
{
    timerAlarmDisable(g_stepTimer);

    portENTER_CRITICAL(&g_motionMux);
    g_motionActive = false;
    g_activeStepRateHz = 0U;
    g_phaseTicksRemaining = 0U;
    g_stepSignalActive = false;
    portEXIT_CRITICAL(&g_motionMux);

    signal_inactive(kStepPin);
}

bool queue_signed_step_rate(int32_t requestedSignedStepRate)
{
    if (g_motionCommandQueue == nullptr) {
        return false;
    }

    request_motion_sequence_abort();
    set_controller_mode(ControllerMode::Manual);

    MotionCommand motionCommand = {
        MotionCommandKind::SetSpeed,
        requestedSignedStepRate,
    };
    return xQueueOverwrite(g_motionCommandQueue, &motionCommand) == pdPASS;
}

bool queue_motion_command(MotionCommandKind kind)
{
    if (g_motionCommandQueue == nullptr) {
        return false;
    }

    request_motion_sequence_abort();
    set_controller_mode(ControllerMode::Manual);

    MotionCommand motionCommand = {kind, 0};
    return xQueueOverwrite(g_motionCommandQueue, &motionCommand) == pdPASS;
}

void set_signed_step_rate(int32_t requestedSignedStepRate, bool enforceSafety = true, bool emitSerial = true)
{
    const int32_t requestedClampedSignedStepRate = clamp_signed_step_rate(requestedSignedStepRate);
    const bool clampedToMaxRate =
        abs_i32(requestedSignedStepRate) > static_cast<int32_t>(kMaxStepRateHz);
    const int32_t appliedSignedStepRate =
        enforceSafety ? apply_motion_safety(requestedClampedSignedStepRate) : requestedClampedSignedStepRate;
    const uint32_t magnitude = magnitude_from_signed_rate(appliedSignedStepRate);
    bool driverWasEnabled = false;
    const int32_t currentSignedStepRate = current_signed_step_rate();

    portENTER_CRITICAL(&g_motionMux);
    g_commandedSignedStepRate = appliedSignedStepRate;
    driverWasEnabled = g_driverEnabled;
    portEXIT_CRITICAL(&g_motionMux);

    if (appliedSignedStepRate == currentSignedStepRate) {
        return;
    }

    if (emitSerial && enforceSafety && (appliedSignedStepRate != requestedClampedSignedStepRate)) {
        Serial.printf(
            "safety override requested=%+ld applied=%+ld\n",
            static_cast<long>(requestedClampedSignedStepRate),
            static_cast<long>(appliedSignedStepRate));
    }

    if (appliedSignedStepRate == 0) {
        stop_motion();
        if (emitSerial) {
            Serial.printf(
                "speed=%+ld steps/s stopped (pulses off; driver may still hold without ENA)\n",
                static_cast<long>(appliedSignedStepRate));
        }
        return;
    }

    const bool clockwise = (appliedSignedStepRate > 0);
    const uint32_t periodTicks = period_ticks_from_rate(magnitude);

    pause_motion_pulses();
    stepper_set_direction(clockwise);

    if (!driverWasEnabled) {
        set_driver_enabled(true);
    }

    portENTER_CRITICAL(&g_motionMux);
    g_activeStepRateHz = magnitude;
    g_phaseTicksRemaining = 0U;
    g_inactiveTicks = periodTicks - kPulseActiveTicks;
    g_stepSignalActive = false;
    g_motionActive = true;
    portEXIT_CRITICAL(&g_motionMux);

    timerWrite(g_stepTimer, 0);
    timerAlarmEnable(g_stepTimer);

    if (emitSerial) {
        if (clampedToMaxRate) {
            Serial.printf(
                "speed=%+ld steps/s (clamped to max, %.2f rpm)\n",
                static_cast<long>(appliedSignedStepRate),
                speed_rpm_from_signed_rate(appliedSignedStepRate));
        } else {
            Serial.printf(
                "speed=%+ld steps/s (%.2f rpm)\n",
                static_cast<long>(appliedSignedStepRate),
                speed_rpm_from_signed_rate(appliedSignedStepRate));
        }
    }
}

bool move_to_cart_position_steps(int32_t targetCartSteps, uint32_t stepRateHz)
{
    const int32_t currentCartSteps = cart_position_steps();
    if (targetCartSteps == currentCartSteps) {
        stop_motion();
        return true;
    }

    const bool movingAwayFromHome = targetCartSteps > currentCartSteps;
    const int32_t signedStepRate = movingAwayFromHome
                                       ? away_from_home_signed_step_rate(stepRateHz)
                                       : home_signed_step_rate(stepRateHz);

    set_signed_step_rate(signedStepRate, false);

    for (;;) {
        if (motion_sequence_abort_requested()) {
            stop_motion();
            Serial.println("Motion sequence aborted");
            return false;
        }

        if (!movingAwayFromHome && read_home_switch_active()) {
            set_raw_position_steps(0);
            if (targetCartSteps <= 0) {
                stop_motion();
                return true;
            }
        }

        const int32_t nextCartSteps = cart_position_steps();
        if ((movingAwayFromHome && (nextCartSteps >= targetCartSteps)) ||
            (!movingAwayFromHome && (nextCartSteps <= targetCartSteps))) {
            stop_motion();
            return true;
        }

        vTaskDelay(kMotionPollTicks);
    }
}

bool run_centering_sequence()
{
    bool axisHomed = false;
    int32_t axisCenterSteps = 0;

    portENTER_CRITICAL(&g_motionMux);
    axisHomed = g_axisHomed;
    axisCenterSteps = g_axisCenterSteps;
    portEXIT_CRITICAL(&g_motionMux);

    if (!axisHomed) {
        Serial.println("Center move unavailable: axis is not homed yet");
        return false;
    }

    clear_motion_sequence_abort();
    Serial.printf(
        "Centering to %ld steps (%.2f mm)\n",
        static_cast<long>(axisCenterSteps),
        steps_to_mm(axisCenterSteps));
    return move_to_cart_position_steps(axisCenterSteps, kMaxStepRateHz);
}

bool run_home_and_center_sequence()
{
    if (read_home_switch_active()) {
        Serial.println("Home switch already active; move the cart off the switch before HOME");
        return false;
    }

    clear_motion_sequence_abort();
    clear_axis_calibration();
    set_homing_in_progress(true);
    set_raw_position_steps(0);

    Serial.println("Seeking home switch and measuring full travel...");
    set_signed_step_rate(home_signed_step_rate(kMaxStepRateHz), false);

    while (!read_home_switch_active()) {
        if (motion_sequence_abort_requested()) {
            stop_motion();
            set_homing_in_progress(false);
            Serial.println("Home sequence aborted");
            return false;
        }

        if (abs_i32(raw_position_steps()) >= kCalibrationMaxTravelSteps) {
            stop_motion();
            clear_axis_calibration();
            set_homing_in_progress(false);
            Serial.println("Home sequence failed: exceeded maximum calibration travel");
            return false;
        }

        vTaskDelay(kMotionPollTicks);
    }

    stop_motion();
    vTaskDelay(kSwitchDebounceTicks);

    const int32_t travelSteps = abs_i32(raw_position_steps());
    if (travelSteps <= 0) {
        clear_axis_calibration();
        set_homing_in_progress(false);
        Serial.println("Home sequence failed: measured zero travel");
        return false;
    }

    set_raw_position_steps(0);
    update_axis_calibration(travelSteps);
    reset_sensor_sample_state();

    int32_t axisCenterSteps = 0;
    int32_t softLimitMinSteps = 0;
    int32_t softLimitMaxSteps = 0;

    portENTER_CRITICAL(&g_motionMux);
    axisCenterSteps = g_axisCenterSteps;
    softLimitMinSteps = g_softLimitMinSteps;
    softLimitMaxSteps = g_softLimitMaxSteps;
    portEXIT_CRITICAL(&g_motionMux);

    Serial.printf(
        "Home found: travel=%ld steps (%.2f mm), center=%ld steps, soft=[%ld,%ld]\n",
        static_cast<long>(travelSteps),
        steps_to_mm(travelSteps),
        static_cast<long>(axisCenterSteps),
        static_cast<long>(softLimitMinSteps),
        static_cast<long>(softLimitMaxSteps));

    const bool centered = move_to_cart_position_steps(axisCenterSteps, kMaxStepRateHz);
    set_homing_in_progress(false);

    if (centered) {
        Serial.println("Axis centered and ready");
    }

    return centered;
}

void handle_command(char* rawCommand)
{
    char* command = trim_in_place(rawCommand);
    long requestedSignedStepRate = 0L;
    float positionGain = 0.0f;
    float velocityGain = 0.0f;
    float angleGain = 0.0f;
    float angleRateGain = 0.0f;

    if (*command == '\0') {
        return;
    }

    if (strcasecmp(command, "HELP") == 0) {
        print_help();
        return;
    }

    if (strcasecmp(command, "STATUS") == 0) {
        (void)sample_sensor_snapshot(true);
        print_status();
        return;
    }

    if (strcasecmp(command, "SENSOR") == 0) {
        print_sensor_diagnostics();
        return;
    }

    if ((strcasecmp(command, "RESTANGLE") == 0) || (strcasecmp(command, "ANGLEZERO") == 0) ||
        (strcasecmp(command, "ZEROANGLE") == 0)) {
        (void)capture_rest_angle_reference();
        return;
    }

    if (strcasecmp(command, "GAINS") == 0) {
        print_teacher_gains();
        return;
    }

    if (strcasecmp(command, "TEACHER ON") == 0) {
        (void)enable_teacher_mode();
        return;
    }

    if (strcasecmp(command, "TEACHER OFF") == 0) {
        disable_teacher_mode(true);
        return;
    }

    if (strcasecmp(command, "LOG ON") == 0) {
        set_telemetry_enabled(true);
        Serial.printf("Telemetry enabled at %lu Hz\n", static_cast<unsigned long>(kTelemetryRateHz));
        return;
    }

    if (strcasecmp(command, "LOG OFF") == 0) {
        set_telemetry_enabled(false);
        Serial.println("Telemetry disabled");
        return;
    }

    if (sscanf(command, "SETK %f %f %f %f", &positionGain, &velocityGain, &angleGain, &angleRateGain) == 4) {
        set_teacher_lqr_gains(positionGain, velocityGain, angleGain, angleRateGain);
        print_teacher_gains();
        return;
    }

    if (strcasecmp(command, "HOME") == 0) {
        if (!queue_motion_command(MotionCommandKind::HomeAndCenter)) {
            Serial.println("Unable to queue HOME command");
        }
        return;
    }

    if (strcasecmp(command, "CENTER") == 0) {
        if (!queue_motion_command(MotionCommandKind::MoveToCenter)) {
            Serial.println("Unable to queue CENTER command");
        }
        return;
    }

    if (strcasecmp(command, "STOP") == 0) {
        if (!queue_motion_command(MotionCommandKind::Stop)) {
            Serial.println("Unable to queue STOP command");
        }
        return;
    }

    if (sscanf(command, "MOVE %ld", &requestedSignedStepRate) == 1) {
        if (!queue_signed_step_rate(static_cast<int32_t>(requestedSignedStepRate))) {
            Serial.println("Unable to queue MOVE command");
        }
        return;
    }

    if (sscanf(command, "SPEED %ld", &requestedSignedStepRate) == 1) {
        if (!queue_signed_step_rate(static_cast<int32_t>(requestedSignedStepRate))) {
            Serial.println("Unable to queue SPEED command");
        }
        return;
    }

    if (sscanf(command, "%ld", &requestedSignedStepRate) == 1) {
        if (!queue_signed_step_rate(static_cast<int32_t>(requestedSignedStepRate))) {
            Serial.println("Unable to queue speed command");
        }
        return;
    }

    Serial.printf("Unknown command: %s\n", command);
    print_help();
}

void poll_serial_commands()
{
    while (Serial.available() > 0) {
        const char incoming = static_cast<char>(Serial.read());

        if ((incoming == '\n') || (incoming == '\r')) {
            if (g_commandOverflow) {
                Serial.println("Command too long");
            } else if (g_commandLength > 0U) {
                g_commandBuffer[g_commandLength] = '\0';
                handle_command(g_commandBuffer);
            }

            g_commandLength = 0U;
            g_commandOverflow = false;
        } else if (!g_commandOverflow) {
            if (g_commandLength < (kCommandBufferSize - 1U)) {
                g_commandBuffer[g_commandLength++] = incoming;
            } else {
                g_commandOverflow = true;
            }
        }
    }
}

void motion_control_task(void* /*parameter*/)
{
    MotionCommand motionCommand = {MotionCommandKind::Stop, 0};

    for (;;) {
        if (xQueueReceive(g_motionCommandQueue, &motionCommand, pdMS_TO_TICKS(1)) == pdPASS) {
            switch (motionCommand.kind) {
                case MotionCommandKind::SetSpeed:
                    clear_motion_sequence_abort();
                    set_signed_step_rate(motionCommand.signedStepRate, true);
                    break;
                case MotionCommandKind::Stop:
                    clear_motion_sequence_abort();
                    set_signed_step_rate(0, false);
                    break;
                case MotionCommandKind::HomeAndCenter:
                    (void)run_home_and_center_sequence();
                    break;
                case MotionCommandKind::MoveToCenter:
                    (void)run_centering_sequence();
                    break;
            }
        } else {
            enforce_runtime_motion_safety();
        }
    }
}

void teacher_control_task(void* /*parameter*/)
{
    TickType_t lastWakeTime = xTaskGetTickCount();

    for (;;) {
        const ControllerMode controllerMode = current_controller_mode();
        const bool teacherActive = controllerMode == ControllerMode::TeacherLqr;
        const bool telemetryActive = telemetry_enabled();

        if (!(teacherActive || telemetryActive)) {
            vTaskDelay(pdMS_TO_TICKS(20));
            lastWakeTime = xTaskGetTickCount();
            continue;
        }

        const bool sampleValid = sample_sensor_snapshot(true);
        if (teacherActive) {
            if (!sampleValid) {
                set_controller_mode(ControllerMode::Manual);
                set_signed_step_rate(0, false, true);
                Serial.println("Teacher LQR disabled: AS5600 read failed");
            } else {
                const SensorSnapshot snapshot = latest_sensor_snapshot();
                if (!snapshot.axisHomed) {
                    set_controller_mode(ControllerMode::Manual);
                    set_signed_step_rate(0, false, true);
                    Serial.println("Teacher LQR disabled: cart is not homed");
                } else if (!snapshot.angleZeroValid) {
                    set_controller_mode(ControllerMode::Manual);
                    set_signed_step_rate(0, false, true);
                    Serial.println("Teacher LQR disabled: angle reference is not zeroed");
                } else {
                    const int32_t requestedStepRate = compute_teacher_command_steps_per_second(snapshot);
                    set_signed_step_rate(requestedStepRate, true, false);
                }
            }
        }

        vTaskDelayUntil(&lastWakeTime, kTeacherControlPeriodTicks);
    }
}

void telemetry_stream_task(void* /*parameter*/)
{
    TickType_t lastWakeTime = xTaskGetTickCount();

    for (;;) {
        if (!telemetry_enabled()) {
            g_telemetryHeaderPrinted = false;
            vTaskDelay(pdMS_TO_TICKS(20));
            lastWakeTime = xTaskGetTickCount();
            continue;
        }

        const SensorSnapshot snapshot = latest_sensor_snapshot();
        if (snapshot.sampleValid && snapshot.sensorOnline) {
            if (!g_telemetryHeaderPrinted) {
                Serial.println(
                    "DATA_HEADER,timestamp_us,cart_home_steps,cart_center_steps,cart_home_mm,cart_center_mm,cart_vel_mm_s,angle_raw_counts,angle_deg,angle_rad,angle_vel_rad_s,command_steps_s,mode,homed,angle_zeroed");
                g_telemetryHeaderPrinted = true;
            }

            Serial.printf(
                "DATA,%llu,%ld,%ld,%.3f,%.3f,%.3f,%u,%.3f,%.6f,%.6f,%ld,%s,%s,%s\n",
                static_cast<unsigned long long>(snapshot.timestampUs),
                static_cast<long>(snapshot.cartHomeSteps),
                static_cast<long>(snapshot.cartCenteredSteps),
                snapshot.cartHomeMm,
                snapshot.cartCenteredMm,
                snapshot.cartVelocityMmPerSec,
                static_cast<unsigned int>(snapshot.angleRawCounts),
                snapshot.angleDegrees,
                snapshot.angleRadians,
                snapshot.angleVelocityRadPerSec,
                static_cast<long>(snapshot.commandStepsPerSecond),
                controller_mode_name(snapshot.controllerMode),
                snapshot.axisHomed ? "yes" : "no",
                snapshot.angleZeroValid ? "yes" : "no");
        }

        vTaskDelayUntil(&lastWakeTime, kTelemetryPeriodTicks);
    }
}

void serial_listener_task(void* /*parameter*/)
{
    for (;;) {
        poll_serial_commands();
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

}  // namespace

void setup()
{
    const esp_reset_reason_t resetReason = esp_reset_reason();

    Serial.begin(kConsoleBaudRate);
    stepper_init();
    stepper_timer_init();
    angle_sensor_init();
    clear_axis_calibration();
    reset_sensor_sample_state();
    g_motionCommandQueue = xQueueCreate(kMotionCommandQueueLength, sizeof(MotionCommand));

    if (g_motionCommandQueue == nullptr) {
        Serial.println("Failed to create motion command queue");
        return;
    }
    Serial.println("TB6600 timer speed controller ready");
    Serial.println(
        "GPIO25=PUL- GPIO26=DIR- GPIO33=DIR-ALT GPIO27=ENA- GPIO32=HOME-SW(NC) GPIO21=SDA GPIO22=SCL");
    Serial.printf(
        "motion_core=%ld serial_core=%ld teacher_core=%ld telemetry_core=%ld baud=%lu\n",
        static_cast<long>(kMotionTaskCore),
        static_cast<long>(kSerialTaskCore),
        static_cast<long>(kTeacherTaskCore),
        static_cast<long>(kTelemetryTaskCore),
        static_cast<unsigned long>(kConsoleBaudRate));
    Serial.printf(
        "reset=%s (%d) type HELP for commands\n",
        reset_reason_name(resetReason),
        static_cast<int>(resetReason));

    (void)xTaskCreatePinnedToCore(
        motion_control_task,
        "motion_control",
        kTaskStackBytes,
        nullptr,
        2,
        &g_motionTaskHandle,
        kMotionTaskCore);
    (void)xTaskCreatePinnedToCore(
        serial_listener_task,
        "serial_listener",
        kTaskStackBytes,
        nullptr,
        1,
        &g_serialTaskHandle,
        kSerialTaskCore);
    (void)xTaskCreatePinnedToCore(
        teacher_control_task,
        "teacher_control",
        kTaskStackBytes,
        nullptr,
        1,
        &g_teacherTaskHandle,
        kTeacherTaskCore);
    (void)xTaskCreatePinnedToCore(
        telemetry_stream_task,
        "telemetry_stream",
        kTaskStackBytes,
        nullptr,
        1,
        &g_telemetryTaskHandle,
        kTelemetryTaskCore);
}

void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}

