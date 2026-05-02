// TB6600 common-anode control from an ESP32 using open-drain GPIOs.
// PUL+ and DIR+ are pulled up externally; the ESP32 only sinks current on PUL-/DIR-.

#include <Arduino.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "driver/gpio.h"
#include "esp32-hal-timer.h"
#include "soc/gpio_struct.h"

namespace {

constexpr gpio_num_t kStepPin = GPIO_NUM_25;
constexpr gpio_num_t kDirPin = GPIO_NUM_26;
constexpr gpio_num_t kDirAltPin = GPIO_NUM_33;
constexpr gpio_num_t kEnablePin = GPIO_NUM_27;

constexpr uint32_t kStepsPerRevolution = 3200;
constexpr uint32_t kPulseActiveUs = 20;
constexpr uint32_t kPulseInactiveMinUs = 20;
constexpr uint32_t kDirectionSetupUs = 2000;
constexpr uint32_t kTimerTickUs = 20;

constexpr int32_t kDefaultSignedStepRate = 200;
constexpr uint32_t kMinStepRateHz = 10;
constexpr uint32_t kMaxStepRateHz = 1000;

constexpr size_t kCommandBufferSize = 64;

constexpr uint32_t kPulseActiveTicks = (kPulseActiveUs + kTimerTickUs - 1U) / kTimerTickUs;
constexpr uint32_t kStepPinMask = (1UL << static_cast<uint32_t>(kStepPin));

// If your mechanical setup spins the opposite direction, change this to 1.
constexpr int kDirLevelClockwise = 0;
// For many TB6600 modules with common-anode wiring, ENA- LOW disables current
// and ENA- HIGH/released enables the outputs. If yours is opposite, swap these.
constexpr int kEnableLevelMotorEnabled = 1;
constexpr int kEnableLevelMotorDisabled = 0;

char g_commandBuffer[kCommandBufferSize] = {};
size_t g_commandLength = 0U;
bool g_commandOverflow = false;

hw_timer_t* g_stepTimer = nullptr;
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

uint32_t magnitude_from_signed_rate(long signedStepRate)
{
    if (signedStepRate == 0L) {
        return 0U;
    }

    const long magnitude = (signedStepRate < 0L) ? -signedStepRate : signedStepRate;
    return clamp_u32(static_cast<uint32_t>(magnitude), kMinStepRateHz, kMaxStepRateHz);
}

int32_t signed_rate_from_direction(bool clockwise, uint32_t magnitude)
{
    return clockwise ? static_cast<int32_t>(magnitude) : -static_cast<int32_t>(magnitude);
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
    bool active = false;
    bool clockwise = true;
    bool driverEnabled = false;
    int dirPinLevel = 0;
    int dirAltPinLevel = 0;
    int enablePinLevel = 0;

    portENTER_CRITICAL(&g_motionMux);
    commandedSignedStepRate = g_commandedSignedStepRate;
    activeSignedStepRate = g_motionActive
                               ? signed_rate_from_direction(g_directionClockwise, g_activeStepRateHz)
                               : 0;
    positionSteps = g_positionSteps;
    active = g_motionActive;
    clockwise = g_directionClockwise;
    driverEnabled = g_driverEnabled;
    portEXIT_CRITICAL(&g_motionMux);

    dirPinLevel = gpio_get_level(kDirPin);
    dirAltPinLevel = gpio_get_level(kDirAltPin);
    enablePinLevel = gpio_get_level(kEnablePin);

    Serial.printf(
        "position=%ld active=%s ena=%s enaPin=%s dir=%s dir26=%s dir33=%s commanded=%+ld steps/s current=%+ld steps/s\n",
        static_cast<long>(positionSteps),
        active ? "yes" : "no",
        driverEnabled ? "ON" : "OFF",
        (enablePinLevel == 0) ? "LOW" : "HIGH",
        clockwise ? "CW" : "CCW",
        (dirPinLevel == 0) ? "LOW" : "HIGH",
        (dirAltPinLevel == 0) ? "LOW" : "HIGH",
        static_cast<long>(commandedSignedStepRate),
        static_cast<long>(activeSignedStepRate));
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

void set_signed_step_rate(int32_t requestedSignedStepRate)
{
    const uint32_t magnitude = magnitude_from_signed_rate(requestedSignedStepRate);
    const int32_t clampedSignedStepRate = (requestedSignedStepRate == 0)
                                              ? 0
                                              : signed_rate_from_direction(requestedSignedStepRate > 0, magnitude);
    bool driverWasEnabled = false;

    portENTER_CRITICAL(&g_motionMux);
    g_commandedSignedStepRate = clampedSignedStepRate;
    driverWasEnabled = g_driverEnabled;
    portEXIT_CRITICAL(&g_motionMux);

    if (clampedSignedStepRate == 0) {
        stop_motion();
        Serial.printf(
            "speed=%+ld steps/s stopped (pulses off; driver may still hold without ENA)\n",
            static_cast<long>(clampedSignedStepRate));
        return;
    }

    const bool clockwise = (clampedSignedStepRate > 0);
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

    Serial.printf(
        "speed=%+ld steps/s (%.2f rpm)\n",
        static_cast<long>(clampedSignedStepRate),
        speed_rpm_from_signed_rate(clampedSignedStepRate));
}

void handle_command(char* rawCommand)
{
    char* command = trim_in_place(rawCommand);
    long requestedSignedStepRate = 0L;

    if (*command == '\0') {
        return;
    }

    if (strcasecmp(command, "HELP") == 0) {
        print_help();
        return;
    }

    if (strcasecmp(command, "STATUS") == 0) {
        print_status();
        return;
    }

    if (strcasecmp(command, "STOP") == 0) {
        set_signed_step_rate(0);
        return;
    }

    if (sscanf(command, "MOVE %ld", &requestedSignedStepRate) == 1) {
        set_signed_step_rate(static_cast<int32_t>(requestedSignedStepRate));
        return;
    }

    if (sscanf(command, "SPEED %ld", &requestedSignedStepRate) == 1) {
        set_signed_step_rate(static_cast<int32_t>(requestedSignedStepRate));
        return;
    }

    if (sscanf(command, "%ld", &requestedSignedStepRate) == 1) {
        set_signed_step_rate(static_cast<int32_t>(requestedSignedStepRate));
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

}  // namespace

void setup()
{
    Serial.begin(9600);
    stepper_init();
    stepper_timer_init();

    Serial.println("TB6600 timer speed controller ready");
    Serial.println("GPIO25=PUL- GPIO26=DIR- GPIO33=DIR-ALT GPIO27=ENA-");
    print_help();
}

void loop()
{
    poll_serial_commands();
    vTaskDelay(pdMS_TO_TICKS(1));
}

