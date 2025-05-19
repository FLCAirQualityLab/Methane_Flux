#include "Particle.h"
#include <string>
#include <Wire.h>
#include <sstream>
#include "Adafruit_ADS1X15.h"
#include "Adafruit_BME680.h"

TCPClient client;
SYSTEM_MODE(AUTOMATIC);
SerialLogHandler logHandler(LOG_LEVEL_INFO);

// Flowrate address and commands
#define SFM4300_ADDR                         0x2A
#define CMD_START_CONTINUOUS_MEASUREMENT_AIR 0x3608  // Air mode :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
#define CMD_STOP_CONTINUOUS_MEASUREMENT      0x3FF9  // Stop measurement :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

// Scale/offset for Air (from Datasheet Table 15)
const float FLOW_SCALE     = 2500.0;   // slm⁻¹
const int16_t FLOW_OFFSET  = -28672;   // raw units
const float TEMP_SCALE     = 200.0;    // °C⁻¹
const int16_t TEMP_OFFSET  =    0;     // raw units
float flowSlm       = NAN;
float tempFlowC     = NAN;
uint16_t statusFlow = 0;

// Initialize main I2C lines
Adafruit_ADS1115 ads_MOX(0x48);
Adafruit_ADS1115 ads_EC(0x49);
Adafruit_BME680 bme680;

// Initialize BO I2C lines
Adafruit_ADS1115 ads_MOX_BO(0x4a);
Adafruit_ADS1115 ads_EC_BO(0x4b);
Adafruit_BME680 bme680_bo;

int sampletime = 10000;
int sample_cnt = 0;
bool active = true;
bool format_uart = false;
String dataBuffer;

// Global buffers for accumulating SGX data
String sgxBuffer = "";
int SGX_ppm = 0;
bool SGX_FLAG = false;


// Cloud functions ************************************************************************************
// Cloud function to activate/deactivate data collection
bool activate(String state) {
    if (state == "T") {
        active = true;
        return true;
    } else if (state == "F") {
        active = false;
        return false;
    } else {
        return false;
    }
}

// Turn on UART formatting
bool formatUART(String state) {
    if (state == "T") {
        format_uart = true;
        return true;
    } else if (state == "F") {
        format_uart = false;
        return false;
    } else {
        return false;
    }
}

//------------------- SFM Functions ------------------
// CRC‑8 (poly 0x31, init 0xFF) for two bytes :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
uint8_t crc8(const uint8_t data[2]) {
    uint8_t crc = 0xFF;
    for (uint8_t i = 0; i < 2; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            crc = (crc & 0x80) ? (crc << 1) ^ 0x31 : (crc << 1);
        }
    }
    return crc;
}

void sfmWriteCmd(uint16_t cmd) {
    Wire.beginTransmission(SFM4300_ADDR);
    Wire.write(cmd >> 8);
    Wire.write(cmd & 0xFF);
    Wire.endTransmission();
}

// Perform one flow/temperature read
bool readFlow() {
    Wire.requestFrom(SFM4300_ADDR, 9);
    if (Wire.available() < 9) {
        Serial.println("SFM4300: not enough bytes");
        return false;
    }
    uint8_t buf[9];
    for (uint8_t i = 0; i < 9; i++) buf[i] = Wire.read();

    if (crc8(buf + 0) != buf[2] ||
        crc8(buf + 3) != buf[5] ||
        crc8(buf + 6) != buf[8]) {
        Serial.println("SFM4300: CRC failed");
        return false;
    }

    // parse
    int16_t rawFlow = (buf[0] << 8) | buf[1];
    int16_t rawTemp = (buf[3] << 8) | buf[4];
    statusFlow      = (buf[6] << 8) | buf[7];

    // convert
    flowSlm   = (rawFlow - FLOW_OFFSET) / FLOW_SCALE;
    tempFlowC = (rawTemp - TEMP_OFFSET) / TEMP_SCALE;

    return true;
}

// ------------------ SGX Functions ------------------
// Convert an 8-character hex string to int.
int HEX_TO_INT(String c) {
    int x;
    char cc[9]; // 8 chars + null terminator
    c.toCharArray(cc, 9);
    std::stringstream ss;
    ss << std::hex << cc;
    ss >> x;
    return x;
}

// Flush any leftover data from Serial1
void flushSerial1() {
    int cnt = 0;
    while (Serial1.available()) {
        Serial1.read();
        cnt++;
        if (cnt > 100) {
            break; // Prevent infinite loop
        }
    }
}

// Send a command to the SGX sensor and wait for an ACK response
bool sendCommand(String cmd) {
    flushSerial1();
    Serial1.print(cmd);
    Serial.print("Sent command: ");
    Serial.println(cmd);
    
    unsigned long startTime = millis();
    String response = "";
    while (millis() - startTime < 1000) { // 1-second timeout
        if (Serial1.available() > 0) {
            char c = Serial1.read();
            response += c;
            if (response.length() >= 8) {
                break;
            }
        }
    }
    
    response.toLowerCase();
    Serial.print("Raw response: ");
    Serial.println(response);
    
    return (response.indexOf("5b414b5d") != -1);
}

// Accumulate available data from Serial1 and remove newline characters
void accumulateSgxData() {
    while (Serial1.available() > 0) {
        char c = Serial1.read();
        sgxBuffer += c;
    }
    sgxBuffer.replace("\n", "");
    sgxBuffer.replace("\r", "");
    // If the buffer grows too long, trim it.
    if (sgxBuffer.length() > 200) {
        int lastStart = sgxBuffer.lastIndexOf("0000005b");
        if (lastStart != -1) {
            sgxBuffer = sgxBuffer.substring(lastStart);
        } else {
            sgxBuffer = "";
        }
    }
}

// Extract 8 hex characters that come immediately after the marker "0000005b".
// Once a complete reading is extracted, clear the sgxBuffer.
String extractSgxReading() {
    int startIdx = sgxBuffer.indexOf("0000005b");
    // We need at least the marker (8 chars) plus another 8 chars for the reading.
    if (startIdx != -1 && sgxBuffer.length() >= (startIdx + 16)) {
        String reading = sgxBuffer.substring(startIdx + 8, startIdx + 16);
        // Clear the buffer after taking the reading.
        sgxBuffer = "";
        return reading;
    }
    return "";
}



// ------------------ Setup ------------------
void setup() {
    delay(2000);
    Serial.begin(9600);
    delay(2000);
    
    // Initialize UART for sensor communication (38400 baud, 8N2)
    Serial1.begin(38400, SERIAL_8N2);
    delay(2000);
    Serial.println("Starting sensor communication...");
    delay(2000);
    
    // Enter CONFIGURATION Mode with [C]
    if (sendCommand("[C]")) {
        Serial.println("Entered CONFIGURATION Mode successfully.");
        String settings = "";
        flushSerial1();
        sendCommand("[I]");
        unsigned long timeout = millis() + 1000;
        while (millis() < timeout) {
            while (Serial1.available()) {
                settings += Serial1.readString();
            }
        }
        Serial.println("Sensor settings: \n" + settings);
    } else {
        Serial.println("Failed to enter CONFIGURATION Mode.");
    }
    delay(1000);
    
    // Enter ENGINEERING Mode with [B]
    if (sendCommand("[B]")) {
        Serial.println("Entered ENGINEERING Mode successfully.");
    } else {
        Serial.println("Failed to enter ENGINEERING Mode.");
    }
    
    // Initialize main BME680
    if (!bme680.begin(0x77)) {
        Serial.println(F("Could not find a valid BME680 sensor, check wiring!"));
        delay(2000);
    }
    bme680.setTemperatureOversampling(BME680_OS_8X);
    bme680.setHumidityOversampling(BME680_OS_2X);
    bme680.setPressureOversampling(BME680_OS_4X);
    bme680.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme680.setGasHeater(320, 150);
    
    // Initialize BO BME680
    if (!bme680_bo.begin(0x76)) {
        Serial.println(F("Could not find a valid BO BME680 sensor, check wiring!"));
        delay(2000);
    }
    bme680_bo.setTemperatureOversampling(BME680_OS_8X);
    bme680_bo.setHumidityOversampling(BME680_OS_2X);
    bme680_bo.setPressureOversampling(BME680_OS_4X);
    bme680_bo.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme680_bo.setGasHeater(320, 150);
    
    // Initialize BO ADCs
    ads_MOX.begin();
    ads_MOX.setGain(GAIN_ONE);
    ads_EC.begin();
    ads_EC.setGain(GAIN_ONE);
    
    // Initialize BO ADCs for BO board
    ads_MOX_BO.begin();
    ads_MOX_BO.setGain(GAIN_ONE);
    ads_EC_BO.begin();
    ads_EC_BO.setGain(GAIN_ONE);
    
    // Initialize SFM4300
    Wire.begin();
    sfmWriteCmd(CMD_START_CONTINUOUS_MEASUREMENT_AIR);
    if(!readFlow()){Serial.println(F("SFM4300 initialization failed."));}
    
    // Register cloud functions
    Particle.function("activate", activate);
    Particle.function("formatUART", formatUART);
    
    //Serial.println("Waiting for sensor warm-up (45 seconds)...");
    //delay(45000);
}

// Main Loop ****************************************************************************************
void loop() {
    readFlow();

    // Read ADC values from main MOX sensors
    double multiplier = 0.1875F; // mV per bit for ADS1115
    short adc_MOX_0 = ads_MOX.readADC_SingleEnded(0);
    short adc_MOX_1 = ads_MOX.readADC_SingleEnded(1);
    short adc_MOX_2 = ads_MOX.readADC_SingleEnded(2);
    double av_MOX_0 = adc_MOX_0 * multiplier;
    double av_MOX_1 = adc_MOX_1 * multiplier;
    double av_MOX_2 = adc_MOX_2 * multiplier;

    // Read ADC values from main EC sensor and SGX analog
    short adc_EC_0 = ads_EC.readADC_SingleEnded(0);
    short adc_EC_1 = ads_EC.readADC_SingleEnded(1);
    double av_EC_0 = adc_EC_0 * multiplier;
    double av_EC_1 = adc_EC_1 * multiplier;

    // Read ADC values from BO MOX sensors
    short adc_MOX_BO_0 = ads_MOX_BO.readADC_SingleEnded(0);
    short adc_MOX_BO_1 = ads_MOX_BO.readADC_SingleEnded(1);
    short adc_MOX_BO_2 = ads_MOX_BO.readADC_SingleEnded(2);
    double av_MOX_BO_0 = adc_MOX_BO_0 * multiplier;
    double av_MOX_BO_1 = adc_MOX_BO_1 * multiplier;
    double av_MOX_BO_2 = adc_MOX_BO_2 * multiplier;

    // Read ADC values from BO EC sensor and SGX analog
    short adc_EC_BO_0 = ads_EC_BO.readADC_SingleEnded(0);
    short adc_EC_BO_1 = ads_EC_BO.readADC_SingleEnded(1); 
    short adc_EC_BO_2 = ads_EC_BO.readADC_SingleEnded(2);
    double av_EC_BO_0 = adc_EC_BO_0 * multiplier;
    double av_EC_BO_1 = adc_EC_BO_1 * multiplier;
    double av_EC_BO_2 = adc_EC_BO_2 * multiplier;



    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Read BME680 values
    if (!bme680.performReading()) {
        Serial.println("Failed to perform main BME680 reading :(");
    }
    double temperatureInC = bme680.temperature;
    double relativeHumidity = bme680.humidity;
    double pressurepa = bme680.pressure;
    double gas_resistance = bme680.gas_resistance;

    // Read BO BME680 values
    if (!bme680_bo.performReading()) {
        Serial.println("Failed to perform BO BME680 reading :(");
    }
    double temperatureInC_bo = bme680_bo.temperature;
    double relativeHumidity_bo = bme680_bo.humidity;
    double pressurepa_bo = bme680_bo.pressure;
    double gas_resistance_bo = bme680_bo.gas_resistance;


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //SGX UART acquisition
    accumulateSgxData();
    String sgxReading = extractSgxReading();
    if (sgxReading.length() == 8) {
        Serial.println("SGX Reading (8 hex chars after marker): " + sgxReading);
        SGX_ppm = HEX_TO_INT(sgxReading);
        SGX_FLAG = true;
    } else {
        Serial.println("No complete SGX reading found.");
        SGX_FLAG = false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Convert values for logging (scaled by 100)
    int i_av_MOX_0 = (int)(av_MOX_0 * 100);
    int i_av_MOX_1 = (int)(av_MOX_1 * 100);
    int i_av_MOX_2 = (int)(av_MOX_2 * 100);
    int i_av_EC_0 = (int)(av_EC_0 * 100);
    int i_av_EC_1 = (int)(av_EC_1 * 100);

    int i_av_MOX_BO_0 = (int)(av_MOX_BO_0 * 100);
    int i_av_MOX_BO_1 = (int)(av_MOX_BO_1 * 100);
    int i_av_MOX_BO_2 = (int)(av_MOX_BO_2 * 100);
    int i_av_EC_BO_0 = (int)(av_EC_BO_0 * 100);
    int i_av_EC_BO_1 = (int)(av_EC_BO_1 * 100);

    if(!SGX_FLAG) {SGX_ppm = 404;} // Set SGX_ppm to 404 if no valid reading is found
    int i_av_EC_BO_2 = (int)(av_EC_BO_2 * 100);
    int i_SGX_ppm = (int)(SGX_ppm * 100);

    int i_temperatureInC = (int)(temperatureInC * 100);
    int i_pressurepa = (int)(pressurepa * 100);
    int i_relativeHumidity = (int)(relativeHumidity * 100);
    int i_gas_resistance = (int)(gas_resistance * 100);

    int i_temperatureInC_bo = (int)(temperatureInC_bo * 100);
    int i_pressurepa_bo = (int)(pressurepa_bo * 100);
    int i_relativeHumidity_bo = (int)(relativeHumidity_bo * 100);
    int i_gas_resistance_bo = (int)(gas_resistance_bo * 100);

    int i_flowSlm = (int)(flowSlm * 100);
    int i_tempFlowC = (int)(tempFlowC * 100);

    //Sample count starts at 1
    sample_cnt++;

    // Timestamp and format data
    int timestamp = (int)Time.now();
    if(format_uart){
        String timeStr = Time.format(timestamp, TIME_FORMAT_DEFAULT);
        String formattedData = String::format(
            "{"
            "\"time\":\"%s\",\n"
            "\"flowSlm\":%.2f,"
            "\"flowTempC\":%.2f,\n"
            "\"adc_MOX_0\":%.2f,"
            "\"adc_MOX_1\":%.2f,"
            "\"adc_MOX_2\":%.2f,"
            "\"adc_EC_0\":%.2f,"
            "\"adc_EC_1\":%.2f,\n"
            "\"adc_MOX_BO_0\":%.2f,"
            "\"adc_MOX_BO_1\":%.2f,"
            "\"adc_MOX_BO_2\":%.2f,"
            "\"adc_EC_BO_0\":%.2f,"
            "\"adc_EC_BO_1\":%.2f,\n"
            "\"adc_EC_BO_2\":%.2f,"
            "\"SGX_UART\":%i,\n"
            "\"temperatureC\":%.2f,"
            "\"pressurehPa\":%.2f,"
            "\"humidityPct\":%.2f,"
            "\"gasResistance\":%.2f,\n"
            "\"temperatureC_bo\":%.2f,"
            "\"pressurehPa_bo\":%.2f,"
            "\"humidityPct_bo\":%.2f,"
            "\"gasResistance_bo\":%.2f"
            "}",
            timeStr.c_str(), flowSlm, tempFlowC, av_MOX_0, av_MOX_1, av_MOX_2, av_EC_0, av_EC_1, 
            av_MOX_BO_0, av_MOX_BO_1, av_MOX_BO_2, av_EC_BO_0, av_EC_BO_1, av_EC_BO_2, SGX_ppm, 
            temperatureInC, pressurepa, relativeHumidity, gas_resistance, 
            temperatureInC_bo, pressurepa_bo, relativeHumidity_bo, gas_resistance_bo
        );
    }

    // Format cloud string
    String data = String::format(
        "%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i",
        timestamp, i_flowSlm, i_tempFlowC, i_av_MOX_0, i_av_MOX_1, i_av_MOX_2, i_av_EC_0, i_av_EC_1, 
        i_av_MOX_BO_0, i_av_MOX_BO_1, i_av_MOX_BO_2, i_av_EC_BO_0, i_av_EC_BO_1, 
        i_av_EC_BO_2, i_SGX_ppm, 
        i_temperatureInC, i_pressurepa, i_relativeHumidity, i_gas_resistance,
        i_temperatureInC_bo, i_pressurepa_bo, i_relativeHumidity_bo, i_gas_resistance_bo
    );

    // Print formatted sensor data and sample count
    if(format_uart) {Serial.println(formattedData + "," + String(sample_cnt) + "\n");}
    Serial.println(data + "," + String(sample_cnt) + "\n");

    // Buffer data for cloud publishing if needed
    if ((dataBuffer.length() + data.length() + 1) > 1024) {
        if (active) {
            Particle.publish("BoroniusData", dataBuffer, PRIVATE);
            Serial.println("Data sent to cloud (buffer full)");
        } else {
            Serial.println("Data collection inactive, discarding buffer");
        }
        dataBuffer = "";
    }
    dataBuffer += data + "\n";

    delay(sampletime);
}
