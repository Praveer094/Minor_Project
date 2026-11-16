// Infrastructure Forensics - ride logger v2
// Logs accelerometer + gyro at 100 Hz, ultrasonic distance at 10 Hz and GPS at 1 Hz to the SD card.
// Every row carries ESP32 millis + the latest GPS time and its age in ms, so
// exact time = GPS time + age. The video-sync script uses this.
//
// v2: the ultrasonic echo is measured by an interrupt, so the 100 Hz
//     accelerometer sampling never pauses while waiting for an echo.
//
// Onboard LED: blinking = logging, no GPS fix yet | solid = logging with GPS fix
// Error blinks (repeating): 2 = MPU6050 missing, 3 = SD card missing, 4 = cannot create file
//
// Ride routine: power on -> wait for solid LED -> start phone video ->
//               tap rig 3 times -> ride -> tap 3 times -> stop video -> wait 2 s -> unplug
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <TinyGPSPlus.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define GPS_RX   16
#define GPS_TX   17
#define TRIG_PIN 27
#define ECHO_PIN 26
#define SD_CS    5
#define LED_PIN  2

const uint32_t SAMPLE_US    = 10000;  // 100 Hz accelerometer sampling
const uint32_t PING_MS      = 100;    // 10 Hz ultrasonic
const uint32_t MAX_ECHO_US  = 25000;  // ~4.3 m; longer echo = no valid reading
const uint16_t FLUSH_EVERY  = 100;    // flush to card once per second

TinyGPSPlus gps;
Adafruit_MPU6050 mpu;
File logFile;

uint32_t nextSampleUs = 0, lastPingMs = 0;
float lastDistCm = -1;
uint16_t linesSinceFlush = 0;

// ---- ultrasonic echo, measured in the background ----
volatile uint32_t echoRiseUs = 0;
volatile uint32_t echoWidthUs = 0;
volatile bool echoReady = false;

void IRAM_ATTR onEcho() {
  if (digitalRead(ECHO_PIN)) {
    echoRiseUs = micros();                 // echo pulse started
  } else {
    echoWidthUs = micros() - echoRiseUs;   // echo pulse ended
    echoReady = true;
  }
}

// Collect the result of the previous ping, then send a new one.
void pingUltrasonic() {
  if (echoReady && echoWidthUs < MAX_ECHO_US) {
    lastDistCm = echoWidthUs * 0.0343f / 2.0f;
  } else {
    lastDistCm = -1;                       // no echo / out of range
  }
  echoReady = false;
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
}

void fail(int blinks) {
  while (true) {
    for (int i = 0; i < blinks; i++) {
      digitalWrite(LED_PIN, HIGH); delay(150);
      digitalWrite(LED_PIN, LOW);  delay(150);
    }
    delay(1000);
  }
}

String nextFileName() {
  char name[24];
  for (int i = 1; i < 1000; i++) {
    snprintf(name, sizeof(name), "/ride_%03d.csv", i);
    if (!SD.exists(name)) return String(name);
  }
  return String("/ride_overflow.csv");
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), onEcho, CHANGE);

  Serial2.setRxBufferSize(1024);           // GPS keeps arriving while the SD card writes
  Serial2.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  Wire.begin(21, 22);
  Wire.setClock(400000);                   // fast I2C: one MPU read takes ~0.5 ms

  if (!mpu.begin()) { Serial.println("MPU6050 not found"); fail(2); }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);

  if (!SD.begin(SD_CS)) { Serial.println("SD init failed"); fail(3); }
  String fname = nextFileName();
  logFile = SD.open(fname, FILE_WRITE);
  if (!logFile) { Serial.println("Cannot create log file"); fail(4); }

  logFile.println("millis,gps_valid,gps_date,gps_time_utc,gps_age_ms,lat,lng,speed_kmh,sats,"
                  "ax,ay,az,gx,gy,gz,dist_cm");
  logFile.flush();
  Serial.print("Logging to "); Serial.println(fname);
  nextSampleUs = micros();
}

void loop() {
  while (Serial2.available()) gps.encode(Serial2.read());

  uint32_t nowMs = millis();
  if (nowMs - lastPingMs >= PING_MS) {
    lastPingMs = nowMs;
    pingUltrasonic();
  }

  // Fixed-rate 100 Hz schedule (no drift); resync if we fall far behind
  if ((int32_t)(micros() - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_US;
    if ((int32_t)(micros() - nextSampleUs) > (int32_t)(5 * SAMPLE_US)) nextSampleUs = micros() + SAMPLE_US;

    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);

    bool valid = gps.location.isValid() && gps.time.isValid() && gps.date.isValid();

    char line[240];
    snprintf(line, sizeof(line),
      "%lu,%d,%04d-%02d-%02d,%02d:%02d:%02d.%02d,%lu,%.6f,%.6f,%.2f,%lu,"
      "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f",
      (unsigned long)millis(), valid ? 1 : 0,
      gps.date.year(), gps.date.month(), gps.date.day(),
      gps.time.hour(), gps.time.minute(), gps.time.second(), gps.time.centisecond(),
      (unsigned long)gps.time.age(),
      gps.location.lat(), gps.location.lng(), gps.speed.kmph(),
      (unsigned long)gps.satellites.value(),
      a.acceleration.x, a.acceleration.y, a.acceleration.z,
      g.gyro.x, g.gyro.y, g.gyro.z,
      lastDistCm);

    logFile.println(line);
    if (++linesSinceFlush >= FLUSH_EVERY) {
      logFile.flush();
      linesSinceFlush = 0;
    }

    digitalWrite(LED_PIN, valid ? HIGH : ((millis() / 500) % 2));
  }
}
