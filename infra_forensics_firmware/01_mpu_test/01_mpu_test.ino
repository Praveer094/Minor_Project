// TEST 1: MPU6050 accelerometer
// Wiring: VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22
// Expected: board lying flat shows az close to 9.8; tapping it makes values jump.
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  if (!mpu.begin()) {
    Serial.println("FAIL: MPU6050 not found. Check SDA=21, SCL=22, VCC=3V3, GND.");
    while (true) delay(10);
  }
  Serial.println("OK: MPU6050 found");
}

void loop() {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  Serial.printf("ax=%6.2f  ay=%6.2f  az=%6.2f  m/s^2\n",
                a.acceleration.x, a.acceleration.y, a.acceleration.z);
  delay(200);
}
