// TEST 2: NEO-6M GPS  (must be OUTDOORS or at an open window, antenna facing sky)
// Wiring: VCC->3V3, GND->GND, GPS TX->GPIO16, GPS RX->GPIO17
// Expected: chars keeps increasing immediately; fix=YES after 2-5 minutes outdoors.
#include <TinyGPSPlus.h>

TinyGPSPlus gps;
uint32_t lastPrint = 0;

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1, 16, 17);
  Serial.println("Waiting for GPS... take it outdoors.");
}

void loop() {
  while (Serial2.available()) gps.encode(Serial2.read());

  if (millis() - lastPrint > 1000) {
    lastPrint = millis();
    Serial.printf("chars=%lu sats=%lu fix=%s lat=%.6f lng=%.6f time=%02d:%02d:%02d UTC\n",
                  (unsigned long)gps.charsProcessed(),
                  (unsigned long)gps.satellites.value(),
                  gps.location.isValid() ? "YES" : "no",
                  gps.location.lat(), gps.location.lng(),
                  gps.time.hour(), gps.time.minute(), gps.time.second());
    if (millis() > 10000 && gps.charsProcessed() < 10)
      Serial.println("FAIL: no data from GPS. Swap the TX/RX wires.");
  }
}
