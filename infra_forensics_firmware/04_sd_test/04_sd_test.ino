// TEST 4: MicroSD module (card formatted FAT32)
// Wiring: VCC->5V (VIN) [use 3V3 only if module is marked 3.3V-only], GND->GND,
//         CS->GPIO5, SCK->GPIO18, MISO->GPIO19, MOSI->GPIO23
// Expected: "OK" lines and the card size.
#include <SPI.h>
#include <SD.h>

void setup() {
  Serial.begin(115200);
  delay(500);
  if (!SD.begin(5)) {
    Serial.println("FAIL: SD init. Check wiring, card inserted, FAT32 format.");
    return;
  }
  Serial.printf("OK: card size = %llu MB\n", SD.cardSize() / (1024ULL * 1024ULL));

  File f = SD.open("/test.txt", FILE_WRITE);
  if (!f) { Serial.println("FAIL: could not create file"); return; }
  f.println("Infrastructure Forensics SD test");
  f.close();

  f = SD.open("/test.txt");
  Serial.print("OK: read back -> ");
  while (f.available()) Serial.write(f.read());
  f.close();
}

void loop() {}
