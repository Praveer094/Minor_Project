// TEST 3: HC-SR04 ultrasonic
// Wiring: VCC->5V (VIN), GND->GND, TRIG->GPIO27,
//         ECHO -> 1k resistor -> junction -> 2k resistor -> GND ; junction -> GPIO26
// NEVER connect ECHO directly to the ESP32.
// Expected: hold your hand 10-50 cm above it; distance follows your hand.
#define TRIG_PIN 27
#define ECHO_PIN 26

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long d = pulseIn(ECHO_PIN, HIGH, 25000);
  if (d == 0) Serial.println("no echo (out of range, or check wiring/divider)");
  else        Serial.printf("distance = %.1f cm\n", d * 0.0343f / 2.0f);
  delay(200);
}
