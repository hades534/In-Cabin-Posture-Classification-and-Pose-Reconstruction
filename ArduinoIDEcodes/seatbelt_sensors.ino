const int sensorPin1 = 34;  // ADC1_CH6
const int sensorPin2 = 35;  // ADC1_CH7
const int sensorPin3 = 32;  // ADC1_CH4
const int sensorPin4 = 33;  // ADC1_CH5
const int sensorPin5 = 36;  // ADC1_CH0 (VP)
const int sensorPin6 = 25;  // ADC2_CH8
const int sensorPin7 = 26;  // ADC2_CH9
const int sensorPin8 = 27;  // ADC2_CH7

void setup() {
  Serial.begin(115200);
  // Giving the serial port a moment to set up
  delay(1000);
}

void loop() {
  // Read values from eight analog sensors (0–4095 on ESP32)
  int sensorValue1 = analogRead(sensorPin1);
  int sensorValue2 = analogRead(sensorPin2);
  int sensorValue3 = analogRead(sensorPin3);
  int sensorValue4 = analogRead(sensorPin4);
  int sensorValue5 = analogRead(sensorPin5);
  int sensorValue6 = analogRead(sensorPin6);
  int sensorValue7 = analogRead(sensorPin7);
  int sensorValue8 = analogRead(sensorPin8);

  // Print sensor values comma-separated
  Serial.print(sensorValue1); Serial.print(',');
  Serial.print(sensorValue2); Serial.print(',');
  Serial.print(sensorValue3); Serial.print(',');
  Serial.print(sensorValue4); Serial.print(',');
  Serial.print(sensorValue5); Serial.print(',');
  Serial.print(sensorValue6); Serial.print(',');
  Serial.print(sensorValue7); Serial.print(',');
  Serial.println(sensorValue8);

  // Short delay between readings
  delay(200);
}
