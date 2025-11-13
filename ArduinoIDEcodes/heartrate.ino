/*
  HLK‑LD6002 Heart‑Rate Reader (V1.3 Protocol)
  
  - UART: 1 382 400 baud
  - Frame TYPE for heartbeat rate: 0x0A15
  - DATA length: 4 bytes (little‑endian float)
  
  Frame structure (all big‑endian except DATA):
  ┌────────┬──────┬──────┬──────┬────────────┬──────┬───────┐
  │ Offset │ Size │ Name │ Type │ Description │   Example   │
  ├────────┼──────┼──────┼──────┼────────────┼─────────────┤
  │ 0      │ 1    │ SOF  │ u8   │ 0x01       │ 0x01        │
  │ 1–2    │ 2    │ ID   │ u16  │ Frame ID   │ —           │
  │ 3–4    │ 2    │ LEN  │ u16  │ Data length (0x0004) │ 0x00 04 │
  │ 5–6    │ 2    │ TYPE │ u16  │ Message type (0x0A15) │ 0A 15  │
  │ 7      │ 1    │ HCS  │ u8   │ ~(SOF⊕ID⊕LEN⊕TYPE)   │ —        │
  │ 8–11   │ 4    │ DATA │ float (LE) │ Heart-rate (BPM) │ —     │
  │ 12     │ 1    │ DCS  │ u8   │ ~(DATA[0]⊕DATA[1]⊕DATA[2]⊕DATA[3]) │ — │
  └────────┴──────┴──────┴──────┴────────────┴─────────────┘

References:  
– HLK‑LD6002 product page (power & data specs) :contentReference[oaicite:0]{index=0}  
– HLK‑LD6002 Communication Protocol V1.3 (June 7 2025) :contentReference[oaicite:1]{index=1}  
*/

#include <Arduino.h>
#define RX_PIN 16    
#define TX_PIN 17    
#define BAUD    1382400UL

// Frame parser state
uint8_t buf[32];
uint8_t idx = 0;

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("HLK‑LD6002 Heart‑Rate Reader");

  // Use Serial1 (change to Serial2 etc. on other MCUs)
  Serial1.begin(BAUD, SERIAL_8N1, RX_PIN, TX_PIN);
}

uint8_t calc_checksum(const uint8_t* data, uint8_t len) {
  uint8_t x = 0;
  for (uint8_t i = 0; i < len; i++) {
    x ^= data[i];
  }
  return ~x;
}

void loop() {
  while (Serial1.available()) {
    uint8_t b = Serial1.read();
    // Add byte to buffer
    if (idx < sizeof(buf)) {
      buf[idx++] = b;
    } else {
      // overflow—reset
      idx = 0;
      continue;
    }

    // Minimum frame size = SOF(1) + ID(2) + LEN(2) + TYPE(2) + HCS(1) + DATA(4) + DCS(1) = 13
    if (idx >= 13) {
      // Look for SOF at buf[0]
      if (buf[0] == 0x01) {
        // Extract LEN and TYPE
        uint16_t len  = (uint16_t(buf[3]) << 8) | buf[4];
        uint16_t type = (uint16_t(buf[5]) << 8) | buf[6];

        // Check we have the full frame
        if (idx >= (7 + 1 + len + 1) && len == 4 && type == 0x0A15) {
          // Verify header checksum (bytes 0 through 6)
          uint8_t expected_hcs = buf[7];
          if (calc_checksum(buf, 7) != expected_hcs) {
            // Bad header checksum
            idx = 0;
            return;
          }

          // Verify data checksum (bytes 8 through 11)
          uint8_t expected_dcs = buf[8 + len];
          if (calc_checksum(buf + 8, len) != expected_dcs) {
            // Bad data checksum
            idx = 0;
            return;
          }

          // Parse float (little endian)
          float hr;
          memcpy(&hr, buf + 8, sizeof(float));

          // Print result
          Serial.print("Heart rate: ");
          Serial.print(hr, 1);
          Serial.println(" BPM");

          // Done—reset buffer for next frame
          idx = 0;
          return;
        }
      }

      // If we didn’t parse a valid frame, shift buffer by one and keep looking
      memmove(buf, buf + 1, --idx);
    }
  }
}
