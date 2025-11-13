const byte s0 = A4; 
const byte s1 = A3; 
const byte s2 = A2; 
const byte s3 = A1;
const byte w0 = 6;  
const byte w1 = 5; 
const byte w2 = 4; 
const byte w3 = 3;
const byte SIG_pin = 0; 
const byte OUT_pin = 2;
const byte STATUS_pin = 8; 
const byte COL_pin = 9;

const boolean muxChannel[16][4]={
    {0,0,0,0}, //channel 0
    {1,0,0,0}, //channel 1
    {0,1,0,0}, //channel 2
    {1,1,0,0}, //channel 3
    {0,0,1,0}, //channel 4
    {1,0,1,0}, //channel 5
    {0,1,1,0}, //channel 6
    {1,1,1,0}, //channel 7
    {0,0,0,1}, //channel 8
    {1,0,0,1}, //channel 9
    {0,1,0,1}, //channel 10
    {1,1,0,1}, //channel 11
    {0,0,1,1}, //channel 12
    {1,0,1,1}, //channel 13
    {0,1,1,1}, //channel 14
    {1,1,1,1}  //channel 15
  };

int valor = 0;               //variable for sending bytes to processing
int calibra[15][15];         //Calibration array for the min values of each od the 225 sensors.
int minsensor=254;          //Variable for staring the min array
int multiplier = 254;
int pastmatrix[15][15];

// Timing variables for 200ms interval
unsigned long previousMillis = 0;
const unsigned long interval = 200; // 200ms interval

void setup(){
    
  pinMode(s0, OUTPUT); 
  pinMode(s1, OUTPUT); 
  pinMode(s2, OUTPUT); 
  pinMode(s3, OUTPUT); 
  
  pinMode(w0, OUTPUT); 
  pinMode(w1, OUTPUT); 
  pinMode(w2, OUTPUT); 
  pinMode(w3, OUTPUT); 
  
  pinMode(OUT_pin, OUTPUT); 
  
  pinMode(STATUS_pin, OUTPUT);
  pinMode(COL_pin, OUTPUT);

  
  digitalWrite(s0, LOW);
  digitalWrite(s1, LOW);
  digitalWrite(s2, LOW);
  digitalWrite(s3, LOW);
  
  digitalWrite(w0, LOW);
  digitalWrite(w1, LOW);
  digitalWrite(w2, LOW);
  digitalWrite(w3, LOW);
  
  digitalWrite(OUT_pin, HIGH);
  digitalWrite(STATUS_pin, HIGH);
  digitalWrite(COL_pin, HIGH);
  
  Serial.begin(115200);
  
  Serial.println("\n\Calibrating...\n");
  
  // Full of 0's of initial matrix
  for(byte j = 0; j < 15; j ++){ 
    writeMux(j);
    for(byte i = 0; i < 15; i ++)
      calibra[j][i] = 0;
  }
  
  // Calibration
  for(byte k = 0; k < 50; k++){  
    for(byte j = 0; j < 15; j ++){ 
      writeMux(j);
      for(byte i = 0; i < 15; i ++)
        calibra[j][i] = calibra[j][i] + readMux(i);
    }
  }
  
  //Print averages
  for(byte j = 0; j < 15; j ++){ 
    writeMux(j);
    for(byte i = 0; i < 15; i ++){
      calibra[j][i] = calibra[j][i]/50;
      if(calibra[j][i] < minsensor)
        minsensor = calibra[j][i];
      Serial.print(calibra[j][i]);
      Serial.print("\t");
    }
  Serial.println(); 
  }
  
  Serial.println();
  Serial.print("Minimum Value: ");
  Serial.println(minsensor);
  Serial.println();
  Serial.println("Starting continuous readings every 200ms...");
  Serial.println("Format: Row,Col,Value");
  Serial.println("------------------------");
  
  digitalWrite(COL_pin, LOW);
}

void loop(){
  unsigned long currentMillis = millis();
  
  // Check if 200ms have passed
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    
    // Read all 225 sensors (15x15 matrix)
    Serial.println("START");
    
    for(int j = 0; j < 15; j++){ 
      writeMux(j);
      
      for(int i = 0; i < 15; i++){
        valor = readMux(i);
        
        // Use full analog range (0-1023) for maximum sensitivity
        // Only apply minimum calibration threshold
        if(valor < calibra[j][i])
          valor = calibra[j][i];
        
        // Map from calibrated minimum to full analog range (1023)
        // This gives much better dynamic range
        valor = map(valor, minsensor, 1023, 0, 1023);
        
        // Clean up any mapping artifacts
        if(valor < 0)
          valor = 0;
        if(valor > 1023)
          valor = 1023;
        
        // Print value followed by space (no newlines until end of row)
        Serial.print(valor);
        Serial.print(" ");
        
        digitalWrite(COL_pin, !digitalRead(COL_pin));
      }
      // New line after each row of 15 values
      Serial.println();
    }
    Serial.println("END");
  }
}

int readMux(byte channel){
  byte controlPin[] = {s0, s1, s2, s3};

  //loop through the 4 sig
  for(int i = 0; i < 4; i ++){
    digitalWrite(controlPin[i], muxChannel[channel][i]);
  }

  //read the value at the SIG pin
  int val = analogRead(SIG_pin);

  //return the value
  return val;
}

void writeMux(byte channel){
  byte controlPin[] = {w0, w1, w2, w3};

  //loop through the 4 sig
  for(byte i = 0; i < 4; i ++){
    digitalWrite(controlPin[i], muxChannel[channel][i]);
  }
}