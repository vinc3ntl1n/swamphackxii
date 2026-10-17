const int BUTTON_PIN = 2;
const int BUZZER_PIN = 8;
const unsigned long DEBOUNCE_MS = 50;
const unsigned long BEEP_MS = 200;

bool alarmOn = false;
bool lastPressed = false;
unsigned long lastChange = 0;
String incoming = "";

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  bool pressed = digitalRead(BUTTON_PIN) == LOW;
  if (pressed != lastPressed && millis() - lastChange > DEBOUNCE_MS) {
    lastChange = millis();
    lastPressed = pressed;
    if (pressed) Serial.println("CANCEL");
  }

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      incoming.trim();
      if (incoming == "A1") alarmOn = true;
      if (incoming == "A0") alarmOn = false;
      incoming = "";
    } else {
      incoming += c;
    }
  }

  if (alarmOn && (millis() / BEEP_MS) % 2 == 0) {
    tone(BUZZER_PIN, 2000);
  } else {
    noTone(BUZZER_PIN);
  }
}
