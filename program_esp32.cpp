// http_led_control.ino
// ESP8266 NodeMCU + 4 LEDs (D5..D8)
// HTTP commands:
//   /cmd?c=ch3al        -> all LEDs ON
//   /cmd?c=tfi          -> all LEDs OFF
//   /cmd?c=sini_bzarba  -> blink all LEDs every 200 ms
//   /cmd?c=sini_bchwiya -> blink all LEDs every 1000 ms

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// ====== WIFI CONFIG (beddel hadchi) ======
const char* WIFI_SSID = "Redmi Note 11";
const char* WIFI_PASS = "123456789omayyy";

// HTTP server on port 80
ESP8266WebServer server(80);

// ---------- PINS (ESP8266 NodeMCU) ----------
const int LED1_PIN = D5;
const int LED2_PIN = D6;
const int LED3_PIN = D7;
const int LED4_PIN = D8;

// --------------- MODES ---------------
enum Mode {
  MODE_OFF = 0,
  MODE_ON,
  MODE_BLINK_FAST,
  MODE_BLINK_SLOW
};

Mode currentMode = MODE_OFF;
bool blinkState = false;
unsigned long lastToggle = 0;

const unsigned long FAST_INTERVAL = 200;   // ms (sini bzarba)
const unsigned long SLOW_INTERVAL = 1000;  // ms (sini bchwiya)


// -------- HELPERS --------
void setAll(bool state) {
  digitalWrite(LED1_PIN, state ? HIGH : LOW);
  digitalWrite(LED2_PIN, state ? HIGH : LOW);
  digitalWrite(LED3_PIN, state ? HIGH : LOW);
  digitalWrite(LED4_PIN, state ? HIGH : LOW);
}

void updateLeds() {
  unsigned long now = millis();

  switch (currentMode) {
    case MODE_OFF:
      setAll(false);
      break;

    case MODE_ON:
      setAll(true);
      break;

    case MODE_BLINK_FAST:
    case MODE_BLINK_SLOW: {
      unsigned long interval =
        (currentMode == MODE_BLINK_FAST) ? FAST_INTERVAL : SLOW_INTERVAL;

      if (now - lastToggle >= interval) {
        lastToggle = now;
        blinkState = !blinkState;
        setAll(blinkState);
      }
      break;
    }
  }
}

// نفس handleCommand, غير دابا غادي نستعملوها من HTTP
void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  Serial.print("CMD: ");
  Serial.println(cmd);

  if (cmd == "CH3AL") {
    currentMode = MODE_ON;
    setAll(true);
  }
  else if (cmd == "TFI") {
    currentMode = MODE_OFF;
    setAll(false);
  }
  else if (cmd == "SINI_BZARBA") {
    currentMode = MODE_BLINK_FAST;
    blinkState = false;
    lastToggle = millis();
  }
  else if (cmd == "SINI_BCHWIYA") {
    currentMode = MODE_BLINK_SLOW;
    blinkState = false;
    lastToggle = millis();
  }
  else {
    Serial.println("Unknown command.");
  }
}

// ====== HTTP HANDLERS ======

void handleRoot() {
  String html = "<h1>ESP LED Controller</h1>";
  html += "<p>Use /cmd?c=ch3al | tfi | sini_bzarba | sini_bchwiya</p>";
  server.send(200, "text/html", html);
}

void handleHttpCommand() {
  if (!server.hasArg("c")) {
    server.send(400, "text/plain", "Missing parameter c");
    return;
  }
  String c = server.arg("c");
  // we accept commands in lowercase like 'ch3al'
  String cmd = c;
  cmd.toLowerCase();

  if (cmd == "ch3al") cmd = "CH3AL";
  else if (cmd == "tfi") cmd = "TFI";
  else if (cmd == "sini_bzarba") cmd = "SINI_BZARBA";
  else if (cmd == "sini_bchwiya") cmd = "SINI_BCHWIYA";

  handleCommand(cmd);
  String resp = "OK: " + cmd;
  server.send(200, "text/plain", resp);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}


// ------------- ARDUINO API -------------

void setup() {
  Serial.begin(115200);

  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);
  pinMode(LED3_PIN, OUTPUT);
  pinMode(LED4_PIN, OUTPUT);

  setAll(false);

  // WiFi connect
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());

  // HTTP routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/cmd", HTTP_GET, handleHttpCommand);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();  // handle HTTP
  updateLeds();           // update blinking
}
