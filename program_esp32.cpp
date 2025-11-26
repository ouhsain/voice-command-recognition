#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "YOUR_WIFI";
const char* password = "PASSWORD";

WebServer server(80);

int led = 2;

void handleCommand() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");

    if (body.indexOf("ch3al") != -1) {
      digitalWrite(led, HIGH);
    }
    else if (body.indexOf("tfi") != -1) {
      digitalWrite(led, LOW);
    }
    else if (body.indexOf("sini bzarba") != -1) {
      for (int i=0; i<5; i++) {
        digitalWrite(led, HIGH);
        delay(100);
        digitalWrite(led, LOW);
        delay(100);
      }
    }
    else if (body.indexOf("sini bchwiya") != -1) {
      digitalWrite(led, HIGH);
      delay(1000);
      digitalWrite(led, LOW);
      delay(1000);
    }

    server.send(200, "application/json", "{\"status\":\"ok\"}");
  }
}

void setup() {
  pinMode(led, OUTPUT);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) delay(500);

  server.on("/command", HTTP_POST, handleCommand);
  server.begin();
}

void loop() {
  server.handleClient();
}
