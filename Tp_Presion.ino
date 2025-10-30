#include <ADS1115_WE.h> 
#include <Wire.h>

#define I2C_ADDRESS 0x48

// Se crea el objeto para el ADS1115
ADS1115_WE adc = ADS1115_WE(I2C_ADDRESS);

// --- Variables para el control de tiempo no bloqueante ---
// Frecuencia de muestreo deseada (128 Hz)
const unsigned long SAMPLING_FREQUENCY = 128; 
// Intervalo en microsegundos entre muestras (1 / 128 Hz = 7813 µs)
const unsigned long interval_micros = 1000000 / SAMPLING_FREQUENCY;
// Variable para guardar el tiempo de la última muestra
unsigned long previous_micros = 0;

void setup() {
  Wire.begin();
  // Se recomienda usar una velocidad de baudios más alta para no crear un cuello de botella
  Serial.begin(9600); 

  // --- Bucle de espera para la conexión del ADS1115 ---
  // El programa no avanzará de este punto hasta que adc.init() sea exitoso.
  Serial.println("Buscando el sensor ADS1115...");
  while(!adc.init()){
    Serial.println("ADS1115 no encontrado. Reintentando en 500ms...");
    delay(500); // Espera medio segundo antes de volver a intentar.
  }
  Serial.println("¡ADS1115 conectado!");

  // --- CONFIGURACIÓN ÚNICA DEL ADC ---
  // Todas estas configuraciones se realizan una sola vez.

  // Rango de voltaje: +/- 2.048V. ¡No aplicar más de VDD + 0.3V a los pines!
  adc.setVoltageRange_mV(ADS1115_RANGE_2048);

  // Canal a medir: AIN0 contra GND.
  // Esto se configura aquí y no se vuelve a llamar en el loop.
  adc.setCompareChannels(ADS1115_COMP_0_GND);

  // Velocidad de conversión: 128 muestras por segundo (SPS).
  // El microcontrolador debe leer los datos a esta velocidad.
  adc.setConvRate(ADS1115_128_SPS);

  // Modo de medición: Continuo. El ADC convierte constantemente.
  adc.setMeasureMode(ADS1115_CONTINUOUS);

}

void loop() {
  // Se obtiene el tiempo actual en microsegundos
  unsigned long current_micros = micros();

  // Comprueba si ha pasado el tiempo suficiente desde la última lectura
  if (current_micros - previous_micros >= interval_micros) {
    // Guarda el tiempo de esta lectura para la próxima comparación
    previous_micros = current_micros;

    // Se lee el último resultado de la conversión directamente.
    // No es necesario llamar a setCompareChannels de nuevo.
    float voltage = adc.getResult_V(); 

    // Se imprime el valor del voltaje con 6 decimales para mayor precisión
    Serial.println(voltage, 6);
  }

}