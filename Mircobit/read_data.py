import serial

port = "/dev/tty.usbmodem14302"  # change if needed
baudrate = 115200

ser = serial.Serial(port, baudrate, timeout=1)

while True:
    line = ser.readline().decode("utf-8").strip()

    if line:
        print(line)
