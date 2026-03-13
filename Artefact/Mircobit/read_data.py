import serial

port = "/dev/tty.usbmodem14302" 
baudrate = 115200

ser = serial.Serial(port, baudrate, timeout=1)

while True:
    line = ser.readline().decode("utf-8").strip()

    if line:
        print(line)



# I will not be using serial due to the data collection being inacurate
# example
# Was getting readings missing/adding numbers/letters
# Making the data usable