import socket
import struct
import pickle
import cv2

# Connect to Raspberry Pi
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '10.125.32.71'  # <-- change this to your Raspberry Pi IP
port = 8000

client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")

print("[INFO] Connected to server. Receiving stream...")

try:
    while True:
        # Retrieve message size
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Retrieve full message data
        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize
        payload = pickle.loads(frame_data)
        frame = cv2.imdecode(payload["frame"], cv2.IMREAD_COLOR)
        boxes = payload["boxes"]

        # Draw bounding boxes (in case you want to redraw on client side)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            track_id = b["track_id"]
            conf = b["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id} | {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

        cv2.imshow("YOLO Stream (with Track IDs)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("[ERROR]", e)
finally:
    client_socket.close()
    cv2.destroyAllWindows()
