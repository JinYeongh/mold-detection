import socket
import json
import os

# 서버 연결 설정
SERVER_HOST = "localhost"
SERVER_PORT = 12345
BUFFER_SIZE = 4096


def send_login(phonenumber: str, password: str) -> dict | None:
    payload = {"protocol": "1_0", "phonenumber": phonenumber, "password": password}
    try:
        with socket.socket() as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            s.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            data = s.recv(BUFFER_SIZE).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print("서버 연결 실패 (login):", e)
        return None

def send_signup(phonenumber: str, password: str) -> dict | None:
    payload = {
        "protocol": "2_0",
        "phonenumber": phonenumber,
        "password": password
    }
    try:
        with socket.socket() as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            s.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            data = s.recv(BUFFER_SIZE).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print("서버 연결 실패 (signup):", e)
        return None


def send_history(phonenumber: str) -> dict | None:
    payload = {"protocol": "100_0", "phonenumber": phonenumber}
    try:
        with socket.socket() as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            s.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            data = s.recv(BUFFER_SIZE).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print("서버 연결 실패 (history):", e)
        return None


def send_save(phonenumber: str, image_path: str, result: str) -> dict | None:
    """
    서버에 이미지 저장 요청 (protocol 20_0)
    """
    try:
        filesize = os.path.getsize(image_path)
        header = {
            "protocol": "20_0",
            "phonenumber": phonenumber,
            "result": result,
            "filename": os.path.basename(image_path),
            "filesize": filesize
        }
        with socket.socket() as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            s.sendall(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n")
            with open(image_path, "rb") as f:
                while True:
                    chunk = f.read(BUFFER_SIZE)
                    if not chunk: break
                    s.sendall(chunk)
            data = s.recv(BUFFER_SIZE).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print("서버 연결 실패 (save):", e)
        return None


def send_request_image(phonenumber: str, filename: str) -> bytes | None:
    """
    서버에 이미지 요청 (protocol 30_0).
    응답으로 이미지 바이트 전체를 반환.
    """
    payload = {
        "protocol": "30_0",
        "phonenumber": phonenumber,
        "filename": filename
    }
    try:
        with socket.socket() as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            # 헤더만 전송 (JSON + 개행)
            s.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            # 서버에서 소켓 닫힐 때까지 이미지 데이터 수신
            data = bytearray()
            while True:
                chunk = s.recv(BUFFER_SIZE)
                if not chunk:
                    break
                data.extend(chunk)
        return bytes(data)
    except Exception as e:
        print("서버 연결 실패 (request_image):", e)
        return None
