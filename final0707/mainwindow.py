import sys
import os
# import rc_img                      # .qrc → rc_img.py 변환 리소스
import torch
import cv2
from ultralytics import YOLO      # YOLOv8 import
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox,
    QLineEdit, QPushButton, QTableWidgetItem, QHeaderView,
    QFileDialog, QGraphicsScene, QDialog,
    QLabel, QVBoxLayout, QAbstractItemView
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from ui_form import Ui_MainWindow
from socket_client import send_login, send_history, send_save, send_request_image, send_signup

# ─── 설정 ─────────────────────────────────────────────

MODEL_PATH = "./RESN0701.pth"
YOLO_PATH  = "./YOLO0707.pt"
PAGE_LOGIN, PAGE_MENU, PAGE_ADD, PAGE_HISTORY = 0, 1, 2, 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_local_model(path=MODEL_PATH):
    model = models.resnet18(weights=None)
    in_f = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_f, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 2)
    )
    model_dict = model.state_dict()

    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    filtered = { k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size() }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model.eval()
    return model.to(device)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.stackedWidget.setCurrentIndex(PAGE_LOGIN)
        self.ui.lineEdit_pw.setEchoMode(QLineEdit.Password)

        self.camera_scene = QGraphicsScene(self)
        self.ui.graphicsView_addimg.setScene(self.camera_scene)

        self.model = load_local_model().to(device)

        self.yolo = YOLO(YOLO_PATH)
        self.yolo.model.to(device)
        if device != 'cpu':
            self.yolo.model.half()
            # 워밍업
            self.yolo.model.warmup(imgsz=(1,3,640,640))

        self.current_image_path = None
        self.check_result = None
        self.cap = None
        self.timer = None

        for btn in [
            self.ui.pushButton_check, self.ui.pushButton_save,
            self.ui.pushButton_shoot, self.ui.pushButton_check0,
            self.ui.pushButton_save0
        ]:
            btn.setEnabled(False)

        tbl = self.ui.tableWidget_history
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        tbl.setAlternatingRowColors(True)
        tbl.setShowGrid(True)
        tbl.cellClicked.connect(self.on_history_cell_clicked)

        self.ui.stackedWidget.currentChanged.connect(self.on_page_changed)

        self.ui.pushButton_login.clicked.connect(self.try_login)
        self.ui.pushButton_signin.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(4)
        )
        self.ui.pushButton_findpw.clicked.connect(
            lambda: QMessageBox.information(self, "PW 찾기", "해당 전화번호로 임시 비밀번호가 전송되었습니다.")
        )
        self.ui.signup_back_Btn.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(0)
        )
        self.ui.signup_save_Btn.clicked.connect(self.try_signup)

        self.ui.pushButton_test.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(PAGE_ADD)
        )
        self.ui.pushButton_history.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(PAGE_HISTORY)
        )
        self.ui.pushButton_logout.clicked.connect(self.logout)
        self.ui.pushButton_exit.clicked.connect(self.confirm_exit)
        self.ui.pushButton_back.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(PAGE_MENU)
        )
        self.ui.pushButton_back_2.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(PAGE_MENU)
        )
        self.ui.pushButton_addnew.clicked.connect(self.add_new_image)
        self.ui.pushButton_cameraon.clicked.connect(self.start_camera)
        self.ui.pushButton_shoot.clicked.connect(self.shoot_image)
        self.ui.pushButton_check.clicked.connect(self.check_image)
        self.ui.pushButton_save.clicked.connect(self.save_image)
        self.ui.pushButton_check0.clicked.connect(self.check_image)
        self.ui.pushButton_save0.clicked.connect(self.save_image)

    def logout(self):
        self.ui.lineEdit_id.clear()
        self.ui.lineEdit_pw.clear()
        self.ui.stackedWidget.setCurrentIndex(PAGE_LOGIN)
        QMessageBox.information(self, "로그아웃", "안전하게 로그아웃되었습니다.")

    def confirm_exit(self):
        reply = QMessageBox.question(
            self, "프로그램 종료", "정말 프로그램을 종료하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QApplication.instance().quit()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "프로그램 종료", "정말 프로그램을 종료하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def on_page_changed(self, index):
        if index == PAGE_ADD:
            self.camera_scene.clear()
            self.stop_camera()
            for btn in [
                self.ui.pushButton_shoot, self.ui.pushButton_check,
                self.ui.pushButton_save, self.ui.pushButton_check0,
                self.ui.pushButton_save0
            ]:
                btn.setEnabled(False)
            self.current_image_path = None
            self.check_result = None
        elif index == PAGE_HISTORY:
            self.load_history()

    def try_login(self):
        ph = self.ui.lineEdit_id.text().strip()
        pw = self.ui.lineEdit_pw.text().strip()
        if not ph or not pw:
            QMessageBox.warning(
                self, "입력 오류", "전화번호와 비밀번호를 모두 입력하세요."
            )
            return
        res = send_login(ph, pw)
        if res is None:
            QMessageBox.critical(self, "오류", "서버에 연결할 수 없습니다.")
        elif res.get("protocol") == "1_1" and res.get("result") == "success":
            QMessageBox.information(self, "성공", "로그인에 성공했습니다!")
            self.ui.stackedWidget.setCurrentIndex(PAGE_MENU)
        else:
            QMessageBox.warning(
                self, "실패", "전화번호 또는 비밀번호가 올바르지 않습니다."
            )

    def load_history(self):
        phone = self.ui.lineEdit_id.text().strip()
        res = send_history(phone)
        tbl = self.ui.tableWidget_history
        headers = ["번호", "시간", "상태", "이미지보기", "파일명"]
        tbl.clear()
        tbl.setColumnCount(len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.setColumnHidden(0, True)
        tbl.setColumnHidden(4, True)
        tbl.verticalHeader().setVisible(False)

        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        if res and res.get("protocol") == "100_1":
            recs = res.get("records", [])
            tbl.setRowCount(len(recs))
            for i, r in enumerate(recs):
                tbl.setItem(i, 0, QTableWidgetItem(str(r.get("num", ""))) )
                raw = r.get("addtime", "")
                try:
                    dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                    tbl.setItem(i, 1, QTableWidgetItem(dt.strftime("%y-%m-%d %H:%M")))
                except:
                    tbl.setItem(i, 1, QTableWidgetItem(raw[:16]))
                status = int(r.get("status", 1))
                item = QTableWidgetItem("✅ 정상" if status == 0 else "❌ 상함")
                item.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(i, 2, item)
                btn = QPushButton("🖼️보기")
                btn.clicked.connect(lambda _, row=i: self.show_history_image(row))
                tbl.setCellWidget(i, 3, btn)
                tbl.setItem(i, 4, QTableWidgetItem(r.get("filename", "")))
        else:
            tbl.setRowCount(0)
            QMessageBox.information(self, "내역 없음", "조회된 이력이 없습니다.")

        tbl.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #fdebd3;
                border: none;
                gridline-color: #dcb891;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #f5d6ab;
                color: #000000;
                border: 1px solid #dcb891;
            }
            QTableCornerButton::section {
                background-color: #f5d6ab;
                border: 1px solid #dcb891;
            }
            QTableWidget::item:selected {
                background-color: #fcd5a4;
            }
        """)
        tbl.setAlternatingRowColors(True)
        tbl.viewport().setStyleSheet("background-color: #fde0b6;")

    def on_history_cell_clicked(self, row, column):
        if column == 3:
            self.show_history_image(row)

    def show_history_image(self, row):
        tbl = self.ui.tableWidget_history
        filename = tbl.model().index(row, 4).data()
        phone = self.ui.lineEdit_id.text().strip()
        data = send_request_image(phone, filename)
        if not data:
            QMessageBox.warning(self, "오류", "이미지 요청에 실패했습니다.")
            return
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, f"hist_{filename}")
        with open(path, "wb") as f:
            f.write(data)
        dlg = QDialog(self)
        dlg.setWindowTitle("이력 이미지")
        lbl = QLabel(dlg)
        lbl.setPixmap(QPixmap(path).scaled(400, 400, Qt.KeepAspectRatio))
        vbox = QVBoxLayout(dlg)
        vbox.addWidget(lbl)
        dlg.exec()

    def add_new_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Images (*.png *.jpg *.bmp *.jpeg)"
        )
        if not file_path:
            return
        self.current_image_path = file_path
        self.camera_scene.clear()
        self.camera_scene.addPixmap(QPixmap(file_path))
        self.ui.graphicsView_addimg.fitInView(
            self.camera_scene.itemsBoundingRect(), Qt.KeepAspectRatio
        )
        self.ui.pushButton_check.setEnabled(True)
        self.ui.pushButton_save.setEnabled(False)
        self.ui.pushButton_check0.setEnabled(False)
        self.ui.pushButton_save0.setEnabled(False)
        self.check_result = None

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("카메라 열기 실패")
            if self.timer is None:
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.ui.pushButton_shoot.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "카메라 오류", str(e))

    def update_frame(self):
        if not (self.cap and self.cap.isOpened()):
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.camera_scene.clear()
        self.camera_scene.addPixmap(pix)
        self.ui.graphicsView_addimg.fitInView(
            self.camera_scene.itemsBoundingRect(), Qt.KeepAspectRatio
        )

    def stop_camera(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def shoot_image(self):
        if not (self.cap and self.cap.isOpened()):
            QMessageBox.warning(self, "캡처 오류", "카메라가 꺼져 있습니다.")
            return
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "캡처 오류", "프레임을 가져올 수 없습니다.")
            return
        self.stop_camera()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.camera_scene.clear()
        self.camera_scene.addPixmap(pix)
        self.ui.graphicsView_addimg.fitInView(
            self.camera_scene.itemsBoundingRect()), Qt.KeepAspectRatio
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%y.%m.%d.%H.%M.%S")
        path = os.path.join(tmp_dir, f"captured_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        self.current_image_path = path
        self.ui.pushButton_check0.setEnabled(True)
        self.ui.pushButton_save0.setEnabled(False)

    def check_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "검사 오류", "이미지를 먼저 첨부하세요.")
            return

        img = cv2.imread(self.current_image_path)
        results = self.yolo(img, imgsz=320, conf=0.3, iou=0.5)
        res   = results[0]
        boxes = res.boxes

        # 디버깅 출력
        print("클래스 목록:", [float(b.conf[0]) for b in boxes])
        print("박스 개수:", len(boxes))

        # ─── 클래스 하나만 있으므로, 박스 유무로 판단 ─────────────────
        if len(boxes) == 0:
            QMessageBox.information(
                self, "전송 차단", "음식이 아니므로 재촬영 또는 재첨부해주세요."
            )
            self.camera_scene.clear()
            self.current_image_path = None
            self.ui.pushButton_check.setEnabled(False)
            self.ui.pushButton_save.setEnabled(False)
            self.ui.pushButton_shoot.setEnabled(True)
            return

        # ResNet으로 정상/상함 분류
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        pil_img = Image.open(self.current_image_path).convert("RGB")
        inp     = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(inp)
        _, pred = torch.max(out, 1)

        self.check_result = "normal" if pred.item() == 0 else "damaged"
        QMessageBox.information(
            self, "검사 결과",
            f"{'✅ 정상' if pred.item()==0 else '❌ 상함'}"
        )
        self.ui.pushButton_save.setEnabled(True)
        self.ui.pushButton_save0.setEnabled(True)

    def save_image(self):
        if not self.current_image_path or not self.check_result:
            return
        res = send_save(
            self.ui.lineEdit_id.text().strip(),
            self.current_image_path,
            self.check_result
        )
        if res and res.get("protocol") == "20_1":
            QMessageBox.information(self, "저장 완료", "이미지가 서버에 저장되었습니다.")
        else:
            QMessageBox.warning(self, "저장 실패", "이미지 저장에 실패했습니다.")

    def try_signup(self):
        phone = self.ui.textEdit_2.toPlainText().strip()
        pw = self.ui.textEdit_3.toPlainText().strip()

        if not phone or not pw:
            QMessageBox.warning(self, "입력 오류", "전화번호와 비밀번호를 모두 입력하세요.")
            return

        res = send_signup(phone, pw)
        if res and res.get("protocol") == "2_1":
            if res.get("result") == "success":
                QMessageBox.information(self, "회원가입 완료", "회원가입이 완료되었습니다.")
                self.ui.stackedWidget.setCurrentIndex(PAGE_LOGIN)
            elif res.get("result") == "duplicate":
                QMessageBox.warning(self, "중복 오류", "이미 존재하는 전화번호입니다.")
            else:
                QMessageBox.critical(self, "회원가입 실패", "회원가입 중 오류가 발생했습니다.")
        else:
            QMessageBox.critical(self, "통신 오류", "서버와 연결할 수 없습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
