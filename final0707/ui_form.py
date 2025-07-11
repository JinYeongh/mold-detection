# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGraphicsView, QGridLayout, QGroupBox,
    QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QStackedWidget, QStatusBar, QTableWidget, QTableWidgetItem,
    QTextEdit, QWidget)
import rc_img

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(450, 800)
        MainWindow.setMinimumSize(QSize(450, 800))
        MainWindow.setMaximumSize(QSize(450, 800))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"")
        self.page_login = QWidget()
        self.page_login.setObjectName(u"page_login")
        self.gridLayout_2 = QGridLayout(self.page_login)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_5 = QLabel(self.page_login)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setPixmap(QPixmap(u":/images/loginmain.png"))
        self.label_5.setScaledContents(True)

        self.gridLayout_2.addWidget(self.label_5, 0, 0, 3, 3)

        self.groupBox_login_lineEdit = QGroupBox(self.page_login)
        self.groupBox_login_lineEdit.setObjectName(u"groupBox_login_lineEdit")
        self.groupBox_login_lineEdit.setMaximumSize(QSize(16777215, 110))
        self.gridLayout_3 = QGridLayout(self.groupBox_login_lineEdit)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_3 = QLabel(self.groupBox_login_lineEdit)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(0, 40))
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255, 255))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText, brush)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.WindowText, brush)
        self.label_3.setPalette(palette)
        font = QFont()
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)

        self.lineEdit_id = QLineEdit(self.groupBox_login_lineEdit)
        self.lineEdit_id.setObjectName(u"lineEdit_id")
        self.lineEdit_id.setMinimumSize(QSize(0, 30))
        self.lineEdit_id.setFont(font)
        self.lineEdit_id.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_id.setClearButtonEnabled(True)

        self.gridLayout_3.addWidget(self.lineEdit_id, 0, 1, 1, 1)

        self.lineEdit_pw = QLineEdit(self.groupBox_login_lineEdit)
        self.lineEdit_pw.setObjectName(u"lineEdit_pw")
        self.lineEdit_pw.setMinimumSize(QSize(0, 30))
        self.lineEdit_pw.setFont(font)
        self.lineEdit_pw.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_pw.setClearButtonEnabled(True)

        self.gridLayout_3.addWidget(self.lineEdit_pw, 1, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox_login_lineEdit)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 40))
        palette1 = QPalette()
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.WindowText, brush)
        self.label_2.setPalette(palette1)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)

        self.pushButton_login = QPushButton(self.groupBox_login_lineEdit)
        self.pushButton_login.setObjectName(u"pushButton_login")
        self.pushButton_login.setMinimumSize(QSize(0, 40))
        font1 = QFont()
        font1.setBold(True)
        self.pushButton_login.setFont(font1)
        self.pushButton_login.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon = QIcon()
        icon.addFile(u":/images/user-lock.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_login.setIcon(icon)

        self.gridLayout_3.addWidget(self.pushButton_login, 0, 2, 2, 1)


        self.gridLayout_2.addWidget(self.groupBox_login_lineEdit, 1, 2, 1, 1)

        self.groupBox_login_button = QGroupBox(self.page_login)
        self.groupBox_login_button.setObjectName(u"groupBox_login_button")
        self.groupBox_login_button.setMaximumSize(QSize(16777215, 130))
        self.gridLayout_4 = QGridLayout(self.groupBox_login_button)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.horizontalSpacer_7 = QSpacerItem(30, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_7, 2, 2, 1, 1)

        self.pushButton_findpw = QPushButton(self.groupBox_login_button)
        self.pushButton_findpw.setObjectName(u"pushButton_findpw")
        self.pushButton_findpw.setMinimumSize(QSize(0, 40))
        self.pushButton_findpw.setFont(font1)
        self.pushButton_findpw.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon1 = QIcon()
        icon1.addFile(u":/images/key-round.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_findpw.setIcon(icon1)

        self.gridLayout_4.addWidget(self.pushButton_findpw, 2, 3, 1, 1)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_8, 2, 0, 1, 1)

        self.pushButton_signin = QPushButton(self.groupBox_login_button)
        self.pushButton_signin.setObjectName(u"pushButton_signin")
        self.pushButton_signin.setMinimumSize(QSize(0, 40))
        self.pushButton_signin.setFont(font1)
        self.pushButton_signin.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon2 = QIcon()
        icon2.addFile(u":/images/user-plus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_signin.setIcon(icon2)

        self.gridLayout_4.addWidget(self.pushButton_signin, 2, 1, 1, 1)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_9, 2, 4, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_login_button, 2, 1, 1, 2)

        self.stackedWidget.addWidget(self.page_login)
        self.page_menu = QWidget()
        self.page_menu.setObjectName(u"page_menu")
        self.groupBox_menu = QGroupBox(self.page_menu)
        self.groupBox_menu.setObjectName(u"groupBox_menu")
        self.groupBox_menu.setGeometry(QRect(120, 450, 221, 231))
        self.groupBox_menu.setMaximumSize(QSize(16777215, 260))
        self.groupBox_menu.setStyleSheet(u"QGroupBox {\n"
"    border: none;\n"
"}\n"
"")
        self.gridLayout_5 = QGridLayout(self.groupBox_menu)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setHorizontalSpacing(6)
        self.gridLayout_5.setVerticalSpacing(1)
        self.gridLayout_5.setContentsMargins(1, 1, 1, 1)
        self.pushButton_history = QPushButton(self.groupBox_menu)
        self.pushButton_history.setObjectName(u"pushButton_history")
        self.pushButton_history.setMaximumSize(QSize(16777215, 50))
        self.pushButton_history.setFont(font1)
        self.pushButton_history.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon3 = QIcon()
        icon3.addFile(u":/images/history.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_history.setIcon(icon3)

        self.gridLayout_5.addWidget(self.pushButton_history, 1, 0, 1, 1)

        self.pushButton_logout = QPushButton(self.groupBox_menu)
        self.pushButton_logout.setObjectName(u"pushButton_logout")
        self.pushButton_logout.setMaximumSize(QSize(16777215, 50))
        self.pushButton_logout.setFont(font1)
        self.pushButton_logout.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon4 = QIcon()
        icon4.addFile(u":/images/log-out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_logout.setIcon(icon4)

        self.gridLayout_5.addWidget(self.pushButton_logout, 2, 0, 1, 1)

        self.pushButton_exit = QPushButton(self.groupBox_menu)
        self.pushButton_exit.setObjectName(u"pushButton_exit")
        self.pushButton_exit.setMaximumSize(QSize(16777215, 50))
        self.pushButton_exit.setFont(font1)
        self.pushButton_exit.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon5 = QIcon()
        icon5.addFile(u":/images/power.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_exit.setIcon(icon5)

        self.gridLayout_5.addWidget(self.pushButton_exit, 3, 0, 1, 1)

        self.pushButton_test = QPushButton(self.groupBox_menu)
        self.pushButton_test.setObjectName(u"pushButton_test")
        self.pushButton_test.setMaximumSize(QSize(16777215, 50))
        self.pushButton_test.setFont(font1)
        self.pushButton_test.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}\n"
"")
        icon6 = QIcon()
        icon6.addFile(u":/images/camera.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_test.setIcon(icon6)

        self.gridLayout_5.addWidget(self.pushButton_test, 0, 0, 1, 1)

        self.gridLayout_6 = QGridLayout(self.page_menu)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.label_menu_img = QLabel(self.page_menu)
        self.label_menu_img.setObjectName(u"label_menu_img")
        self.label_menu_img.setStyleSheet(u"")
        self.label_menu_img.setPixmap(QPixmap(u":/images/loginmain.png"))
        self.label_menu_img.setScaledContents(True)

        self.gridLayout_6.addWidget(self.label_menu_img, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.page_menu)
        self.label_menu_img.raise_()
        self.groupBox_menu.raise_()
        self.page_add = QWidget()
        self.page_add.setObjectName(u"page_add")
        self.graphicsView_addimg = QGraphicsView(self.page_add)
        self.graphicsView_addimg.setObjectName(u"graphicsView_addimg")
        self.graphicsView_addimg.setGeometry(QRect(25, 30, 400, 470))
        self.graphicsView_addimg.setStyleSheet(u"background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:0, stop:0 rgba(255, 255, 255, 161));\n"
"border: none;\n"
"")
        self.graphicsView_addimg.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphicsView_addimg.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.label = QLabel(self.page_add)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 570, 16, 18))
        self.groupBox_add_button = QGroupBox(self.page_add)
        self.groupBox_add_button.setObjectName(u"groupBox_add_button")
        self.groupBox_add_button.setGeometry(QRect(25, 630, 400, 80))
        font2 = QFont()
        font2.setPointSize(13)
        self.groupBox_add_button.setFont(font2)
        self.groupBox_add_button.setStyleSheet(u"QGroupBox {\n"
"    border: none;                    /* \ud14c\ub450\ub9ac \uc81c\uac70 */\n"
"    margin-top: 30px;\n"
"    color: black;                   /* \ub0b4\ubd80 \uae00\uc528\ub294 \uac80\uc815 */\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"    background-color: transparent;  /* \ud0c0\uc774\ud2c0 \ubc30\uacbd \ud22c\uba85 */\n"
"    color: black;                   /* \ud0c0\uc774\ud2c0 \uae00\uc528\ub294 \uac80\uc815 */\n"
"    padding: 0 5px;\n"
"}")
        self.gridLayout_7 = QGridLayout(self.groupBox_add_button)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setHorizontalSpacing(1)
        self.gridLayout_7.setVerticalSpacing(8)
        self.gridLayout_7.setContentsMargins(1, 1, 1, 1)
        self.horizontalSpacer_4 = QSpacerItem(5, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_4, 0, 1, 1, 1)

        self.pushButton_addnew = QPushButton(self.groupBox_add_button)
        self.pushButton_addnew.setObjectName(u"pushButton_addnew")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_addnew.sizePolicy().hasHeightForWidth())
        self.pushButton_addnew.setSizePolicy(sizePolicy)
        self.pushButton_addnew.setMinimumSize(QSize(50, 50))
        self.pushButton_addnew.setFont(font1)
        self.pushButton_addnew.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon7 = QIcon()
        icon7.addFile(u":/images/image-plus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_addnew.setIcon(icon7)

        self.gridLayout_7.addWidget(self.pushButton_addnew, 0, 6, 1, 1)

        self.pushButton_check = QPushButton(self.groupBox_add_button)
        self.pushButton_check.setObjectName(u"pushButton_check")
        sizePolicy.setHeightForWidth(self.pushButton_check.sizePolicy().hasHeightForWidth())
        self.pushButton_check.setSizePolicy(sizePolicy)
        self.pushButton_check.setMinimumSize(QSize(50, 50))
        self.pushButton_check.setFont(font1)
        self.pushButton_check.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color: #FFECD2;\n"
"    color: #777777;\n"
"}\n"
"")
        icon8 = QIcon()
        icon8.addFile(u":/images/search-check.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_check.setIcon(icon8)

        self.gridLayout_7.addWidget(self.pushButton_check, 0, 4, 1, 1)

        self.pushButton_save = QPushButton(self.groupBox_add_button)
        self.pushButton_save.setObjectName(u"pushButton_save")
        sizePolicy.setHeightForWidth(self.pushButton_save.sizePolicy().hasHeightForWidth())
        self.pushButton_save.setSizePolicy(sizePolicy)
        self.pushButton_save.setMinimumSize(QSize(50, 50))
        self.pushButton_save.setFont(font1)
        self.pushButton_save.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color: #FFECD2;\n"
"    color: #777777;\n"
"}")
        icon9 = QIcon()
        icon9.addFile(u":/images/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_save.setIcon(icon9)

        self.gridLayout_7.addWidget(self.pushButton_save, 0, 2, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(5, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_5, 0, 3, 1, 1)

        self.pushButton_back = QPushButton(self.groupBox_add_button)
        self.pushButton_back.setObjectName(u"pushButton_back")
        sizePolicy.setHeightForWidth(self.pushButton_back.sizePolicy().hasHeightForWidth())
        self.pushButton_back.setSizePolicy(sizePolicy)
        self.pushButton_back.setMinimumSize(QSize(50, 50))
        self.pushButton_back.setFont(font1)
        self.pushButton_back.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon10 = QIcon()
        icon10.addFile(u":/images/arrow-left.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_back.setIcon(icon10)

        self.gridLayout_7.addWidget(self.pushButton_back, 0, 0, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(5, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_6, 0, 5, 1, 1)

        self.label_4 = QLabel(self.page_add)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(10, 630, 16, 18))
        self.groupBox = QGroupBox(self.page_add)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(25, 520, 400, 80))
        self.groupBox.setFont(font2)
        self.groupBox.setStyleSheet(u"QGroupBox {\n"
"    border: none;                    /* \ud14c\ub450\ub9ac \uc81c\uac70 */\n"
"    margin-top: 30px;\n"
"    color: black;                   /* \ub0b4\ubd80 \uae00\uc528\ub294 \uac80\uc815 */\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"    background-color: transparent;  /* \ud0c0\uc774\ud2c0 \ubc30\uacbd \ud22c\uba85 */\n"
"    color: black;                   /* \ud0c0\uc774\ud2c0 \uae00\uc528\ub294 \uac80\uc815 */\n"
"    padding: 0 5px;\n"
"}")
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.gridLayout_11 = QGridLayout(self.groupBox)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.gridLayout_11.setHorizontalSpacing(1)
        self.gridLayout_11.setContentsMargins(1, 1, 1, 1)
        self.pushButton_shoot = QPushButton(self.groupBox)
        self.pushButton_shoot.setObjectName(u"pushButton_shoot")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(40)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton_shoot.sizePolicy().hasHeightForWidth())
        self.pushButton_shoot.setSizePolicy(sizePolicy1)
        self.pushButton_shoot.setMinimumSize(QSize(40, 50))
        self.pushButton_shoot.setMaximumSize(QSize(120, 16777215))
        self.pushButton_shoot.setFont(font1)
        self.pushButton_shoot.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color: #FFECD2;\n"
"    color: #777777;\n"
"}\n"
"")
        self.pushButton_shoot.setIcon(icon6)

        self.gridLayout_11.addWidget(self.pushButton_shoot, 0, 0, 1, 1)

        self.pushButton_save0 = QPushButton(self.groupBox)
        self.pushButton_save0.setObjectName(u"pushButton_save0")
        sizePolicy1.setHeightForWidth(self.pushButton_save0.sizePolicy().hasHeightForWidth())
        self.pushButton_save0.setSizePolicy(sizePolicy1)
        self.pushButton_save0.setMinimumSize(QSize(40, 50))
        self.pushButton_save0.setFont(font1)
        self.pushButton_save0.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color: #FFECD2;\n"
"    color: #777777;\n"
"}\n"
"")
        self.pushButton_save0.setIcon(icon9)

        self.gridLayout_11.addWidget(self.pushButton_save0, 0, 2, 1, 1)

        self.pushButton_cameraon = QPushButton(self.groupBox)
        self.pushButton_cameraon.setObjectName(u"pushButton_cameraon")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy2.setHorizontalStretch(40)
        sizePolicy2.setVerticalStretch(50)
        sizePolicy2.setHeightForWidth(self.pushButton_cameraon.sizePolicy().hasHeightForWidth())
        self.pushButton_cameraon.setSizePolicy(sizePolicy2)
        self.pushButton_cameraon.setMinimumSize(QSize(40, 50))
        self.pushButton_cameraon.setFont(font1)
        self.pushButton_cameraon.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        icon11 = QIcon()
        icon11.addFile(u":/images/video.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_cameraon.setIcon(icon11)

        self.gridLayout_11.addWidget(self.pushButton_cameraon, 0, 6, 1, 1)

        self.pushButton_check0 = QPushButton(self.groupBox)
        self.pushButton_check0.setObjectName(u"pushButton_check0")
        sizePolicy1.setHeightForWidth(self.pushButton_check0.sizePolicy().hasHeightForWidth())
        self.pushButton_check0.setSizePolicy(sizePolicy1)
        self.pushButton_check0.setMinimumSize(QSize(40, 50))
        self.pushButton_check0.setFont(font1)
        self.pushButton_check0.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color: #FFECD2;\n"
"    color: #777777;\n"
"}\n"
"")
        self.pushButton_check0.setIcon(icon8)

        self.gridLayout_11.addWidget(self.pushButton_check0, 0, 4, 1, 1)

        self.horizontalSpacer = QSpacerItem(5, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_11.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(5, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_11.addItem(self.horizontalSpacer_2, 0, 5, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(5, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_11.addItem(self.horizontalSpacer_3, 0, 3, 1, 1)

        self.gridLayout_8 = QGridLayout(self.page_add)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.IMG = QLabel(self.page_add)
        self.IMG.setObjectName(u"IMG")
        self.IMG.setPixmap(QPixmap(u":/images/mainda.png"))
        self.IMG.setScaledContents(True)

        self.gridLayout_8.addWidget(self.IMG, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.page_add)
        self.IMG.raise_()
        self.groupBox.raise_()
        self.groupBox_add_button.raise_()
        self.label.raise_()
        self.label_4.raise_()
        self.graphicsView_addimg.raise_()
        self.page_history = QWidget()
        self.page_history.setObjectName(u"page_history")
        self.tableWidget_history = QTableWidget(self.page_history)
        self.tableWidget_history.setObjectName(u"tableWidget_history")
        self.tableWidget_history.setGeometry(QRect(25, 30, 400, 600))
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.tableWidget_history.sizePolicy().hasHeightForWidth())
        self.tableWidget_history.setSizePolicy(sizePolicy3)
        self.tableWidget_history.setMinimumSize(QSize(0, 600))
        self.tableWidget_history.setMaximumSize(QSize(400, 600))
        font3 = QFont()
        font3.setPointSize(12)
        self.tableWidget_history.setFont(font3)
        self.tableWidget_history.setStyleSheet(u"")
        self.groupBox_history_button = QGroupBox(self.page_history)
        self.groupBox_history_button.setObjectName(u"groupBox_history_button")
        self.groupBox_history_button.setGeometry(QRect(35, 660, 391, 68))
        self.groupBox_history_button.setStyleSheet(u"QGroupBox{\n"
"border:none;\n"
"\n"
"}")
        self.gridLayout_10 = QGridLayout(self.groupBox_history_button)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.pushButton_back_2 = QPushButton(self.groupBox_history_button)
        self.pushButton_back_2.setObjectName(u"pushButton_back_2")
        self.pushButton_back_2.setMinimumSize(QSize(0, 50))
        self.pushButton_back_2.setFont(font1)
        self.pushButton_back_2.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")
        self.pushButton_back_2.setIcon(icon10)

        self.gridLayout_10.addWidget(self.pushButton_back_2, 0, 0, 1, 1)

        self.label_6 = QLabel(self.page_history)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(9, 9, 432, 736))
        self.label_6.setPixmap(QPixmap(u":/images/loginmain.png"))
        self.label_6.setScaledContents(True)
        self.stackedWidget.addWidget(self.page_history)
        self.label_6.raise_()
        self.groupBox_history_button.raise_()
        self.tableWidget_history.raise_()
        self.page_signup = QWidget()
        self.page_signup.setObjectName(u"page_signup")
        self.textEdit = QTextEdit(self.page_signup)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(40, 460, 371, 181))
        self.textEdit.setStyleSheet(u"QTextEdit {\n"
"    background: transparent;\n"
"    border: none;\n"
"    color: #8c5a2b;\n"
"    font-weight: bold;\n"
"    font-size: 12pt;\n"
"}\n"
"")
        self.label_7 = QLabel(self.page_signup)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(10, 9, 432, 736))
        self.label_7.setPixmap(QPixmap(u":/images/mainda.png"))
        self.label_7.setScaledContents(True)
        self.label_8 = QLabel(self.page_signup)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(40, 30, 381, 51))
        font4 = QFont()
        font4.setPointSize(19)
        font4.setBold(True)
        self.label_8.setFont(font4)
        self.label_8.setStyleSheet(u"QLabel {\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.sss = QLabel(self.page_signup)
        self.sss.setObjectName(u"sss")
        self.sss.setGeometry(QRect(30, 90, 394, 1))
        self.sss.setMaximumSize(QSize(16777215, 1))
        self.sss.setStyleSheet(u"background-color:#838383;")
        self.groupBox_2 = QGroupBox(self.page_signup)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(35, 100, 380, 55))
        self.gridLayout_9 = QGridLayout(self.groupBox_2)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.gridLayout_9.setContentsMargins(8, 8, 8, 8)
        self.sign_id_lb = QLabel(self.groupBox_2)
        self.sign_id_lb.setObjectName(u"sign_id_lb")
        self.sign_id_lb.setStyleSheet(u"QLabel {\n"
"    background-image: url(:/images/user.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"}\n"
"")

        self.gridLayout_9.addWidget(self.sign_id_lb, 0, 0, 1, 1)

        self.label_10 = QLabel(self.groupBox_2)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_9.addWidget(self.label_10, 0, 1, 1, 1)

        self.textEdit_2 = QTextEdit(self.groupBox_2)
        self.textEdit_2.setObjectName(u"textEdit_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.textEdit_2.sizePolicy().hasHeightForWidth())
        self.textEdit_2.setSizePolicy(sizePolicy4)
        self.textEdit_2.setMaximumSize(QSize(230, 16777215))

        self.gridLayout_9.addWidget(self.textEdit_2, 0, 2, 1, 1)

        self.gridLayout_9.setColumnStretch(0, 1)
        self.gridLayout_9.setColumnStretch(1, 1)
        self.gridLayout_9.setColumnStretch(2, 6)
        self.groupBox_3 = QGroupBox(self.page_signup)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(35, 160, 380, 55))
        self.gridLayout_12 = QGridLayout(self.groupBox_3)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.sign_pw_lb = QLabel(self.groupBox_3)
        self.sign_pw_lb.setObjectName(u"sign_pw_lb")
        self.sign_pw_lb.setStyleSheet(u"QLabel {\n"
"    background-image: url(:/images/key-round.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"}\n"
"")

        self.gridLayout_12.addWidget(self.sign_pw_lb, 0, 0, 1, 1)

        self.label_11 = QLabel(self.groupBox_3)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_12.addWidget(self.label_11, 0, 1, 1, 1)

        self.textEdit_3 = QTextEdit(self.groupBox_3)
        self.textEdit_3.setObjectName(u"textEdit_3")
        sizePolicy4.setHeightForWidth(self.textEdit_3.sizePolicy().hasHeightForWidth())
        self.textEdit_3.setSizePolicy(sizePolicy4)
        self.textEdit_3.setMaximumSize(QSize(230, 16777215))

        self.gridLayout_12.addWidget(self.textEdit_3, 0, 2, 1, 1)

        self.gridLayout_12.setColumnStretch(0, 1)
        self.gridLayout_12.setColumnStretch(1, 1)
        self.gridLayout_12.setColumnStretch(2, 6)
        self.groupBox_4 = QGroupBox(self.page_signup)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(35, 220, 380, 55))
        self.gridLayout_13 = QGridLayout(self.groupBox_4)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.label_15 = QLabel(self.groupBox_4)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_13.addWidget(self.label_15, 0, 1, 1, 1)

        self.label_9 = QLabel(self.groupBox_4)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setStyleSheet(u"QLabel {\n"
"    \n"
"	image: url(:/images/phone.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"}\n"
"")

        self.gridLayout_13.addWidget(self.label_9, 0, 0, 1, 1)

        self.textEdit_4 = QTextEdit(self.groupBox_4)
        self.textEdit_4.setObjectName(u"textEdit_4")
        sizePolicy4.setHeightForWidth(self.textEdit_4.sizePolicy().hasHeightForWidth())
        self.textEdit_4.setSizePolicy(sizePolicy4)
        self.textEdit_4.setMaximumSize(QSize(230, 16777215))

        self.gridLayout_13.addWidget(self.textEdit_4, 0, 2, 1, 1)

        self.gridLayout_13.setColumnStretch(0, 1)
        self.gridLayout_13.setColumnStretch(1, 1)
        self.gridLayout_13.setColumnStretch(2, 6)
        self.groupBox_5 = QGroupBox(self.page_signup)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(35, 280, 380, 55))
        self.gridLayout_14 = QGridLayout(self.groupBox_5)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.label_16 = QLabel(self.groupBox_5)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setStyleSheet(u"QLabel {\n"
"    background-image: url(:/images/user-plus.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"}\n"
"")

        self.gridLayout_14.addWidget(self.label_16, 0, 0, 1, 1)

        self.label_17 = QLabel(self.groupBox_5)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setStyleSheet(u"")

        self.gridLayout_14.addWidget(self.label_17, 0, 1, 1, 1)

        self.textEdit_5 = QTextEdit(self.groupBox_5)
        self.textEdit_5.setObjectName(u"textEdit_5")
        sizePolicy4.setHeightForWidth(self.textEdit_5.sizePolicy().hasHeightForWidth())
        self.textEdit_5.setSizePolicy(sizePolicy4)
        self.textEdit_5.setMinimumSize(QSize(230, 0))
        self.textEdit_5.setMaximumSize(QSize(230, 16777215))

        self.gridLayout_14.addWidget(self.textEdit_5, 0, 2, 1, 1)

        self.groupBox_6 = QGroupBox(self.page_signup)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setGeometry(QRect(35, 340, 380, 60))
        self.gridLayout_15 = QGridLayout(self.groupBox_6)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.label_21 = QLabel(self.groupBox_6)
        self.label_21.setObjectName(u"label_21")

        self.gridLayout_15.addWidget(self.label_21, 0, 1, 1, 1)

        self.label_20 = QLabel(self.groupBox_6)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setStyleSheet(u"QLabel {\n"
"    image: url(:/images/addr.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"}\n"
"")

        self.gridLayout_15.addWidget(self.label_20, 0, 0, 1, 1)

        self.textEdit_6 = QTextEdit(self.groupBox_6)
        self.textEdit_6.setObjectName(u"textEdit_6")
        sizePolicy4.setHeightForWidth(self.textEdit_6.sizePolicy().hasHeightForWidth())
        self.textEdit_6.setSizePolicy(sizePolicy4)
        self.textEdit_6.setMinimumSize(QSize(230, 0))
        self.textEdit_6.setMaximumSize(QSize(230, 16777215))

        self.gridLayout_15.addWidget(self.textEdit_6, 0, 2, 1, 1)

        self.groupBox_7 = QGroupBox(self.page_signup)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setGeometry(QRect(35, 405, 380, 60))
        self.gridLayout_16 = QGridLayout(self.groupBox_7)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.label_19 = QLabel(self.groupBox_7)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_16.addWidget(self.label_19, 0, 1, 1, 1)

        self.label_18 = QLabel(self.groupBox_7)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setStyleSheet(u"QLabel {\n"
"    image: url(:/images/email.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"}\n"
"")

        self.gridLayout_16.addWidget(self.label_18, 0, 0, 1, 1)

        self.textEdit_7 = QTextEdit(self.groupBox_7)
        self.textEdit_7.setObjectName(u"textEdit_7")
        sizePolicy4.setHeightForWidth(self.textEdit_7.sizePolicy().hasHeightForWidth())
        self.textEdit_7.setSizePolicy(sizePolicy4)
        self.textEdit_7.setMaximumSize(QSize(230, 16777215))

        self.gridLayout_16.addWidget(self.textEdit_7, 0, 2, 1, 1)

        self.gridLayout_16.setColumnStretch(0, 1)
        self.groupBox_8 = QGroupBox(self.page_signup)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setGeometry(QRect(50, 630, 361, 77))
        self.groupBox_8.setStyleSheet(u"QGroupBox {\n"
"    border: none;\n"
"}\n"
"")
        self.gridLayout_17 = QGridLayout(self.groupBox_8)
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.signup_save_Btn = QPushButton(self.groupBox_8)
        self.signup_save_Btn.setObjectName(u"signup_save_Btn")
        self.signup_save_Btn.setMinimumSize(QSize(150, 50))
        self.signup_save_Btn.setMaximumSize(QSize(150, 16777215))
        self.signup_save_Btn.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")

        self.gridLayout_17.addWidget(self.signup_save_Btn, 0, 2, 1, 1)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_17.addItem(self.horizontalSpacer_10, 0, 1, 1, 1)

        self.signup_back_Btn = QPushButton(self.groupBox_8)
        self.signup_back_Btn.setObjectName(u"signup_back_Btn")
        self.signup_back_Btn.setMinimumSize(QSize(150, 50))
        self.signup_back_Btn.setMaximumSize(QSize(150, 16777215))
        self.signup_back_Btn.setStyleSheet(u"QPushButton {\n"
"    background-color: #FFE0B2;\n"
"    color: #333333;\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 10px 20px;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FFD180;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #FFCC80;\n"
"}")

        self.gridLayout_17.addWidget(self.signup_back_Btn, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.page_signup)
        self.label_7.raise_()
        self.groupBox_7.raise_()
        self.groupBox_6.raise_()
        self.groupBox_5.raise_()
        self.groupBox_4.raise_()
        self.groupBox_3.raise_()
        self.groupBox_2.raise_()
        self.sss.raise_()
        self.label_8.raise_()
        self.textEdit.raise_()
        self.groupBox_8.raise_()

        self.gridLayout.addWidget(self.stackedWidget, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 450, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(4)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_5.setText("")
        self.groupBox_login_lineEdit.setTitle("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"PW", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"ID", None))
        self.pushButton_login.setText(QCoreApplication.translate("MainWindow", u" \ub85c\uadf8\uc778", None))
        self.groupBox_login_button.setTitle("")
        self.pushButton_findpw.setText(QCoreApplication.translate("MainWindow", u" PW \ucc3e\uae30", None))
        self.pushButton_signin.setText(QCoreApplication.translate("MainWindow", u" \ud68c\uc6d0\uac00\uc785", None))
        self.groupBox_menu.setTitle("")
        self.pushButton_history.setText(QCoreApplication.translate("MainWindow", u" \uc870\ud68c\uc774\ub825", None))
        self.pushButton_logout.setText(QCoreApplication.translate("MainWindow", u" \ub85c\uadf8\uc544\uc6c3", None))
        self.pushButton_exit.setText(QCoreApplication.translate("MainWindow", u" \uc885\ub8cc\ud558\uae30", None))
        self.pushButton_test.setText(QCoreApplication.translate("MainWindow", u" \uc0ac\uc9c4\ucc0d\uae30", None))
        self.label_menu_img.setText("")
        self.label.setText("")
        self.groupBox_add_button.setTitle(QCoreApplication.translate("MainWindow", u"\ud30c\uc77c\ucca8\ubd80", None))
        self.pushButton_addnew.setText(QCoreApplication.translate("MainWindow", u"\uc0ac\uc9c4", None))
        self.pushButton_check.setText(QCoreApplication.translate("MainWindow", u"\uac80\uc0ac", None))
        self.pushButton_save.setText(QCoreApplication.translate("MainWindow", u"\uc800\uc7a5", None))
        self.pushButton_back.setText(QCoreApplication.translate("MainWindow", u"\ub4a4\ub85c", None))
        self.label_4.setText("")
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\ucd2c\uc601\ucca8\ubd80", None))
        self.pushButton_shoot.setText(QCoreApplication.translate("MainWindow", u"\ucea1\uccd0", None))
        self.pushButton_save0.setText(QCoreApplication.translate("MainWindow", u"\uc800\uc7a5", None))
        self.pushButton_cameraon.setText(QCoreApplication.translate("MainWindow", u"\ucd2c\uc601", None))
        self.pushButton_check0.setText(QCoreApplication.translate("MainWindow", u"\uac80\uc0ac", None))
        self.IMG.setText("")
        self.groupBox_history_button.setTitle("")
        self.pushButton_back_2.setText(QCoreApplication.translate("MainWindow", u" \ub4a4\ub85c\uac00\uae30", None))
        self.label_6.setText("")
        self.textEdit.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Ubuntu Sans'; font-size:12pt; font-weight:700; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt; font-weight:400;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:400; color:#8c5a2b;\">\ud68c\uc0ac\uba85: \ud504\ub808\uc26c \uc8fc\uc2dd\ud68c\uc0ac</span></p>\n"
"<p align=\"center\" style=\""
                        " margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:400; color:#8c5a2b;\">\ub300\ud45c : \ud55c\ud604\ud76c</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:400; color:#8c5a2b;\">TEL : 010 - 1234 - 5678</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:400; color:#8c5a2b;\">\uc8fc\uc18c : \uad11\uc8fc\uad11\uc5ed\uc2dc \uc0c1\uacf5\ud68c\uc758\uc18c \uad11\uc8fc\uc778\ub825\uac1c\ubc1c\uc6d0</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:400; color:#8c5a2b;\"><br /></p>"
                        "\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:400; color:#8c5a2b;\">\ubcf8 \uc560\ud50c\ub9ac\ucf00\uc774\uc158\uc758 \ucd5c\uc885 \ud65c\uc6a9 \uc5ec\ubd80\uc5d0 \ub300\ud55c </span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:400; color:#8c5a2b;\">\ucc45\uc784\uacfc \ud310\ub2e8\uc740 \uc0ac\uc6a9\uc790 \ubcf8\uc778\uc5d0\uac8c \uc788\uc2b5\ub2c8\ub2e4.</span></p></body></html>", None))
        self.label_7.setText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"\ud68c\uc6d0\uac00\uc785", None))
        self.sss.setText("")
        self.groupBox_2.setTitle("")
        self.sign_id_lb.setText("")
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"ID :", None))
        self.groupBox_3.setTitle("")
        self.sign_pw_lb.setText("")
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"PW :", None))
        self.groupBox_4.setTitle("")
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Phone :", None))
        self.label_9.setText("")
        self.groupBox_5.setTitle("")
        self.label_16.setText("")
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Name :", None))
        self.groupBox_6.setTitle("")
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Address :</p></body></html>", None))
        self.label_20.setText("")
        self.groupBox_7.setTitle("")
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"    Email :", None))
        self.label_18.setText("")
        self.groupBox_8.setTitle("")
        self.signup_save_Btn.setText(QCoreApplication.translate("MainWindow", u"\uc644\ub8cc", None))
        self.signup_back_Btn.setText(QCoreApplication.translate("MainWindow", u"\ub098\uac00\uae30", None))
    # retranslateUi

