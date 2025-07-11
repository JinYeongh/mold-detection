#include <vector>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <string>
#include <thread>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <mysql/mysql.h>
#include "cJSON.h"

#define PORT 12345
#define BUFFER_SIZE 4096

// DB 접속 정보
const char* DB_HOST    = "localhost";
const char* DB_USER    = "root";
const char* DB_PASS    = "1234";
const char* DB_NAME    = "FRESH";

// 업로드 루트 디렉토리
const char* UPLOAD_DIR = "uploads";

// ----------------- 로그인 검증 함수 -----------------
bool verify_login(const std::string& phone,
                  const std::string& password,
                  std::string& err_reason)
{
    MYSQL* conn = mysql_init(nullptr);
    if (!conn) {
        err_reason = "mysql_init 실패";
        return false;
    }
    if (!mysql_real_connect(conn, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, nullptr, 0)) {
        err_reason = std::string("DB 연결 실패: ") + mysql_error(conn);
        mysql_close(conn);
        return false;
    }

    std::string query =
        "SELECT PASSWORD_HASH FROM USER_INFO WHERE PHONENUMBER='" + phone + "'";
    if (mysql_query(conn, query.c_str()) != 0) {
        err_reason = std::string("쿼리 실패: ") + mysql_error(conn);
        mysql_close(conn);
        return false;
    }

    MYSQL_RES* res = mysql_store_result(conn);
    if (!res || mysql_num_rows(res) == 0) {
        err_reason = "가입되지 않은 전화번호";
        if (res) mysql_free_result(res);
        mysql_close(conn);
        return false;
    }
    MYSQL_ROW row = mysql_fetch_row(res);
    const char* db_pw = row[0];
    bool ok = (password == db_pw);
    if (!ok) {
        err_reason = "비밀번호 불일치";
    }
    mysql_free_result(res);
    mysql_close(conn);
    return ok;
}

// ----------------- 클라이언트 핸들러 -----------------
void handle_client(int client_sock) {
    // 1) 헤더(JSON)만 '\n'까지 읽기
    std::string hdr;
    char ch;
    while (true) {
        int r = recv(client_sock, &ch, 1, 0);
        if (r <= 0) { close(client_sock); return; }
        if (ch == '\n') break;
        hdr.push_back(ch);
    }
    cJSON* root = cJSON_Parse(hdr.c_str());
    if (!root) { close(client_sock); return; }
    const char* proto = cJSON_GetObjectItem(root, "protocol")->valuestring;

    // --- 로그인 처리 (1_0) ---
    if (strcmp(proto, "1_0") == 0) {
        std::string phone = cJSON_GetObjectItem(root, "phonenumber")->valuestring;
        std::string pw    = cJSON_GetObjectItem(root, "password")->valuestring;
        std::string reason;
        bool ok = verify_login(phone, pw, reason);

        cJSON* res = cJSON_CreateObject();
        cJSON_AddStringToObject(res, "protocol", ok ? "1_1" : "1_2");
        cJSON_AddStringToObject(res, "result",   ok ? "success" : "fail");
        cJSON_AddStringToObject(res, "error",    ok ? "" : reason.c_str());
        char* resp_str = cJSON_PrintUnformatted(res);
        send(client_sock, resp_str, strlen(resp_str), 0);
        cJSON_free(resp_str);
        cJSON_Delete(res);
    }

    else if (strcmp(proto, "2_0") == 0) {
    const char* phone = cJSON_GetObjectItem(root, "phonenumber")->valuestring;
    const char* password = cJSON_GetObjectItem(root, "password")->valuestring;

    MYSQL* conn = mysql_init(NULL);
    mysql_real_connect(conn, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0);

    std::string check_query = "SELECT * FROM USER_INFO WHERE PHONENUMBER='" + std::string(phone) + "'";
    mysql_query(conn, check_query.c_str());
    MYSQL_RES* res = mysql_store_result(conn);

    cJSON* res_json = cJSON_CreateObject();
    cJSON_AddStringToObject(res_json, "protocol", "2_1");

    if (mysql_num_rows(res) > 0) {
        cJSON_AddStringToObject(res_json, "result", "duplicate");
    } else {
        std::string insert_query = "INSERT INTO USER_INFO VALUES('" + std::string(phone) + "', '" + std::string(password) + "')";
        if (mysql_query(conn, insert_query.c_str()) == 0) {
            cJSON_AddStringToObject(res_json, "result", "success");
        } else {
            cJSON_AddStringToObject(res_json, "result", "fail");
        }
    }

    char* json_str = cJSON_PrintUnformatted(res_json);
    send(client_sock, json_str, strlen(json_str), 0);

    free(json_str);
    cJSON_Delete(res_json);
    mysql_free_result(res);
    mysql_close(conn);
}


    // --- 이력조회 처리 (100_0) ---
    else if (strcmp(proto, "100_0") == 0) {
        std::string phone = cJSON_GetObjectItem(root, "phonenumber")->valuestring;
        MYSQL* conn = mysql_init(nullptr);
        if (conn && mysql_real_connect(conn, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, nullptr, 0)) {
            std::string q =
                "SELECT NUM, ADDTIME, STATUS, FILENAME "
                "FROM SEARCH_INFO "
                "WHERE PHONENUMBER='" + phone + "' ORDER BY ADDTIME DESC";
            if (mysql_query(conn, q.c_str()) == 0) {
                MYSQL_RES* dbres = mysql_store_result(conn);
                cJSON* res = cJSON_CreateObject();
                cJSON_AddStringToObject(res, "protocol", "100_1");
                cJSON* arr = cJSON_CreateArray();
                MYSQL_ROW row;
                while ((row = mysql_fetch_row(dbres))) {
                    cJSON* obj = cJSON_CreateObject();
                    cJSON_AddNumberToObject(obj, "num", std::atoi(row[0]));
                    cJSON_AddStringToObject(obj, "addtime", row[1] ? row[1] : "");
                    cJSON_AddNumberToObject(obj, "status", std::atoi(row[2]));
                    cJSON_AddStringToObject(obj, "filename", row[3] ? row[3] : "");
                    cJSON_AddItemToArray(arr, obj);
                }
                cJSON_AddItemToObject(res, "records", arr);
                char* resp_str = cJSON_PrintUnformatted(res);
                send(client_sock, resp_str, strlen(resp_str), 0);
                cJSON_free(resp_str);
                cJSON_Delete(res);
                mysql_free_result(dbres);
            }
            mysql_close(conn);
        }
    }
    // --- 이미지 저장 처리 (20_0) ---
    else if (strcmp(proto, "20_0") == 0) {
        std::string phone    = cJSON_GetObjectItem(root, "phonenumber")->valuestring;
        std::string result   = cJSON_GetObjectItem(root, "result")->valuestring;
        std::string filename = cJSON_GetObjectItem(root, "filename")->valuestring;
        int filesize         = cJSON_GetObjectItem(root, "filesize")->valueint;

        // 사용자별 디렉토리 생성
        std::string user_dir = std::string(UPLOAD_DIR) + "/" + phone;
        mkdir(UPLOAD_DIR, 0755);
        mkdir(user_dir.c_str(), 0755);

        // 타임스탬프 생성
        std::time_t t = std::time(nullptr);
        char ts[16];
        std::strftime(ts, sizeof(ts), "%Y%m%d%H%M%S", std::localtime(&t));
        std::string new_filename = std::string(ts) + "_" + filename;

        // 이미지 데이터 수신
        std::vector<char> img_buf(filesize);
        int recvd = 0;
        while (recvd < filesize) {
            int r = recv(client_sock, img_buf.data() + recvd,
                         filesize - recvd, 0);
            if (r <= 0) break;
            recvd += r;
        }

        // 파일 저장
        std::string fullpath = user_dir + "/" + new_filename;
        FILE* fp = fopen(fullpath.c_str(), "wb");
        if (fp) { fwrite(img_buf.data(), 1, recvd, fp); fclose(fp); }

        // DB 기록
        MYSQL* conn = mysql_init(nullptr);
        if (conn && mysql_real_connect(conn, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, nullptr, 0)) {
            int status = (result == "normal" ? 0 : 1);
            std::string ins =
                "INSERT INTO SEARCH_INFO (PHONENUMBER, STATUS, FILENAME, ADDTIME) VALUES ('" +
                phone + "', " + std::to_string(status) + ", '" + new_filename + "', NOW())";
            if (mysql_query(conn, ins.c_str()) != 0) {
                std::cerr << "[ERROR] INSERT 실패: " << mysql_error(conn) << std::endl;
            }
            mysql_close(conn);
        }

        // 응답 전송
        cJSON* res = cJSON_CreateObject();
        cJSON_AddStringToObject(res, "protocol", "20_1");
        cJSON_AddStringToObject(res, "result",   "success");
        char* rstr = cJSON_PrintUnformatted(res);
        send(client_sock, rstr, strlen(rstr), 0);
        cJSON_free(rstr);
        cJSON_Delete(res);
    }
    // --- 이미지 전송 처리 (30_0) ---
    else if (strcmp(proto, "30_0") == 0) {
        std::string phone    = cJSON_GetObjectItem(root, "phonenumber")->valuestring;
        std::string filename = cJSON_GetObjectItem(root, "filename")->valuestring;
        std::string path     = std::string(UPLOAD_DIR) + "/" + phone + "/" + filename;
        FILE* fp = fopen(path.c_str(), "rb");
        if (fp) {
            char buf[BUFFER_SIZE];
            size_t n;
            while ((n = fread(buf, 1, BUFFER_SIZE, fp)) > 0) {
                send(client_sock, buf, n, 0);
            }
            fclose(fp);
        } else {
            std::cerr << "[ERROR] 파일 열기 실패: " << path << std::endl;
        }
    }

    cJSON_Delete(root);
    close(client_sock);
}

int main() {
    mkdir(UPLOAD_DIR, 0755);
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serv_addr{};
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port        = htons(PORT);

    bind(server_sock, (sockaddr*)&serv_addr, sizeof(serv_addr));
    listen(server_sock, 5);
    std::cout << "서버 실행 중 (포트 " << PORT << ")..." << std::endl;

    while (true) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);
        int client_sock = accept(server_sock, (sockaddr*)&client_addr, &len);
        std::thread(handle_client, client_sock).detach();
    }

    close(server_sock);
    return 0;
}
