import requests

# Hàm gọi API Germini để lấy khuyến nghị chữa bệnh
def get_chatbot_response(user_input):
    try:
        api_url = "https://aistudio.google.com/apikey"  # Thay bằng URL chính xác của API
        headers = {
            "Authorization": "Bearer AIzaSyDCQFy8uoZDLiIxV6-Jx2smDmkUZezVQ4M",  # Thay token này bằng token hợp lệ
            "Content-Type": "text/html"  # Thay vì application/json, dùng text/html cho dữ liệu HTML
        }
        
        # Chuyển dữ liệu thành HTML (ví dụ, bao bọc câu hỏi vào thẻ <html>)
        payload = f"<html><body><p>{user_input}</p></body></html>"

        # Sử dụng data để gửi dữ liệu dưới dạng HTML
        response = requests.post(api_url, headers=headers, data=payload)

        # Kiểm tra mã trạng thái HTTP
        if response.status_code == 200:
            try:
                response_text = response.text  # Trả về nội dung HTML
                return response_text
            except Exception as e:
                return f"Lỗi khi xử lý phản hồi: {e}"
        else:
            return f"Lỗi API: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Lỗi kết nối: {e}"

# Ví dụ gọi hàm
user_input = "Tôi bị đau đầu, phải làm sao?"
response = get_chatbot_response(user_input)
print(response)
