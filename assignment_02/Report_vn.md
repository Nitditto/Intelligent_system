# Báo cáo Bài tập 2: Dự đoán Giá Bất động sản Việt Nam

Báo cáo này bao gồm hai phần chính:
1. **Phần 1:** Bản dịch tiếng Việt hoàn chỉnh từ báo cáo gốc [Report.md](file:///e:/CODING/IS_AI/intelligent_system_assignments/assignment_02/Report.md), giữ nguyên toàn bộ các thông tin và thông số kỹ thuật.
2. **Phần 2:** Giải thích chi tiết từng cell một của hai file Notebook: [Training_1.ipynb](file:///e:/CODING/IS_AI/intelligent_system_assignments/assignment_02/Training_1.ipynb) và [Visualization_2.ipynb](file:///e:/CODING/IS_AI/intelligent_system_assignments/assignment_02/Visualization_2.ipynb), giải nghĩa mã nguồn, các thông số, ý nghĩa của các biểu đồ và số liệu tương ứng.

---

# PHẦN 1: BÁO CÁO DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM (BẢN DỊCH TIẾNG VIỆT)

## 1. Giới thiệu
Báo cáo này tài liệu hóa quá trình thiết kế, triển khai và đánh giá một hệ thống thông minh nhằm dự đoán giá thị trường của các bất động sản nhà ở tại Việt Nam dựa trên các thuộc tính tin đăng. Hệ thống sử dụng một số mô hình hồi quy Học máy (Machine Learning) truyền thống để chuyển đổi dữ liệu thô từ tin đăng (diện tích, vị trí, cấu trúc, tình trạng pháp lý, nội thất) thành một ước tính giá liên tục, đóng vai trò trợ giúp định giá sơ bộ cho người mua, người bán và môi giới bất động sản.

## 2. Định nghĩa hệ thống
### Tuyên bố hệ thống (System Statement)
Hệ thống thông minh này là một trợ lý ước tính giá. Hệ thống nhận đầu vào là thông tin thô của một tin đăng bất động sản — bao gồm các thông số vật lý (diện tích, mặt tiền, chiều rộng ngõ), cấu trúc (số tầng, số phòng ngủ, số phòng tắm), vị trí (địa chỉ/quận huyện) và các thuộc tính định tính (hướng nhà, tình trạng pháp lý, tình trạng nội thất) — và chuyển đổi chúng thành một vector đặc trưng số (Feature Vector). Một mô hình hồi quy đã qua huấn luyện sau đó sẽ ánh xạ vector này thành một mức giá dự đoán (đơn vị: tỷ VNĐ), cung cấp cho người dùng một điểm tham chiếu nhanh chóng và dựa trên dữ liệu thực tế trước khi đàm phán giao dịch.

### Sơ đồ hệ thống (System Diagram)
```mermaid
flowchart LR
    A["Đầu vào:<br/>Dữ liệu thô từ<br/>Tin đăng BĐS"] --> B["Biểu diễn:<br/>Vector đặc trưng<br/>(Numerics chuẩn hóa +<br/>Categoricals One-Hot)"]
    B --> C["Mô hình:<br/>Bộ hồi quy ML<br/>đã huấn luyện"]
    C --> D["Dự đoán:<br/>Giá trị liên tục<br/>(Tỷ VNĐ)"]
```

## 3. Định nghĩa bài toán
*   **Hệ thống giải quyết vấn đề thực tế nào?** Nó giải quyết khó khăn trong việc ước tính giá thị trường hợp lý cho một bất động sản mà không cần phải so sánh thủ công hàng chục tin đăng tương tự, giúp người mua tránh mua hớ và người bán/môi giới định giá cạnh tranh hơn.
*   **Hệ thống nhận thông tin gì?** Các thuộc tính bất động sản: `Area` (Diện tích), `Frontage` (Mặt tiền), `Access Road` (Chiều rộng ngõ), `Floors` (Số tầng), `Bedrooms` (Phòng ngủ), `Bathrooms` (Phòng tắm), `House direction` (Hướng nhà), `Balcony direction` (Hướng ban công), `Legal status` (Pháp lý), `Furniture state` (Nội thất) và một chuỗi văn bản tự do `Address` (từ đó trích xuất ra Quận/Huyện `District`).
*   **Thông tin được biểu diễn nội bộ như thế nào?** Dưới dạng một vector đặc trưng số đã được chuẩn hóa và mã hóa one-hot (367 chiều sau khi mã hóa); biến mục tiêu `Price` (Giá) được mô hình hóa trong không gian log (`log1p(Price)`) để giảm độ lệch phải (skewness) rất lớn của nó, sau đó được chuyển đổi ngược về đơn vị tỷ VNĐ để đánh giá.
*   **Mô hình học được gì?** Mối quan hệ thống kê giữa các thuộc tính của bất động sản/vị trí và giá trị log tương ứng của giá, được học từ khoảng 30.000 tin đăng lịch sử.
*   **Quyết định hoặc dự đoán nào được tạo ra?** Một dự đoán giá liên tục (đơn vị: tỷ VNĐ), không phải là phân loại danh mục.
*   **Ai hoặc cái gì sử dụng dự đoán này?** Người mua, người bán và môi giới bất động sản thông qua ứng dụng web đã triển khai (`app/`) để tham khảo nhanh khi đàm phán giá.

## 4. Tập dữ liệu
*   **Nguồn dữ liệu:** [House Price Prediction Dataset Vietnam - 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024) (Kaggle, tác giả: `nguyentiennhan`) — tập hợp khoảng 30.000 tin đăng bất động sản nhà ở tại Việt Nam, lưu cục bộ dưới tên `vietnam_housing_dataset.csv`.
*   **Hiện tượng thực tế nào được biểu diễn?** Hành vi định giá của thị trường bất động sản nhà ở tại Việt Nam, bị chi phối bởi kích thước, vị trí và các thuộc tính chất lượng của bất động sản.
*   **Một mẫu quan sát (observation) là gì?** Một tin đăng bất động sản đơn lẻ (một dòng trong bảng dữ liệu = một căn nhà/căn hộ được rao bán).
*   **Các đặc trưng (features) là gì?** Address, Area, Frontage, Access Road, House direction, Balcony direction, Floors, Bedrooms, Bathrooms, Legal status, Furniture state (11 cột thô).
*   **Biến mục tiêu (target) là gì?** `Price` (đơn vị: tỷ VNĐ).
*   **Biến mục tiêu là số hay phân loại?** Số (liên tục).
*   **Đây là bài toán hồi quy hay phân loại?** Hồi quy (Regression).
*   **Có bao nhiêu mẫu quan sát?** 30.229 tin đăng (sau khi loại bỏ các dòng thiếu biến mục tiêu `Price`).
*   **Có bao nhiêu đặc trưng đầu vào?** 11 cột thô; sau khi mã hóa one-hot các trường phân loại (bao gồm 342 quận/huyện duy nhất được trích xuất từ địa chỉ `Address`), đầu vào của mô hình có 367 chiều.
*   **Các đặc trưng số gồm những gì?** Area, Frontage, Access Road, Floors, Bedrooms, Bathrooms.
*   **Các đặc trưng phân loại gồm những gì?** District (trích xuất từ Address), House direction, Balcony direction, Legal status, Furniture state.

## 5. Biểu diễn dữ liệu

| Đặc trưng | Loại dữ liệu | Cách biểu diễn | Ý nghĩa thực tế |
| --- | --- | --- | --- |
| Area | Số | Giá trị thực | Diện tích sàn của bất động sản (m²) |
| Frontage | Số | Giá trị thực | Chiều rộng mặt tiền tiếp giáp đường (m) |
| Access Road | Số | Giá trị thực | Chiều rộng của ngõ/đường đi vào nhà (m) |
| Floors | Số | Số nguyên | Số tầng |
| Bedrooms | Số | Số nguyên | Số phòng ngủ |
| Bathrooms | Số | Số nguyên | Số phòng tắm |
| District (từ Address) | Phân loại | Mã hóa One-hot (342 danh mục) | Quận/Huyện hành chính trích xuất từ chuỗi địa chỉ thô |
| House direction | Phân loại | Mã hóa One-hot | Hướng la bàn của ngôi nhà (ví dụ: Đông - Nam) |
| Balcony direction | Phân loại | Mã hóa One-hot | Hướng la bàn của ban công |
| Legal status | Phân loại | Mã hóa One-hot | Tình trạng giấy tờ sở hữu (Sổ đỏ/Sổ hồng, HĐMB...) |
| Furniture state | Phân loại | Mã hóa One-hot | Mức độ trang bị nội thất (Đầy đủ, Cơ bản...) |

*(Lưu ý: Các giá trị số bị thiếu được điền bằng giá trị trung vị (median) của cột tương ứng; các giá trị phân loại bị thiếu được điền bằng chuỗi `'Unknown'` trước khi mã hóa. Tỷ lệ khuyết dữ liệu khá cao ở một số cột, ví dụ: Hướng ban công thiếu 82.6%, Hướng nhà thiếu 70.3%, Tình trạng nội thất thiếu 46.7%. Các cột phân loại được chuyển đổi bằng `pd.get_dummies(drop_first=True)`. Biến mục tiêu `Price` được biến đổi bằng `log1p` trước khi huấn luyện và nghịch đảo bằng `expm1` trước khi tính toán các độ đo đánh giá. Toàn bộ 367 chiều đầu vào được chuẩn hóa bằng `StandardScaler` khớp (fit) trên tập huấn luyện).*

## 6. Các phương pháp Học máy truyền thống

### 1. Hồi quy tuyến tính (Linear Regression)
*   **Đầu vào nhận được:** Vector đặc trưng đã chuẩn hóa và mã hóa one-hot; biến mục tiêu trong không gian log-price.
*   **Mối quan hệ cần học:** Mối quan hệ tuyến tính giữa vector đặc trưng và log(Price).
*   **Các tham số học được:** Một trọng số (hệ số hệ thống) cho mỗi chiều đặc trưng và một sai số chặn (bias).
*   **Tiêu chí tối ưu hóa:** Giảm thiểu sai số bình phương trung bình (MSE) giữa log-price dự đoán và thực tế.
*   **Giả định của mô hình:** Tính tuyến tính giữa các đặc trưng và biến mục tiêu log, không có đa cộng tuyến nghiêm trọng, và phần dư có phương sai không đổi (homoscedasticity).
*   **Điểm mạnh:** Huấn luyện cực nhanh, các hệ số trọng số có thể giải thích trực quan, là điểm tham chiếu (baseline) tốt.
*   **Điểm yếu:** Không thể nắm bắt các tác động phi tuyến của giá (ví dụ: diện tích tăng lên một ngưỡng nào đó thì giá trị biên sẽ giảm dần) hoặc các tương tác đặc trưng (như Quận × Pháp lý).

### 2. Bộ hồi quy vector hỗ trợ với nhân RBF (SVR RBF Kernel)
*   **Đầu vào nhận được:** Vector đặc trưng đã chuẩn hóa, biến mục tiêu log-price.
*   **Mối quan hệ cần học:** Một hàm phi tuyến cố gắng giữ hầu hết các điểm huấn luyện nằm trong một biên sai số $\epsilon$ quanh mặt phẳng dự đoán, sử dụng nhân RBF để chiếu dữ liệu vào không gian nhiều chiều hơn.
*   **Các tham số học được:** Các vector hỗ trợ (support vectors), trọng số của chúng, và tầm ảnh hưởng của nhân (`gamma`).
*   **Tiêu chí tối ưu hóa:** Tối thiểu hóa độ phức tạp của mô hình trong khi chấp nhận các sai số nhỏ nằm trong khoảng $\epsilon$ (hàm mất mát $\epsilon$-insensitive), được kiểm soát bởi các siêu tham số `C` và `gamma`.
*   **Giả định của mô hình:** Các điểm nằm gần nhau trong không gian đặc trưng biến đổi sẽ có mức giá tương đương.
*   **Điểm mạnh:** Nắm bắt tốt các mối quan hệ phi tuyến phức tạp mà không cần thiết kế đặc trưng thủ công; chống nhiễu tốt với các điểm ngoại lệ trung bình.
*   **Điểm yếu:** Chi phí huấn luyện rất lớn trên tập dữ liệu lớn (30.000 dòng); cần tinh chỉnh siêu tham số (`C`, `gamma`) kỹ lưỡng; khó giải thích.

### 3. Bộ hồi quy vector hỗ trợ với nhân tuyến tính (SVR Linear Kernel)
*   **Đầu vào nhận được:** Vector đặc trưng đã chuẩn hóa, biến mục tiêu log-price.
*   **Mối quan hệ cần học:** Một mặt phẳng giá tuyến tính giữ hầu hết các điểm dữ liệu trong biên $\epsilon$, tương tự như Hồi quy tuyến tính nhưng sử dụng hàm mất mát dựa trên biên thay vì bình phương sai số tối thiểu.
*   **Các tham số học được:** Vector trọng số và bias định hình siêu phẳng hồi quy tuyến tính.
*   **Tiêu chí tối ưu hóa:** Hàm mất mát $\epsilon$-insensitive bình phương phần dư, được điều tiết bởi siêu tham số phạt `C`.
*   **Giả định của mô hình:** Mặt phẳng giá xấp xỉ tuyến tính trong không gian đặc trưng đã mã hóa.
*   **Điểm mạnh:** Chống điểm ngoại lệ tốt hơn hồi quy tuyến tính thông thường; hội tụ nhanh hơn nhân RBF nhờ công thức giải thuật tối ưu khi mẫu lớn (`dual=False`).
*   **Điểm yếu:** Bị giới hạn bởi tính tuyến tính; không tự học được các tương tác phi tuyến giữa các biến.

### 4. Hồi quy K lân cận gần nhất (K-Nearest Neighbors Regressor)
*   **Đầu vào nhận được:** Vector đặc trưng đã chuẩn hóa (bắt buộc vì KNN tính toán khoảng cách hình học).
*   **Mối quan hệ cần học:** Không học một hàm toán học cụ thể nào; dự đoán chỉ đơn giản là trung bình cộng (có thể tính theo trọng số khoảng cách) của giá của `k` bất động sản giống nhất trong tập huấn luyện.
*   **Các tham số học được:** Không có tham số nào (KNN là thuật toán dạng lazy-learning, ghi nhớ toàn bộ tập dữ liệu).
*   **Tiêu chí tối ưu hóa:** Khoảng cách Euclidean trong không gian đặc trưng chuẩn hóa 367 chiều.
*   **Giả định của mô hình:** Các bất động sản có thuộc tính giống nhau thì giá sẽ gần nhau; mọi chiều đặc trưng đều đóng góp quan trọng vào khoảng cách.
*   **Điểm mạnh:** Đơn giản, nắm bắt tốt các cấu trúc phi tuyến cục bộ, không mất thời gian huấn luyện.
*   **Điểm yếu:** Dự đoán chậm khi tập dữ liệu lớn; hiệu năng giảm mạnh trong không gian thưa thớt nhiều chiều (367 chiều, chủ yếu là các cột one-hot nhị phân) — hiện tượng "lời nguyền đa chiều" (curse of dimensionality).

### 5. Hồi quy Rừng ngẫu nhiên (Random Forest Regressor)
*   **Đầu vào nhận được:** Vector đặc trưng mã hóa one-hot không nhất thiết phải chuẩn hóa (mô hình dạng cây không nhạy cảm với tỷ lệ đặc trưng).
*   **Mối quan hệ cần học:** Phân chia phi tuyến, phân cấp không gian đặc trưng bằng cách trung bình hóa kết quả của nhiều cây quyết định độc lập.
*   **Các tham số học được:** Một tập hợp nhiều cây quyết định, mỗi cây chứa các quy tắc phân nhánh dựa trên ngưỡng đặc trưng.
*   **Tiêu chí tối ưu hóa:** Giảm thiểu phương sai (sai số bình phương) tại mỗi điểm chia nhánh, tính trung bình trên các mẫu bootstrap.
*   **Giả định của mô hình:** Không có giả định phân phối nghiêm ngặt; giả định rằng việc gộp nhiều cây độc lập sẽ làm triệt tiêu phương sai.
*   **Điểm mạnh:** Tự động nắm bắt các mối quan hệ phi tuyến và tương tác đặc trưng phức tạp (như Diện tích × Quận huyện); hoạt động rất tốt với các cột one-hot thưa thớt.
*   **Điểm yếu:** Tiêu tốn nhiều bộ nhớ và tài nguyên tính toán khi lưu trữ rừng cây; kém trực quan hơn mô hình tuyến tính đơn giản.

### 6. Hồi quy XGBoost (XGBoost Regressor)
*   **Đầu vào nhận được:** Vector đặc trưng mã hóa one-hot không cần chuẩn hóa.
*   **Mối quan hệ cần học:** Một hàm phi tuyến phức tạp được xây dựng bằng cách cộng dồn tuần tự các cây quyết định nhỏ, mỗi cây sau sẽ sửa chữa sai số phần dư của cây trước đó.
*   **Các tham số học được:** Chuỗi các cây tăng cường với các điểm chia nhánh và trọng số lá tối ưu.
*   **Tiêu chí tối ưu hóa:** Tối thiểu hóa hàm mục tiêu có chứa thành phần phạt độ phức tạp (regularization) để tránh quá khớp (overfitting).
*   **Giả định của mô hình:** Sai số phần dư của các cây trước đó vẫn chứa các tín hiệu quy luật mà các cây sau có thể học và sửa chữa.
*   **Điểm mạnh:** Đạt độ chính xác hàng đầu đối với dữ liệu dạng bảng (tabular data); xử lý xuất sắc các biến vị trí phân loại cao (342 quận/huyện); cơ chế kiểm soát quá khớp tích hợp.
*   **Điểm yếu:** Nhiều siêu tham số cần tinh chỉnh (`learning_rate`, `max_depth`, `n_estimators`); có thể bị quá khớp nếu không điều chỉnh đúng; là mô hình dạng hộp đen khó giải thích nhất.

## 7. Thiết kế thực nghiệm

### Thực nghiệm 1: So sánh mô hình (Model Comparison)
*   **Câu hỏi:** Mô hình hồi quy nào trong số 6 mô hình dự đoán tốt nhất giá trị bất động sản dưới cùng một điều kiện dữ liệu?
*   **Thiết lập:** Cả 6 mô hình được huấn luyện trên cùng một tỷ lệ chia 80/20 (`random_state=42`), sử dụng chung ma trận đặc trưng đầu vào, với mục tiêu là `log1p(Price)`. Kết quả dự đoán được chuyển đổi ngược bằng `expm1` trước khi tính điểm trên tập kiểm thử độc lập.

### Thực nghiệm 2: Khảo sát siêu tham số (Hyperparameter Investigation)
*   **Câu hỏi:** Số lượng lân cận (`n_neighbors`) và cơ chế tính trọng số (`uniform` so với `distance`) ảnh hưởng thế nào đến độ chính xác hồi quy của KNN trên tập dữ liệu này?
*   **Thiết lập:** Thực hiện `GridSearchCV` (3-fold) trên tập huấn luyện đã chuẩn hóa với các giá trị `n_neighbors` ∈ {3, 5, 7, 9} và `weights` ∈ {uniform, distance}, tối ưu hóa theo âm sai số bình phương trung bình (-MSE) trong không gian log-price.
*   **Kết quả:**

| n_neighbors | weights | CV RMSE (không gian log) |
| --- | --- | --- |
| 3 | distance | 0.2807 |
| 3 | uniform | 0.2820 |
| 5 | distance | 0.2720 |
| 5 | uniform | 0.2751 |
| 7 | distance | 0.2688 |
| 7 | uniform | 0.2737 |
| **9** | **distance** | **0.2684 (Tốt nhất)** |
| 9 | uniform | 0.2751 |

Sai số giảm dần một cách nhất quán khi số lân cận `k` tăng từ 3 lên 9 (giúp giảm phương sai và tránh quá khớp với các lân cận bị nhiễu). Trọng số theo khoảng cách (`distance`) luôn cho kết quả tốt hơn trọng số đều (`uniform`) ở mọi mức `k`. Cấu hình tối ưu được tìm thấy là `n_neighbors=9, weights='distance'`.

### Thực nghiệm 3: Khảo sát biểu diễn đặc trưng (Feature Representation Investigation)
*   **Câu hỏi:** Việc mô hình hóa biến mục tiêu dưới dạng `log1p(Price)` thay vì giữ nguyên giá trị gốc `Price` có giúp cải thiện hiệu năng của Hồi quy tuyến tính hay không?
*   **Thiết lập:** Huấn luyện Hồi quy tuyến tính hai lần trên cùng một tập dữ liệu: một lần trực tiếp trên giá trị gốc `Price`, và một lần trên `log1p(Price)` rồi chuyển đổi ngược bằng `expm1`. Đánh giá cả hai trên cùng một tập kiểm thử.
*   **Kết quả:**

| Mục tiêu (Target) | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Giá trị gốc `Price` | **1.2747** | **1.6391** | **0.4490** |
| Biến đổi `log1p(Price)` | 1.3001 | 1.7478 | 0.3735 |

**Trái với kỳ vọng ban đầu**, việc mô hình hóa trực tiếp giá trị gốc mang lại kết quả MAE, RMSE và R² tốt hơn rõ rệt so với việc biến đổi log đối với mô hình Hồi quy tuyến tính. Nguyên nhân là vì hàm chuyển đổi ngược `expm1()` là một phép toán phi tuyến: việc giảm thiểu sai số bình phương trong không gian log không đồng nghĩa với việc giảm thiểu sai số bình phương trong không gian giá gốc ban đầu (hệ quả từ bất đẳng thức Jensen). Phép đổi ngược này đã khuếch đại các sai số nhỏ ở không gian log thành sai số lớn ở không gian tiền tệ gốc đối với mô hình tuyến tính đơn giản. (Các mô hình cây như Random Forest hay XGBoost ít bị ảnh hưởng bởi điều này do cấu trúc phân mảnh phi tuyến của chúng).

## 8. Kết quả
*Các kết quả dưới đây thu được bằng cách chạy file [Report_Notebook.ipynb](Report_Notebook.ipynb).*

| Mô hình | MAE (Tỷ VNĐ) | MSE | RMSE (Tỷ VNĐ) | R² Score | Độ chính xác MAPE |
| --- | --- | --- | --- | --- | --- |
| Baseline (Dự báo trung bình) | 1.8438 | 4.8760 | 2.2082 | -0.0000 | — |
| Hồi quy tuyến tính (Linear Reg) | 1.3001 | 3.0549 | 1.7478 | 0.3735 | 74.02% |
| SVR (Tuyến tính) | 1.3147 | 3.0400 | 1.7435 | 0.3765 | 73.94% |
| KNN (k=9, distance) | 1.2747 | 2.8730 | 1.6950 | 0.4108 | 74.37% |
| SVR (RBF) | 1.1195 | 2.3166 | 1.5221 | 0.5249 | 77.54% |
| Rừng ngẫu nhiên (Random Forest) | 1.0738 | 2.1475 | 1.4654 | 0.5596 | 77.96% |
| **XGBoost** | **1.0498** | **1.9727** | **1.4045** | **0.5954** | **78.91%** |

*(Tất cả giá trị sai số được quy về đơn vị tỷ VNĐ sau khi chuyển đổi ngược từ không gian log. Mọi mô hình đều vượt trội rõ rệt so với Baseline - mô hình cơ sở chỉ dự báo bằng giá trị trung bình với R² bằng 0).*

**Các độ đo phù hợp:** Hệ thống báo cáo 5 độ đo. **R²** thể hiện phần trăm phương sai của giá được mô hình giải thích, nhưng **MAE** và **RMSE** (đều ở đơn vị tỷ VNĐ) mang tính thực tiễn cao hơn cho người dùng cuối vì chúng biểu thị mức sai lệch trung bình bằng tiền tệ thực tế. **MSE** phản ánh trực tiếp mục tiêu tối ưu hóa của các thuật toán huấn luyện, trong khi **RMSE** (căn bậc hai của MSE) phạt nặng các sai số lớn — rất phù hợp cho dữ liệu bất động sản có nhiều căn nhà giá trị cực cao. **Độ chính xác MAPE** (`max(0, (1 - MAPE) * 100)`) cung cấp một tỷ lệ phần trăm trực quan giúp người dùng không chuyên dễ hiểu năng lực của mô hình.

## 9. So sánh mô hình
**XGBoost** đạt hiệu năng xuất sắc nhất trên mọi độ đo (R² = 0.5954, MAE = 1.05 tỷ VNĐ, độ chính xác MAPE = 78.91%) và được tự động chọn làm mô hình cuối cùng cho ứng dụng. **Random Forest** bám sát ở vị trí thứ hai (R² = 0.5596). Cả hai mô hình dạng cây này đều vượt trội hơn hẳn các mô hình phi cây, chứng minh rằng giá nhà phụ thuộc nhiều vào các tương tác đặc trưng phi tuyến tính phức tạp (ví dụ: diện tích lớn chỉ có giá trị cao khi nằm ở khu vực đắc địa và có pháp lý đầy đủ). **SVR (RBF)** là mô hình phi cây tốt nhất (R² = 0.5249), cho thấy nhân phi tuyến đã khôi phục được một phần các mối quan hệ phức tạp này. **KNN**, **SVR (Tuyến tính)** và **Hồi quy tuyến tính** có kết quả kém nhất, lý do bởi KNN bị ảnh hưởng bởi không gian thưa thớt 367 chiều, còn các mô hình tuyến tính hoàn toàn bỏ qua tương tác giữa các thuộc tính.

## 10. Phân tích cách biểu diễn dữ liệu
*   **Tại sao biểu diễn dạng vector đặc trưng lại phù hợp?** Mỗi tin đăng bất động sản có một tập hợp cố định các thuộc tính độc lập (diện tích, phòng ngủ, pháp lý, quận huyện), rất phù hợp để ánh xạ thành một vector đặc trưng có độ dài cố định — tiêu chuẩn cho dữ liệu bảng.
*   **Biểu diễn này bảo toàn thông tin nào?** Nó bảo toàn giá trị đo lường vật lý chính xác (diện tích, mặt tiền, ngõ vào) và danh tính cụ thể của các thuộc tính phân loại (hướng, pháp lý, nội thất, quận huyện).
*   **Biểu diễn này có thể làm mất thông tin nào?** Nó làm mất đi các chi tiết mô tả tự do trong địa chỉ cụ thể (chỉ giữ lại tên Quận/Huyện), thông tin trực quan (hình ảnh nhà đất), xu hướng biến động giá theo thời gian của thị trường, và tọa độ GPS chính xác.
*   **Bài toán này có thể biểu diễn dưới dạng ảnh không?** Có một phần — hình ảnh nhà đất có thể được đưa qua mạng CNN để đánh giá chất lượng xây dựng và nội thất, nhưng các thông tin cốt lõi như pháp lý hay diện tích vẫn cần dữ liệu dạng bảng.
*   **Có thể biểu diễn dạng chuỗi không?** Có, nếu một bất động sản hoặc một khu vực có nhiều giao dịch được ghi nhận theo thời gian, ta có thể dùng mô hình chuỗi thời gian (như LSTM) để dự đoán xu hướng tăng giá.
*   **Có thể biểu diễn dạng đồ thị (Graph) không?** Rất phù hợp — các bất động sản có thể coi là các nút (nodes) kết nối với nhau dựa trên khoảng cách địa lý hoặc khu vực lân cận, cho phép Mạng nơ-ron đồ thị (GNN) lan truyền tín hiệu giá từ các căn nhà tương đồng ở gần đó.
*   **Có thể biểu diễn bằng Embedding học được không?** Có — thay vì mã hóa one-hot 342 quận huyện tạo ra ma trận rất thưa, việc học một không gian Vector nhúng (Embedding) cho các quận huyện hoặc tọa độ GPS sẽ giảm đáng kể số chiều và tăng lượng thông tin vị trí.
*   **Điều gì sẽ thay đổi nếu cách biểu diễn thay đổi?** Việc sử dụng Embedding hoặc Đồ thị sẽ giúp giảm số chiều đầu vào (tránh lời nguyền đa chiều đối với KNN), cải thiện các mô hình dựa trên khoảng cách/biên, nhưng sẽ yêu cầu hạ tầng dữ liệu địa lý phức tạp hơn và mất đi tính giải thích trực quan đơn giản.

## 11. Ứng dụng thông minh
Mô hình `xgboost.pkl` (được tự động chọn nhờ đạt R² cao nhất) đã được tích hợp vào ứng dụng Full-stack (Backend FastAPI + Frontend React/Capacitor). Người dùng có thể nhập các thông tin chi tiết của căn nhà trên giao diện Web hoặc Mobile để nhận ngay giá dự báo từ API backend.

## 12. Hạn chế
*   **Dữ liệu khuyết quá nhiều:** Hướng ban công (khuyết 82.6%), Hướng nhà (khuyết 70.3%), Nội thất (khuyết 46.7%), Ngõ vào (khuyết 44.0%), Mặt tiền (khuyết 38.3%). Việc điền khuyết bằng trung vị hoặc giá trị `'Unknown'` chỉ là giải pháp tình thế và có thể làm giảm độ tin cậy của dự đoán.
*   **Mã hóa One-Hot quận huyện số chiều lớn:** Tạo ra 341 cột nhị phân thưa thớt, gây bất lợi cho KNN và làm tăng nguy cơ quá khớp cho mô hình đối với các quận huyện có quá ít mẫu.
*   **Lệch phép chuyển đổi ngược Log:** Phép đổi ngược `expm1` có thể vô tình làm tăng sai số tiền tệ thực tế cho các mô hình tuyến tính đơn giản.
*   **Thiếu tọa độ GPS:** Chỉ phân loại theo Quận/Huyện khiến mô hình không thể đánh giá các yếu tố vị trí chi tiết như gần trường học, bệnh viện, hay mặt đường lớn.
*   **Giới hạn khoảng giá:** Dữ liệu huấn luyện chủ yếu nằm trong khoảng 1.0 đến 11.5 tỷ VNĐ; mô hình sẽ dự đoán không chính xác cho các bất động sản siêu sang hoặc siêu rẻ nằm ngoài khoảng này.

## 13. Suy ngẫm
*   **Hệ thống nhận thông tin gì?** Các thông số vật lý, bố cục phòng, hướng, pháp lý, nội thất và văn bản địa chỉ thô của căn nhà.
*   **Biểu diễn nội bộ ra sao?** Một vector 367 chiều chuẩn hóa và biến mục tiêu được huấn luyện trong không gian log.
*   **Mô hình học được gì từ các ví dụ?** Học được các quy luật định giá phi tuyến tính (ví dụ: diện tích lớn ở trung tâm và có sổ đỏ thì giá trị sẽ tăng vượt bậc).
*   **Tại sao hệ thống xử lý được đầu vào chưa từng thấy?** Vì nó đã khái quát hóa được các quy luật chung từ 30.000 mẫu huấn luyện thay vì học vẹt, thể hiện qua việc mô hình đạt điểm tốt trên tập kiểm thử độc lập 20%.
*   **Thành phần nào được gọi là "thông minh"?** Quá trình huấn luyện, nơi mô hình tự động điều chỉnh hàng ngàn trọng số và quy tắc phân nhánh để khớp tối đa với dữ liệu thực tế mà không cần lập trình các quy tắc định giá thủ công.
*   **Hạn chế nào cản trước hệ thống thông minh hơn?** Hệ thống không thể xem ảnh nhà (thiếu thị giác máy), không có tư duy bản đồ địa lý thực sự, không biết xu hướng thị trường tại thời điểm thực tế, và không thể tự giải thích logic định giá bằng ngôn ngữ tự nhiên.

## 14. Kết luận
Chúng tôi đã xây dựng thành công một hệ thống thông minh dự đoán giá nhà tại Việt Nam. Trải qua 6 mô hình ML truyền thống, XGBoost đạt hiệu năng cao nhất (R² = 0.5954, MAE ≈ 1.05 tỷ VNĐ). Các thực nghiệm kiểm soát chỉ ra rằng KNN đạt kết quả tốt nhất ở `k=9` với trọng số khoảng cách, và việc biến đổi log đối với biến mục tiêu cần được lựa chọn cẩn thận để tránh sai số chuyển đổi ngược trên các mô hình tuyến tính. Ứng dụng đã được triển khai hoàn chỉnh giúp định giá nhanh chóng, song hiệu năng trong tương lai cần được cải thiện bằng cách bổ sung tọa độ GPS, ảnh chụp bất động sản và các vector nhúng vị trí chuyên sâu.

---

# PHẦN 2: GIẢI THÍCH CHI TIẾT CÁC NOTEBOOK TỪNG CELL MỘT

## MỤC A: Notebook Huấn Luyện Dữ Liệu (`Training_1.ipynb`)

### [Cell 0 (Markdown)]
*   **Nội dung:** Tiêu đề của notebook và giới thiệu 6 mô hình hồi quy sẽ được đánh giá, bao gồm các thước đo đánh giá hiệu năng chính: MAE, RMSE, R-squared ($R^2$). Hứa hẹn sẽ vẽ biểu đồ so sánh Thực tế vs Dự đoán (Actual vs Predicted Density) và biểu đồ phân phối sai số phần dư (Residual Distribution), cuối cùng xếp hạng mô hình.
*   **Ý nghĩa:** Cung cấp mục tiêu và cấu trúc tổng quát của toàn bộ quy trình huấn luyện trong notebook.

### [Cell 1 (Code)]
*   **Nội dung:** Khai báo thư viện, tải tập dữ liệu, xử lý điền khuyết (imputation), trích xuất quận huyện, mã hóa đặc trưng, phân tách tập dữ liệu và chuẩn hóa đặc trưng.
*   **Giải thích mã nguồn & thông số:**
    *   **Thư viện:** `pandas` để xử lý bảng; `numpy` cho các phép toán số học; `matplotlib` & `seaborn` để vẽ đồ thị; `joblib` để lưu trữ mô hình pkl; `sklearn` cho các công cụ ML và đánh giá; `xgboost` cho mô hình cây quyết định tăng cường.
    *   **Điền khuyết (Imputation):**
        *   Các giá trị số khuyết (`numerical_cols`) được điền bằng **Trung vị (Median)** của cột đó (`df[col].fillna(df[col].median())`). Trung vị được chọn thay vì trung bình vì nó ít bị ảnh hưởng bởi các giá trị ngoại lệ (outliers) cực đoan trong dữ liệu bất động sản.
        *   Các cột phân loại khuyết (`categorical_cols`) được điền bằng chuỗi đại diện `'Unknown'` để giữ nguyên thông tin rằng dữ liệu này bị khuyết chứ không tự ý gán bừa.
    *   **Trích xuất Quận/Huyện (`extract_district`):**
        Do địa chỉ Việt Nam thường viết theo thứ tự tăng dần từ số nhà, đường, phường, quận/huyện, tỉnh/thành phố và phân tách bởi dấu phẩy, hàm này phân tách chuỗi địa chỉ theo dấu phẩy và lấy phần tử thứ 2 từ dưới lên (`parts[-2]`) để lấy đúng tên Quận/Huyện (ví dụ: "Quận Ba Đình").
    *   **Mã hóa One-Hot (`pd.get_dummies`):**
        Chuyển các biến danh mục thành các cột nhị phân (0 hoặc 1). Tham số `drop_first=True` giúp tránh bẫy đa cộng tuyến (Dummy Variable Trap) bằng cách loại bỏ cột phân loại đầu tiên, vì trạng thái của nó có thể được suy ra trực tiếp từ các cột còn lại.
    *   **Biến đổi Log mục tiêu (`y_log = np.log1p(df['Price'])`):**
        Sử dụng phép biến đổi $\log(1 + x)$ trên biến giá nhà. Việc này cực kỳ quan trọng vì giá bất động sản có phân phối lệch phải rất nặng (nhiều căn nhà giá trị rất lớn tạo đuôi dài trên biểu đồ). Phép biến đổi log giúp thu nhỏ khoảng cách giữa các giá trị lớn, đưa phân phối về dạng chuẩn (Gaussian) hơn, giúp các thuật toán học dễ dàng hơn.
    *   **Chuẩn hóa (`StandardScaler`):**
        Chuẩn hóa các biến số về dạng có trung bình bằng 0 và độ lệch chuẩn bằng 1. Đây là bước bắt buộc đối với các thuật toán đo khoảng cách (KNN) hoặc tối ưu biên (SVR), đảm bảo các đặc trưng có đơn vị lớn (ví dụ diện tích 100m²) không lấn át các đặc trưng có đơn vị nhỏ (như số phòng ngủ 3 phòng).

### [Cell 2 (Markdown) & Cell 3 (Code)] - Hồi quy tuyến tính (Linear Regression)
*   **Nội dung:** Thiết lập GridSearch tối ưu hóa tham số cho mô hình Hồi quy tuyến tính, dự đoán trên tập kiểm thử, đổi ngược giá trị log về giá gốc, tính toán các chỉ số và vẽ biểu đồ.
*   **Thông số siêu tham số:**
    *   `'fit_intercept': [True, False]`: Xác định xem mô hình có tính toán sai số chặn (hệ số tự do $w_0$) hay ép đường thẳng phải đi qua gốc tọa độ. Kết quả GridSearch sẽ chọn tùy chọn tốt nhất dựa trên điểm đánh giá chéo (Cross Validation).
*   **Ý nghĩa các biểu đồ:**
    *   **Biểu đồ bên trái (Hexbin Density Plot):** Trục X biểu diễn Giá thực tế, Trục Y biểu diễn Giá dự báo. Đường chéo nét đứt màu đỏ biểu thị dự báo hoàn hảo (Thực tế = Dự báo). Do tập test có tới 6.000 điểm, nếu dùng biểu đồ điểm (scatter) thông thường các điểm dữ liệu sẽ chồng lên nhau thành một khối đen đặc (overplotting), không thể nhận diện được mật độ. Đồ thị Hexbin chia mặt phẳng thành các ô lục giác nhỏ và tô màu dựa trên số lượng điểm rơi vào ô đó. Màu xanh càng đậm biểu thị mật độ tập trung điểm càng cao. Chúng ta muốn thấy vùng màu xanh đậm bám sát đường đỏ nét đứt.
    *   **Biểu đồ bên phải (Residual Distribution Histogram + KDE):** Trục X biểu thị Sai số phần dư (Giá thực tế - Giá dự đoán), Trục Y biểu thị tần suất xuất hiện. Đường nét đứt màu đỏ là vạch số 0 (sai số bằng 0). Đồ thị này hiển thị phân phối của các sai số. Một mô hình tốt sẽ có phân phối sai số đối xứng qua vạch số 0, có đỉnh rất cao và hẹp tại 0 (phần lớn sai số nhỏ) và hai bên đuôi ngắn (rất ít sai lệch lớn). Đuôi kéo dài về bên phải biểu thị mô hình đang dự báo thấp hơn nhiều so với giá trị thực tế của một số căn nhà đắt tiền.

### [Cell 4 (Markdown) & Cell 5 (Code)] - SVR với nhân RBF (SVR RBF Kernel)
*   **Nội dung:** Tinh chỉnh và huấn luyện mô hình Vector hỗ trợ sử dụng nhân hàm cơ sở xuyên tâm (RBF).
*   **Thông số siêu tham số & Tối ưu hóa:**
    *   `'C': [0.1, 1, 10]`: Tham số phạt (regularization). Giá trị `C` càng cao thì mô hình càng cố gắng khớp chính xác toàn bộ điểm huấn luyện (dễ quá khớp), giá trị `C` thấp sẽ chấp nhận sai số lớn hơn để đổi lấy mặt phẳng dự đoán trơn tru hơn (dễ dưới khớp).
    *   `'gamma': ['scale', 'auto', 0.1, 1]`: Định hình bán kính ảnh hưởng của các vector hỗ trợ đơn lẻ. `gamma` cao nghĩa là tầm ảnh hưởng ngắn (mặt phẳng dự đoán uốn lượn phức tạp hơn), `gamma` thấp nghĩa là tầm ảnh hưởng rộng (mặt phẳng dự đoán phẳng hơn).
    *   **Tối ưu hóa tốc độ huấn luyện (Fast Tuning):** Nhân RBF của SVR tính toán ma trận khoảng cách giữa mọi cặp điểm dữ liệu, độ phức tạp tính toán là $O(n^3)$ với số mẫu $n$. Chạy GridSearch trên toàn bộ 24.000 dòng huấn luyện sẽ mất nhiều giờ. Để giải quyết, code trích xuất **20% dữ liệu huấn luyện ngẫu nhiên** (`train_size=0.20`) để chạy GridSearch tìm ra bộ tham số tốt nhất trước, sau đó lấy bộ tham số tối ưu đó để khớp (fit) duy nhất một lần trên **100% dữ liệu huấn luyện**. Đây là kỹ thuật thông minh giúp cân bằng giữa độ chính xác và thời gian xử lý.
*   **Đồ thị Hexbin & Residual:** Cho thấy vùng màu đậm tập trung sát đường đỏ hơn so với Hồi quy tuyến tính, sai số phân phối hẹp hơn quanh vạch số 0, chứng tỏ mô hình học được các quy luật phi tuyến tốt hơn.

### [Cell 6 (Markdown) & Cell 7 (Code)] - SVR với nhân Tuyến tính (SVR Linear Kernel)
*   **Nội dung:** Huấn luyện mô hình Vector hỗ trợ nhân tuyến tính tối ưu hóa tốc độ.
*   **Thông số siêu tham số & Tối ưu hóa:**
    *   Sử dụng lớp `LinearSVR` thay thế cho lớp `SVR(kernel='linear')`. Lý do: `SVR` sử dụng giải thuật LibSVM có độ phức tạp cao, trong khi `LinearSVR` sử dụng giải thuật LibLinear được tối ưu hóa chuyên biệt cho nhân tuyến tính.
    *   `dual=False`: Giải bài toán tối ưu primal thay vì dual. Theo tài liệu của Scikit-Learn, khi số lượng mẫu dữ liệu ($n \approx 30.000$) lớn hơn nhiều so với số lượng đặc trưng ($p \approx 367$), việc đặt `dual=False` giúp thuật toán hội tụ nhanh hơn hàng chục lần.
    *   `loss='squared_epsilon_insensitive'`: Sử dụng hàm mất mát bình phương sai số vượt quá ngưỡng $\epsilon$, giúp mô hình mượt mà hơn và hội tụ nhanh hơn.
    *   `'epsilon': [0.0, 0.1, 0.2]`: Độ rộng của biên không bị phạt sai số. Bất kỳ điểm nào nằm trong biên $\epsilon$ quanh đường dự báo đều được coi là có sai số bằng 0.

### [Cell 8 (Markdown) & Cell 9 (Code)] - K lân cận gần nhất (KNN Regressor)
*   **Nội dung:** Tinh chỉnh số lượng lân cận và trọng số cho mô hình KNN.
*   **Thông số siêu tham số:**
    *   `'n_neighbors': [3, 5, 7, 9]`: Số lượng điểm tương đồng gần nhất được dùng để trung bình hóa giá trị dự báo.
    *   `'weights': ['uniform', 'distance']`:
        *   `'uniform'`: Coi mọi lân cận có vai trò ngang nhau khi tính trung bình giá trị.
        *   `'distance'`: Tính trọng số tỉ lệ nghịch với khoảng cách (điểm nào càng gần căn nhà cần định giá thì giá của nó càng đóng góp nhiều vào kết quả dự đoán).

### [Cell 10 (Markdown) & Cell 11 (Code)] - Rừng ngẫu nhiên (Random Forest)
*   **Nội dung:** Huấn luyện mô hình ensemble đóng gói (bagging) gồm nhiều cây quyết định độc lập.
*   **Thông số siêu tham số:**
    *   `'n_estimators': [50, 100, 200]`: Số lượng cây quyết định trong rừng. Càng nhiều cây, kết quả dự báo càng ổn định và giảm phương sai nhưng huấn luyện càng lâu.
    *   `'max_depth': [None, 10, 20]`: Độ sâu tối đa của mỗi cây. `None` nghĩa là cây sẽ phân nhánh cho đến khi các lá đều thuần khiết, `10` hoặc `20` giới hạn độ sâu để ngăn chặn quá khớp (overfitting).
    *   `'min_samples_split': [2, 5]`: Số lượng mẫu tối thiểu cần thiết trong một nút để tiếp tục phân nhánh. Giá trị cao hơn ngăn mô hình tạo các nhánh quá chi tiết.

### [Cell 12 (Markdown) & Cell 13 (Code)] - XGBoost Regressor
*   **Nội dung:** Huấn luyện mô hình ensemble tăng cường độ dốc (gradient boosting) liên kết tuần tự.
*   **Thông số siêu tham số:**
    *   `'n_estimators': [100, 200]`: Số lượng cây quyết định được xây dựng nối tiếp nhau.
    *   `'learning_rate': [0.05, 0.1, 0.2]`: Tỷ lệ học (shrinkage factor), giới hạn mức độ đóng góp của mỗi cây mới vào tổng thể mô hình. Tỷ lệ học nhỏ kết hợp với nhiều cây thường giúp mô hình học mịn hơn và tránh quá khớp.
    *   `'max_depth': [3, 5, 7]`: Độ sâu tối đa của các cây yếu. Do XGBoost xây dựng các cây sửa sai nối tiếp, các cây này thường nông hơn (độ sâu 3 đến 7) để hoạt động như các bộ học yếu (weak learners).

### [Cell 14 (Markdown) & Cell 15 (Code)] - Bảng xếp hạng mô hình cuối cùng (Final Model Ranking)
*   **Nội dung:** Tổng hợp các số liệu đánh giá của cả 6 mô hình, in ra bảng định dạng đẹp mắt và vẽ biểu đồ cột xếp hạng theo điểm R-squared ($R^2$).
*   **Đồ thị cột xếp hạng (Bar Chart):** 
    Trục Y hiển thị tên mô hình, trục X hiển thị điểm $R^2$ tương ứng. Biểu đồ sử dụng dải màu `viridis` (từ tím đến vàng sáng) biểu thị mức độ hiệu năng tăng dần. Trên mỗi đầu cột, code hiển thị nhãn chi tiết chứa điểm $R^2$ chính xác và phần trăm độ chính xác MAPE tương ứng (ví dụ: `0.5954 | Acc: 78.91%`). Đồ thị này giúp người thiết kế hệ thống thấy ngay XGBoost là mô hình vượt trội nhất, theo sau là Random Forest, còn Hồi quy tuyến tính xếp ở cuối bảng.

---

## MỤC B: Notebook Trực Quan Hóa Dữ Liệu (`Visualization_2.ipynb`)

### [Cell 0 (Markdown)]
*   **Nội dung:** Giới thiệu tổng quan về các đặc trưng có trong tập dữ liệu (Numerical & Categorical).
*   **Ý nghĩa:** Xác lập ngữ cảnh nghiên cứu phân tích dữ liệu khám phá (EDA).

### [Cell 1 (Code)]
*   **Nội dung:** Nhập thư viện đồ họa (`matplotlib`, `seaborn`), tải tập tin dữ liệu csv và hiển thị 5 dòng đầu tiên (`df.head()`).
*   **Ý nghĩa:** Xác nhận dữ liệu được nạp chính xác và hiển thị trực quan cấu trúc ban đầu của bảng dữ liệu.

### [Cell 2 (Markdown) & Cell 3 (Code)] - Phân phối biến mục tiêu (Distribution of Price)
*   **Nội dung:** Vẽ biểu đồ phân phối (Histogram + KDE) và biểu đồ hộp (Boxplot) của biến mục tiêu `Price` (Giá nhà).
*   **Đồ thị thể hiện gì:**
    *   **Histogram + KDE (Teal):** Trục X hiển thị Giá nhà (tỷ VNĐ), Trục Y hiển thị tần suất xuất hiện. Đường cong mịn màu xanh là ước lượng mật độ nhân (KDE). Đồ thị này cho thấy một phân phối lệch phải cực kỳ nặng (Right-skewed distribution). Đại đa số các căn nhà tập trung trong khoảng từ 1.0 đến 6.0 tỷ VNĐ (vùng đỉnh nhô cao), và có một chiếc đuôi rất dài kéo về phía bên phải biểu thị các căn biệt thự/nhà mặt phố siêu sang có giá lên đến 10 tỷ - 11.5 tỷ VNĐ.
    *   **Boxplot (Biểu đồ hộp):** Hiển thị các giá trị tứ phân vị của giá nhà: Hộp biểu thị khoảng liên tứ phân vị (IQR) từ phân vị thứ 25 ($Q_1$) đến phân vị thứ 75 ($Q_3$). Vạch đứng bên trong hộp là Trung vị ($Q_2$). Hai râu (whiskers) kéo dài ra hai bên đại diện cho khoảng giá trị không chứa điểm ngoại lệ. Các điểm chấm tròn nằm ngoài râu bên phải chính là các **điểm ngoại lệ (outliers)** — các bất động sản giá trị cao vượt trội so với mặt bằng chung.
*   **Ý nghĩa thực tế:** Thị trường có một mặt bằng giá phổ thông (xung quanh trung vị ~4 tỷ VNĐ), nhưng phân phối bị kéo dãn bởi các căn nhà đắt đỏ. Đây là lý do chính buộc chúng ta phải sử dụng phép biến đổi log đối với biến giá trước khi đưa vào huấn luyện mô hình.

### [Cell 4 (Markdown) & Cell 5 (Code)] - Phân tích đặc trưng số (Numerical Features Analysis)
*   **Nội dung:** Vẽ biểu đồ phân tán (Scatter plot) kết hợp đường xu hướng hồi quy (regplot) giữa Diện tích (`Area`), Mặt tiền (`Frontage`) đối với Giá nhà (`Price`).
*   **Đồ thị thể hiện gì:**
    *   **Trục X:** Diện tích (m²) hoặc Mặt tiền (m), **Trục Y:** Giá (tỷ VNĐ).
    *   Các chấm tròn biểu thị các căn nhà riêng lẻ. Đường thẳng màu đỏ là đường hồi quy tuyến tính xu hướng (Trendline).
    *   Đồ thị cho thấy mối quan hệ đồng biến (tương quan thuận): Khi diện tích lớn hơn hoặc mặt tiền rộng hơn thì giá nhà có xu hướng tăng lên (đường màu đỏ dốc lên trên). Tuy nhiên, độ phân tán của các điểm khá lớn, chứng tỏ diện tích/mặt tiền không phải là yếu tố duy nhất quyết định giá, mà còn phụ thuộc vào các tương tác khác (như vị trí quận huyện).
*   **Ý nghĩa thực tế:** Xác nhận hai biến số vật lý này là những yếu tố dự báo quan trọng và có tác động thuận chiều trực tiếp đến giá nhà.

### [Cell 6 (Markdown) & Cell 7 (Code)] - Phân tích đặc trưng phân loại (Categorical Features Analysis)
*   **Nội dung:** Vẽ các biểu đồ hộp (Boxplot) thể hiện mức giá nhà theo từng nhóm phân loại của Tình trạng nội thất (`Furniture state`) và Tình trạng pháp lý (`Legal status`).
*   **Đồ thị thể hiện gì:**
    *   **Biểu đồ trên (Furniture state vs Price):** Trục X gồm các nhóm nội thất (Full, Basic, None, Unknown), trục Y là Giá nhà. Biểu đồ hộp của nhóm `Full` (Đầy đủ nội thất) có hộp và đường trung vị nằm cao hơn rõ rệt so với nhóm `None` (Không nội thất) hoặc `Basic` (Nội thất cơ bản), cho thấy nhà có đầy đủ nội thất có giá thị trường trung bình cao hơn.
    *   **Biểu đồ dưới (Legal status vs Price):** Trục X gồm các nhóm pháp lý (Have certificate - Đã có sổ, Waiting... - Đang chờ sổ, Sale contract - HĐMB, Unknown). Nhóm có sổ đỏ/sổ hồng (`Have certificate`) chiếm đa số và có mức giá phân bổ cao, ổn định hơn. Các giao dịch dạng Hợp đồng mua bán (`Sale contract`) hoặc đang chờ sổ thường có biên dao động giá rộng và giá trị trung bình thấp hơn do rủi ro pháp lý cao hơn.
*   **Ý nghĩa thực tế:** Xác nhận tình trạng pháp lý rõ ràng và trang bị nội thất đầy đủ là những nhân tố tạo ra giá trị thặng dư lớn cho bất động sản.

### [Cell 8 (Markdown) & Cell 9 (Code)] - Cấu trúc căn nhà (Discrete Property Layout)
*   **Nội dung:** Vẽ các biểu đồ cột tần suất (Countplot) cho các biến rời rạc: Số phòng ngủ (`Bedrooms`), Số phòng tắm (`Bathrooms`) và Số tầng (`Floors`).
*   **Đồ thị thể hiện gì:**
    *   Trục X là số lượng phòng/tầng, Trục Y là số lượng tin đăng tương ứng trên thị trường.
    *   **Bedrooms:** Phổ biến nhất là nhà có 3 hoặc 4 phòng ngủ.
    *   **Bathrooms:** Phổ biến nhất là nhà có 2 hoặc 3 phòng tắm.
    *   **Floors:** Phổ biến nhất là nhà xây từ 3 đến 5 tầng.
*   **Ý nghĩa thực tế:** Vẽ nên chân dung cấu trúc tiêu chuẩn của một căn nhà phổ thông đang được giao dịch tại Việt Nam (nhà ống 3-5 tầng, 3-4 phòng ngủ, 2-3 vệ sinh).

### [Cell 10 (Markdown) & Cell 11 (Code)] - Ma trận hệ số tương ứng (Correlation Matrix Heatmap)
*   **Nội dung:** Vẽ bản đồ nhiệt (Heatmap) thể hiện hệ số tương quan Pearson giữa tất cả các biến số.
*   **Đồ thị thể hiện gì:**
    *   Các ô chứa giá trị hệ số tương quan nằm trong khoảng từ -1 đến 1. Màu đỏ đậm biểu thị tương quan thuận mạnh mẽ (+1), màu xanh đậm biểu thị tương quan nghịch mạnh (-1), màu nhạt biểu thị ít hoặc không tương quan (0).
    *   **Hệ số giữa các biến với Price:**
        *   `Area` (Diện tích) tương quan mạnh nhất với `Price` (~0.55).
        *   `Bathrooms` (Số vệ sinh) và `Bedrooms` (Số phòng ngủ) cũng tương quan khá cao với giá (~0.45).
        *   `Access Road` (Độ rộng ngõ) có tương quan thuận nhẹ với giá.
    *   **Tương quan đa chiều (Multicollinearity):**
        *   Có sự tương quan khá cao giữa `Bedrooms` và `Bathrooms` (~0.70). Điều này hợp lý vì nhà nhiều phòng ngủ thường được xây nhiều phòng vệ sinh đi kèm. Tuy nhiên hệ số này chưa vượt quá ngưỡng nguy hiểm (>0.80) gây ra đa cộng tuyến nghiêm trọng đối với mô hình hồi quy tuyến tính.
*   **Ý nghĩa thực tế:** Định lượng chính xác tầm quan trọng của từng biến số vật lý và kiểm soát hiện tượng đa cộng tuyến giữa các biến cấu trúc phòng.

### [Cell 12 (Markdown) & Cell 13 (Code)] - Phân tích hướng nhà và ban công (House & Balcony Direction)
*   **Nội dung:** Sắp xếp hướng la bàn theo thứ tự chuẩn hóa và vẽ song song:
    *   *Phần A:* Tần suất xuất hiện (Countplot) của Hướng nhà & Hướng ban công.
    *   *Phần B:* Phân phối giá nhà (Boxplot) theo từng hướng tương ứng.
*   **Đồ thị thể hiện gì:**
    *   **Phần A (Supply):** Các hướng chính như Nam, Đông - Nam và Bắc có tần suất tin đăng lớn nhất. Đặc biệt hướng Đông - Nam rất phổ biến do đặc thù khí hậu Việt Nam (đón gió mát mùa hè, tránh gió mùa đông bắc).
    *   **Phần B (Market Value):** Các biểu đồ hộp phân phối giá nhà theo các hướng có chiều cao hộp và trung vị khá tương đồng nhau. Điều này cho thấy hướng nhà/ban công tuy ảnh hưởng đến quyết định mua nhưng **không tạo ra sự chênh lệch lớn về giá trị bất động sản** trên diện rộng.
*   **Ý nghĩa thực tế:** Hướng nhà/ban công là yếu tố ảnh hưởng đến thanh khoản (độ dễ bán) nhiều hơn là quyết định trực tiếp đến giá bán.

### [Cell 14 (Markdown) , Cell 15 (Code) & Cell 16 (Code)] - Phân tích Vị trí Địa lý theo Quận/Huyện (District Analysis)
*   **Nội dung:** Trích xuất Quận/Huyện từ địa chỉ thô, lọc ra top 15 quận/huyện có nhiều tin đăng nhất và vẽ song song:
    *   *Plot A:* Số lượng bất động sản rao bán ở từng quận/huyện.
    *   *Plot B:* Phân phối giá nhà ở từng quận/huyện tương ứng.
*   **Đồ thị thể hiện gì:**
    *   **Plot A (Countplot nằm ngang - Viridis):** Cho thấy khu vực nào có nguồn cung bất động sản dồi dào nhất (các quận trung tâm và quận đang phát triển nhanh như Hà Đông, Nam Từ Liêm, Hoàng Mai, Cầu Giấy...).
    *   **Plot B (Boxplot nằm ngang - Set2):** Cho thấy sự phân hóa giá trị rõ rệt theo khu vực địa lý. Ví dụ: Các quận trung tâm lâu đời (như Quận Hoàn Kiếm, Hai Bà Trưng, Ba Đình, Cầu Giấy) có dải giá nằm dịch hẳn về phía bên phải và trung vị rất cao, thể hiện mức giá đắt đỏ. Trong khi các quận/huyện vùng ven (như Hà Đông, Long Biên, Thanh Trì) có mức giá trung bình thấp hơn nhiều.
*   **Ý nghĩa thực tế:** Khẳng định giá trị cốt lõi của bất động sản: **"Vị trí, vị trí và vị trí"**. Quận/Huyện là biến phân loại có sức ảnh hưởng lớn nhất đến giá trị cuối cùng của căn nhà.

### [Cell 17 (Markdown) & Cell 18 (Markdown)] - Giới thiệu Phân tích Nâng cao
*   **Nội dung:** Giới thiệu các phần đồ thị nâng cao tiếp theo.

### [Cell 19 (Code)] - Biến đổi logarit biến mục tiêu (Raw Price vs. Log Price)
*   **Nội dung:** Vẽ biểu đồ so sánh phân phối giá gốc (Raw Price) và giá sau khi biến đổi log (Log-transformed Price).
*   **Đồ thị thể hiện gì:**
    *   **Biểu đồ trái (Raw Price - Teal):** Phân phối lệch phải cực kỳ nặng. Cột tần suất dồn cục ở bên trái và kéo dài một vệt dài thưa thớt về bên phải. Điều này vi phạm giả định phân phối chuẩn của các sai số trong mô hình tuyến tính.
    *   **Biểu đồ phải (Log Price - Darkblue):** Sau khi biến đổi $\log(1 + x)$, phân phối đã co lại và có hình dạng chuông đối xứng rất đẹp mắt (Gaussian-like distribution).
*   **Ý nghĩa thực tế:** Chứng minh bằng hình ảnh sự cần thiết của việc biến đổi logarit biến mục tiêu trước khi huấn luyện mô hình để cải thiện hiệu năng học tập của thuật toán.

### [Cell 20 (Markdown) & Cell 21 (Code)] - Phân tích mật độ dữ liệu khuyết (Missing Data Percentage)
*   **Nội dung:** Tính toán tỷ lệ phần trăm ô trống (NaN) của từng đặc trưng và vẽ biểu đồ cột nằm ngang từ cao đến thấp.
*   **Đồ thị thể hiện gì:**
    *   Trục Y là các đặc trưng bị khuyết dữ liệu, Trục X là tỷ lệ phần trăm khuyết.
    *   Hiển thị rõ rệt: Hướng ban công (`Balcony direction`) khuyết nhiều nhất (82.60%), theo sau là Hướng nhà (`House direction` - 70.29%), Nội thất (`Furniture state` - 46.73%), Chiều rộng ngõ (`Access Road` - 44.00%), và Mặt tiền (`Frontage` - 38.33%).
*   **Ý nghĩa thực tế:** Cảnh báo người xây dựng mô hình về chất lượng dữ liệu thô từ internet. Tỷ lệ khuyết cao ở các biến hướng và nội thất cho thấy người đăng tin thường bỏ qua các thông tin này, đòi hỏi hệ thống phải có phương án điền khuyết hợp lý.

### [Cell 22 (Markdown) & Cell 23 (Code)] - Tương quan đa biến nâng cao (Layout Specs vs Price)
*   **Nội dung:** Vẽ các biểu đồ hộp giới hạn số phòng ngủ, phòng tắm tối đa ở mức 6 và vẽ biểu đồ phân tán kết hợp đường xu hướng cho Chiều rộng ngõ (`Access Road`) đối với Giá nhà.
*   **Đồ thị thể hiện gì:**
    *   **Price by Bedrooms & Bathrooms (Capped at 6+):** Cho thấy xu hướng giá nhà tăng dần một cách rõ rệt khi số phòng ngủ tăng từ 1 lên 5 phòng, và số phòng tắm tăng từ 1 lên 5 phòng. Tuy nhiên ở mức 6 phòng trở lên, giá nhà có xu hướng đi ngang hoặc giảm nhẹ (biểu thị sự bão hòa của cấu trúc nhà ống hoặc phân khúc nhà cho thuê chung cư mini có giá trị giao dịch trung bình khác biệt).
    *   **Price vs Access Road Width (Orange scatter + Red trendline):** Đường xu hướng màu đỏ dốc lên rõ ràng, chứng tỏ ngõ càng rộng (xe ba gác, xe ô tô tránh nhau được) thì giá nhà càng cao.
*   **Ý nghĩa thực tế:** Xác định các điểm bão hòa của số lượng phòng trong nhà ống và định lượng giá trị tăng thêm của bất động sản khi có ngõ vào rộng rãi.

### [Cell 24 (Markdown) & Cell 25 (Code)] - Đồ thị ma trận điểm phân tán (Pairplot)
*   **Nội dung:** Trích xuất mẫu ngẫu nhiên 3.000 dòng, cắt bỏ các điểm ngoại lệ biên (percentile 1% - 99%) để tránh kéo dãn trục tọa độ, loại bỏ biến rời rạc `Floors` để tránh các đường sọc thẳng đứng nhiễu, vẽ ma trận phân tán `pairplot` cho các biến: Area, Frontage, Access Road và Price.
*   **Đồ thị thể hiện gì:**
    *   Vẽ một ma trận các biểu đồ: Các ô trên đường chéo chính hiển thị biểu đồ phân phối mật độ (KDE) của từng biến đơn lẻ. Các ô còn lại là biểu đồ phân tán (Scatter plot) thể hiện mối quan hệ chéo giữa từng cặp biến.
    *   **Đường chéo chính:** Cho thấy hình dạng phân phối mật độ của Area, Frontage, Access Road và Price sau khi lọc nhiễu biên.
    *   **Các ô phân tán:** Cho thấy các đám mây điểm biểu thị tương quan đồng thời giữa các biến. Ví dụ: mối quan hệ giữa Area và Frontage cho thấy các căn nhà có diện tích lớn thường cũng có mặt tiền rộng hơn.
*   **Ý nghĩa thực tế:** Đây là biểu đồ tổng quan cao cấp nhất cho phép người phân tích dữ liệu quan sát đồng thời sự phân phối và mối tương quan chéo của toàn bộ các biến số quan trọng nhất trong hệ thống định giá bất động sản.
