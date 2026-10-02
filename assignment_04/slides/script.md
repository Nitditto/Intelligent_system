# Giải thích kiến thức — Convolutional Neural Networks (CNN)

Tài liệu này giải thích chi tiết toàn bộ 45 slide của bộ `CNN_Presentation.pptx`, giúp người thuyết trình hiểu sâu từng khái niệm/công thức để tự diễn đạt bằng lời của mình, không cần đọc theo kịch bản. Mỗi slide có thêm mục **Ví dụ minh họa** — một liên tưởng đời thường dễ hình dung, kèm một ví dụ số/tính toán cụ thể — để người mới nắm bắt nhanh hơn.

Cấu trúc theo 5 phần của bộ slide:

| Phần | Slide | Nội dung |
|---|---|---|
| I | 1–3 | Mở đầu & vấn đề cốt lõi (vì sao cần CNN) |
| II | 4–16 | Khái niệm & cơ chế toán học của CNN |
| III | 17–33 | Lịch sử phát triển kiến trúc CNN (Neocognitron → ViT) |
| IV | 34–39 | Demo code: cài đặt CNN từ NumPy đến PyTorch |
| V | 40–45 | Cầu nối lý thuyết với 3 bài thực hành thực tế |

---

## PHẦN I — MỞ ĐẦU & VISION SCOPE (Slide 1–3)

### Slide 1 — Convolutional Neural Networks (CNN)

Đây là slide tiêu đề. CNN (Convolutional Neural Network — Mạng nơ-ron tích chập) là một kiến trúc mạng nơ-ron sâu chuyên xử lý dữ liệu có cấu trúc lưới không gian (điển hình nhất là ảnh 2D). Ảnh trên slide (pipeline CNN điển hình) cho thấy luồng xử lý tổng quát: ảnh đầu vào → các khối trích xuất đặc trưng (feature extraction) → lớp phân loại → nhãn đầu ra. Đây là "bản đồ" toàn bộ nội dung sẽ được giải thích chi tiết trong các slide sau.

**Ví dụ minh họa:** Hãy tưởng tượng một dây chuyền đóng gói trong nhà máy: nguyên liệu thô (ảnh) đi qua các trạm kiểm tra khác nhau (các lớp CNN), mỗi trạm chỉ soi một loại đặc điểm (màu, hình, kích thước), và trạm cuối cùng dán nhãn sản phẩm (dự đoán lớp). CNN hoạt động tương tự — không có một "con mắt thần" nhìn ra kết quả ngay, mà là một chuỗi bước xử lý tuần tự.

---

### Slide 2 — Overview

Slide này chia bài trình bày thành 4 phần (section) lớn:
1. **Overview & Vision Scope** — tại sao ảnh 2D cần một kiến trúc riêng, không thể dùng mạng nơ-ron thường (MLP).
2. **Concept & Mechanics** — các phép toán nền tảng: tích chập (convolution), stride, padding, ReLU.
3. **Model Evolution** — dòng lịch sử các kiến trúc CNN nổi bật, từ Neocognitron (1980) đến Vision Transformer (2020).
4. **Code Demo & Verification** — cài đặt thực tế bằng NumPy (từ đầu), rồi so sánh với PyTorch.

**Ví dụ minh họa:** Giống như học nấu ăn — trước tiên hiểu tại sao cần công thức đúng (Phần I), sau đó học các kỹ thuật cơ bản như thái, xào, hầm (Phần II), rồi xem cách các đầu bếp nổi tiếng qua từng thời kỳ đã cải tiến món ăn ra sao (Phần III), và cuối cùng tự tay vào bếp nấu thử (Phần IV).

---

### Slide 3 — 2D Structure vs. Vector Unrolling

Đây là câu hỏi mở đầu quan trọng nhất: **tại sao không dùng mạng nơ-ron thường (MLP — Multi-Layer Perceptron) cho ảnh?**

Một ảnh là ma trận 2D (chiều cao × chiều rộng). Để đưa vào MLP, người ta phải "duỗi thẳng" (flatten) ma trận đó thành một vector 1D. Vấn đề là:

- **Phá vỡ quan hệ không gian (Spatial Adjacency Destruction):** hai pixel nằm cạnh nhau theo chiều dọc trong ảnh gốc, sau khi flatten có thể cách nhau hàng trăm vị trí trong vector — mạng không còn "biết" chúng liền kề nhau.
- **Mất tính bất biến dịch chuyển (Translation Equivariance):** nếu một vật thể trong ảnh dịch chuyển 1 pixel, toàn bộ vector đầu vào thay đổi hoàn toàn → MLP phải học lại từ đầu cho mỗi vị trí có thể xuất hiện của vật thể, cực kỳ lãng phí.
- **Convolution giải quyết vấn đề này** bằng cách giữ nguyên cấu trúc lưới 2D và trượt cùng một bộ lọc (kernel) qua toàn bộ ảnh — không quan tâm vật thể ở đâu, kernel vẫn nhận diện được vì dùng chung trọng số.

**Ví dụ minh họa:** Hãy hình dung một ảnh 4×4 pixel, đánh số từ 0 đến 15 theo hàng: pixel số 5 và pixel số 9 nằm ngay phía trên/dưới nhau trong ảnh gốc (cùng cột, cách nhau 1 hàng), nhưng sau khi duỗi thành vector 1D, chúng cách nhau đúng 4 vị trí — với ảnh 224×224 thực tế, khoảng cách này lên tới 224 vị trí. Mạng MLP nhìn vector này hoàn toàn không biết "phần tử số 5" và "phần tử số 229" (tức 5+224) thực ra là hai pixel liền kề nhau trên ảnh.

---

## PHẦN II — KHÁI NIỆM & CƠ CHẾ TOÁN HỌC (Slide 4–16)

### Slide 4 — Networks as Function Composition

Một mạng nơ-ron sâu về bản chất là **một hàm hợp** (composite function) của nhiều hàm đơn giản xếp chồng lên nhau:

```
F(X; Θ) = f_L ∘ f_{L-1} ∘ ... ∘ f_1(X)
```

Trong đó mỗi lớp `f_l` là một biến đổi affine (nhân ma trận trọng số + cộng bias) theo sau bởi một hàm phi tuyến σ:

```
f_l(h) = σ(W_l · h + b_l)
```

Ý nghĩa: dữ liệu đầu vào X đi qua lớp 1, kết quả đi tiếp qua lớp 2, cứ thế cho đến lớp cuối L. Mỗi lớp "biến đổi" cách biểu diễn dữ liệu — từ pixel thô ở đầu vào, dần dần thành các đặc trưng trừu tượng hơn (cạnh → hình dạng → khái niệm) ở các lớp sâu, cho tới khi dữ liệu ở dạng có thể phân loại tuyến tính được ở lớp cuối.

**Ví dụ minh họa:** Giống như một dây chuyền dịch thuật qua nhiều trạm: câu tiếng Việt gốc → trạm 1 dịch sang cấu trúc ngữ pháp tạm → trạm 2 chỉnh từ vựng → trạm 3 tinh chỉnh văn phong → câu tiếng Anh hoàn chỉnh. Mỗi trạm nhận đầu ra của trạm trước làm đầu vào, không trạm nào "nhìn thấy" câu gốc trực tiếp trừ trạm đầu tiên. Với con số cụ thể: nếu X là ảnh 3 lớp (L=3), thì F(X) = f_3(f_2(f_1(X))) — dữ liệu phải đi qua đúng 3 "trạm biến đổi" trước khi ra kết quả cuối.

---

### Slide 5 — Why Non-Linearity Is Mandatory

Đây là một chứng minh toán học quan trọng: **nếu không có hàm phi tuyến, việc xếp chồng nhiều lớp là vô nghĩa.**

Giả sử tất cả các hàm kích hoạt σ là tuyến tính (σ(z) = z). Khi đó:

```
F(X) = W_L(W_{L-1}(...(W_1 X)...)) = (W_L · W_{L-1} · ... · W_1) X = W_eff · X + b_eff
```

Tích của nhiều ma trận vẫn chỉ là **một ma trận duy nhất**. Nghĩa là một mạng 100 lớp toàn tuyến tính, về mặt biểu diễn, tương đương với **một lớp hồi quy tuyến tính duy nhất** — không có thêm khả năng biểu diễn nào dù xếp bao nhiêu lớp.

Hàm kích hoạt phi tuyến (như ReLU: f(x) = max(0, x)) phá vỡ tính chất này: nó "gấp" (fold) không gian vector thành nhiều vùng quyết định tuyến tính từng đoạn (piecewise linear), và số lượng vùng này tăng theo cấp số nhân với độ sâu mạng — đây là nguồn gốc sức mạnh biểu diễn của mạng sâu.

**Ví dụ minh họa:** Giống như việc nhân nhiều lần với số 2 rồi số 3 rồi số 5 — kết quả cuối cùng luôn có thể viết gọn thành nhân với 30 (2×3×5) ngay từ đầu, không cần làm 3 bước. Ví dụ số: giả sử W_1 = 2, W_2 = 3, W_3 = 5 (các "lớp" chỉ là phép nhân vô hướng để dễ hình dung). Với đầu vào X = 10: đi qua 3 lớp tuần tự cho ra 10×2×3×5 = 300 — con số này hoàn toàn giống với việc nhân trực tiếp X với W_eff = 2×3×5 = 30 ngay từ đầu (10×30 = 300). Ba lớp "giả vờ sâu" này thực chất chỉ là một phép nhân duy nhất.

---

### Slide 6 — The MLP Parameter Explosion

Đây là minh chứng bằng số liệu cụ thể cho vấn đề nêu ở slide 3. Giả sử dùng MLP thường cho ảnh 224×224×3 (ảnh màu chuẩn):

- Số input sau khi flatten: 224 × 224 × 3 = 150,528 giá trị.
- Nếu lớp ẩn đầu tiên có 1,024 neuron, số trọng số cần học ở **riêng lớp 1** là: 150,528 × 1,024 ≈ **154 triệu tham số**.

So sánh: một bộ lọc convolution kích thước 3×3×3 (3 kênh màu) chỉ cần:
```
3 × 3 × 3 = 27 trọng số + 1 bias = 28 tham số
```

Chênh lệch là hơn **5 triệu lần**. Lý do: convolution dùng **local connectivity** (mỗi neuron chỉ "nhìn" một vùng nhỏ) và **weight sharing** (cùng một bộ lọc dùng lại cho mọi vị trí trong ảnh) — hai nguyên lý sẽ được giải thích kỹ ở slide 9.

Hệ quả của việc có quá nhiều tham số: dễ overfitting (mạng học thuộc lòng dữ liệu training thay vì tổng quát hóa), và tốn bộ nhớ GPU khổng lồ.

**Ví dụ minh họa:** 154 triệu tham số tương đương việc mỗi người trong một thành phố 1.5 triệu dân phải nhớ riêng 100 con số khác nhau — trong khi convolution chỉ cần cả thành phố cùng nhớ chung 28 con số (một tờ "hướng dẫn dùng chung"). Nếu convolution có 28 tham số là 1 trang giấy A4, thì 154 triệu tham số của MLP tương đương một chồng giấy A4 cao khoảng... hơn 15 km (giả sử 1 trang ghi được 10,000 số).

---

### Slide 7 — Hubel & Wiesel (1959)

CNN không phải phát minh thuần túy toán học — nó lấy cảm hứng từ sinh học thần kinh. Năm 1959, hai nhà khoa học thần kinh David Hubel và Torsten Wiesel (đoạt giải Nobel 1981) nghiên cứu vỏ não thị giác sơ cấp (V1 — primary visual cortex) của mèo, phát hiện:

- **Simple cells (tế bào đơn giản):** chỉ phản ứng (kích hoạt) khi nhìn thấy các cạnh có hướng cụ thể (0°, 45°, 90°...) tại một vùng nhỏ, cục bộ trong trường thị giác — không phản ứng với ánh sáng toàn trường.
- **Orientation tuning (điều hưởng theo hướng):** mỗi tế bào có "hướng ưa thích" riêng, phản ứng mạnh nhất ở góc đó và yếu dần khi góc lệch đi.
- **Complex cells (tế bào phức tạp):** tổng hợp (pool) đầu ra của nhiều simple cells lân cận, tạo ra tính bất biến dịch chuyển nhỏ.

Đây chính là nguyên mẫu sinh học cho ý tưởng **receptive field** (trường tiếp nhận cục bộ) và **pooling** trong CNN hiện đại — sẽ được hiện thực hóa lần đầu trong Neocognitron (slide 17).

**Ví dụ minh họa:** Giống như một đội bảo vệ, mỗi người chỉ canh gác đúng một cánh cửa (một vùng nhỏ) thay vì cố nhìn bao quát cả tòa nhà cùng lúc — mỗi bảo vệ (simple cell) chỉ báo động khi có người đi qua đúng cửa mình phụ trách theo đúng một hướng cụ thể (ví dụ chỉ báo động người đi từ trái sang phải). Thí nghiệm gốc: Hubel & Wiesel chiếu các vạch sáng ở nhiều góc độ khác nhau (0°, 45°, 90°...) trước mắt mèo và đo tín hiệu điện từ neuron — mỗi neuron chỉ "bừng sáng" tín hiệu khi vạch sáng đúng góc nó thích, gần như im lặng ở các góc khác.

---

### Slide 8 — 2D Convolution Mechanics

Đây là công thức toán học cốt lõi của CNN. Phép **tích chập rời rạc** (discrete convolution, thực chất trong deep learning là cross-correlation) được định nghĩa:

```
S(i,j) = ΣΣ I(i+m, j+n)·K(m,n) + b
```

Diễn giải: một ma trận nhỏ K×K gọi là **kernel** (hay filter) trượt qua từng vị trí (i,j) của ảnh đầu vào I. Tại mỗi vị trí, ta nhân từng phần tử tương ứng giữa kernel và vùng ảnh đang xét (element-wise multiplication), rồi cộng tổng lại (sum), cộng thêm bias b, ra một giá trị số duy nhất — đây là một điểm trong **feature map** đầu ra.

Kernel đóng vai trò như một "bộ dò mẫu hình": nếu vùng ảnh đang xét khớp với mẫu mà kernel được huấn luyện để nhận diện (ví dụ cạnh dọc), giá trị đầu ra sẽ lớn (kích hoạt mạnh); nếu không khớp, giá trị sẽ nhỏ hoặc âm.

**Ví dụ minh họa:** Giống như dùng một con dấu (kernel) ấn lần lượt lên từng vị trí trên một tờ giấy lớn (ảnh) — mỗi lần ấn cho ra một điểm mực đậm nhạt tùy độ khớp giữa hoa văn con dấu và hoa văn tờ giấy tại chỗ đó. Ví dụ số: với kernel dò cạnh dọc đơn giản 1×3 là [-1, 0, 1], áp lên vùng ảnh có giá trị pixel [2, 2, 8] (từ tối sang sáng — có cạnh): kết quả = (-1)×2 + 0×2 + 1×8 = 6 (giá trị lớn — phát hiện cạnh). Áp lên vùng đồng nhất [5, 5, 5] (không có cạnh): kết quả = (-1)×5 + 0×5 + 1×5 = 0 (không phát hiện gì).

---

### Slide 9 — Two Core CNN Axioms

Hai nguyên lý (axiom) làm nên hiệu quả tham số của CNN, đã nhắc ở slide 6:

1. **Local Connectivity (kết nối cục bộ):** mỗi neuron đầu ra chỉ kết nối với một vùng nhỏ K×K của đầu vào (không phải toàn bộ ảnh như MLP). Điều này dựa trên quan sát thực tế: các pixel ở gần nhau thường có tương quan mạnh (ví dụ cùng thuộc một cạnh hoặc bề mặt), còn pixel ở xa nhau ít liên quan.

2. **Weight Sharing (chia sẻ trọng số):** cùng một bộ kernel (cùng bộ trọng số) được dùng lại cho **mọi vị trí** trong ảnh, thay vì học một bộ trọng số riêng cho từng vị trí. Điều này tạo ra tính chất **translation equivariance** (bất biến dịch chuyển): nếu đầu vào dịch chuyển, đầu ra dịch chuyển tương ứng theo, không cần học lại — công thức: f(g(x)) = g(f(x)) với g là phép dịch chuyển.

Kết quả: thay vì học hàng triệu trọng số độc lập, mạng chỉ cần học một bộ kernel nhỏ, dùng chung khắp ảnh — hiệu quả tham số vượt trội.

**Ví dụ minh họa:** Local connectivity giống như đọc sách bằng cách chỉ nhìn qua một khung cửa sổ nhỏ di chuyển dọc trang giấy, thay vì cố nhìn cả trang cùng lúc. Weight sharing giống như một máy dò kim loại ở sân bay — cùng một máy, cùng một cách dò, được dùng cho mọi hành khách đi qua, chứ không phải chế tạo riêng một máy dò cho mỗi người. Nếu một chiếc kim loại (đối tượng cần phát hiện) nằm ở vị trí A hay vị trí B trong hành lý, máy vẫn phát hiện được như nhau — không cần "học lại" cho từng vị trí.

---

### Slide 10 — Stride & Padding

Hai siêu tham số (hyperparameter) kiểm soát kích thước không gian của đầu ra convolution:

**Công thức tổng quát:**
```
H_out = ⌊(H_in − K + 2P) / S⌋ + 1
```
Trong đó: H_in = kích thước đầu vào, K = kích thước kernel, P = padding, S = stride.

- **Padding (đệm biên):** thêm các hàng/cột giá trị 0 ở viền ảnh trước khi convolution.
  - *Valid padding* (P=0): không đệm gì cả → đầu ra nhỏ hơn đầu vào (co lại ở biên).
  - *Same padding* (P = (K−1)/2): đệm vừa đủ để đầu ra giữ nguyên kích thước đầu vào (H_out = H_in).

- **Stride (bước nhảy):** khoảng cách kernel di chuyển sau mỗi bước.
  - Stride = 1: trượt từng pixel một, đầu ra gần bằng kích thước đầu vào.
  - Stride > 1 (ví dụ 2): nhảy 2 pixel mỗi bước → đầu ra co lại một nửa, đồng thời mở rộng receptive field (vùng ảnh mà mỗi output "nhìn thấy") nhanh hơn.

**Ví dụ minh họa:** Stride giống như bước chân khi đi bộ đo đạc một căn phòng — bước ngắn (stride 1) đo được chi tiết hơn nhưng mất nhiều bước hơn, bước dài (stride 2) đo nhanh nhưng bỏ sót chi tiết ở giữa. Padding giống như dán thêm viền giấy trắng quanh một bức ảnh trước khi cắt, để phần rìa ảnh gốc không bị "hy sinh" khi cắt. Ví dụ số cụ thể: ảnh H_in=28, kernel K=3, không đệm (P=0), stride S=1: H_out = ⌊(28−3+0)/1⌋+1 = 26. Nếu dùng same padding (P=1): H_out = ⌊(28−3+2)/1⌋+1 = 28 — giữ nguyên kích thước.

---

### Slide 11 — Receptive Field Growth

**Receptive field** (trường tiếp nhận) là vùng ảnh gốc mà một điểm ở lớp sâu "nhìn thấy" được (thông qua các lớp convolution trước đó). Nó tăng dần khi đi qua nhiều lớp, theo công thức đệ quy:

```
RF_l = RF_{l-1} + (K_l − 1) · Π S_i
```

**Insight quan trọng của VGGNet:** thay vì dùng 1 kernel lớn 5×5, xếp chồng **2 kernel 3×3** liên tiếp cho ra **cùng receptive field 5×5**, nhưng:
- Số tham số: 2×(3×3) = 18 so với 1×(5×5) = 25 → tiết kiệm 28%.
- Có thêm 1 lớp phi tuyến (ReLU) ở giữa → tăng khả năng biểu diễn.

Đây là lý do các kiến trúc hiện đại (từ VGGNet trở đi) ưu tiên xếp chồng nhiều kernel nhỏ thay vì dùng ít kernel lớn.

**Ví dụ minh họa:** Giống như đứng ở tầng cao của một tòa nhà, càng lên cao thì tầm nhìn (receptive field) càng bao quát được nhiều khu phố hơn, dù mắt mỗi người vẫn chỉ có góc nhìn cố định — chỉ là vị trí đứng cao hơn thì "gộp" được nhiều góc nhìn từ các tầng dưới. Ví dụ số: lớp 1 dùng kernel 3×3 (K=3, S=1) cho RF_1 = 3. Lớp 2 cũng kernel 3×3 chồng lên: RF_2 = 3 + (3−1)×1 = 5 — đúng bằng vùng nhìn của 1 kernel 5×5 duy nhất, nhưng chỉ tốn 9+9=18 tham số thay vì 25.

---

### Slide 12 — 3D Tensor Convolutions

Ảnh thực tế không chỉ có 2 chiều không gian (H, W) mà còn có chiều **kênh** (channel) — ví dụ ảnh RGB có 3 kênh màu. Convolution thực tế hoạt động trên tensor 3D:

- Đầu vào có shape (C_in × H × W).
- Mỗi bộ lọc (filter) có shape (C_in × K_h × K_w) — **cùng số kênh với đầu vào**.
- Một filter tính convolution trên **cả 3 kênh cùng lúc**, cộng tổng lại thành **một** feature map 2D duy nhất.
- Dùng C_out filter khác nhau → cho ra C_out feature map, xếp chồng thành tensor đầu ra (C_out × H_out × W_out).

Điểm dễ nhầm: filter không phải 2D mà luôn có cùng độ sâu kênh với đầu vào — "quét" theo không gian nhưng "tổng hợp" theo kênh.

**Ví dụ minh họa:** Giống như một chiếc bánh sandwich nhiều lớp (đỏ - xanh lá - xanh dương, tương ứng 3 kênh màu) — một chiếc "dao cắt đặc biệt" (filter) cắt xuyên qua cả 3 lớp cùng lúc tại một vị trí, gộp thông tin của cả 3 lớp thành một lát cắt duy nhất (1 giá trị trong feature map). Ví dụ số: ảnh RGB có shape (3×32×32). Dùng C_out=64 filter, mỗi filter có shape (3×3×3). Mỗi filter "cắt" xuyên 3 kênh màu tại một vị trí, cho ra 1 giá trị; lặp lại 64 filter khác nhau cho ra tensor đầu ra (64×H_out×W_out) — từ 3 kênh màu ban đầu "nở" thành 64 kênh đặc trưng.

---

### Slide 13 — Counting Convolution Parameters

Công thức tổng quát tính số tham số của một lớp convolution:

```
Params = (K_h · K_w · C_in + 1) · C_out
```

Trong đó +1 là bias cho mỗi filter. Ví dụ cụ thể trên slide: K=3, C_in=64, C_out=128:

```
(3 × 3 × 64 + 1) × 128 = (576 + 1) × 128 = 577 × 128 = 73,856 tham số
```

**Điểm mấu chốt:** công thức này **không phụ thuộc vào kích thước ảnh đầu vào (H, W)** — chỉ phụ thuộc vào kích thước kernel và số kênh. Đây là lý do một model CNN có thể áp dụng cho ảnh có độ phân giải khác nhau mà không cần thay đổi số tham số (khác hẳn MLP, nơi số tham số phụ thuộc trực tiếp vào H×W như đã thấy ở slide 6).

**Ví dụ minh họa:** Giống như một khuôn bánh quy — chỉ cần đúc một lần, có thể dùng để cắt bánh quy trên bất kỳ tấm bột nào, dù tấm bột to hay nhỏ, số lượng "khuôn" (tham số) cần dùng vẫn không đổi. Ví dụ số nhỏ hơn để tính tay: K=3, C_in=3 (ảnh RGB), C_out=16: Params = (3×3×3 + 1) × 16 = (27+1)×16 = 28×16 = 448 tham số — dù áp dụng cho ảnh 32×32 hay ảnh 256×256 thì con số 448 này hoàn toàn không đổi.

---

### Slide 14 — ReLU and Sparsity

**ReLU (Rectified Linear Unit)** là hàm kích hoạt phổ biến nhất hiện nay:

```
f(x) = max(0, x)
```
Đạo hàm: f'(x) = 1 khi x > 0, và 0 khi x < 0.

**Tại sao ReLU tốt hơn Sigmoid/Tanh (vấn đề vanishing gradient)?**
Sigmoid/Tanh "bão hòa" (saturate) ở giá trị đầu vào lớn hoặc nhỏ — đạo hàm của chúng tiến gần 0 ở hai đầu. Khi lan truyền ngược (backpropagation) qua nhiều lớp, các đạo hàm gần-0 này nhân dồn lại làm gradient "biến mất" (vanishing gradient), khiến các lớp đầu mạng gần như không học được gì. ReLU giữ đạo hàm **không đổi bằng 1** với mọi x dương, nên gradient truyền ngược ổn định hơn, cho phép huấn luyện mạng rất sâu (100+ lớp).

**Sparsity (tính thưa):** vì ReLU đưa mọi giá trị âm về 0, khoảng ~50% neuron trong một lớp sẽ "tắt" (giá trị 0) ở bất kỳ thời điểm nào — tạo ra biểu diễn thưa, giống cách não bộ sinh học chỉ kích hoạt một phần nhỏ neuron cho mỗi kích thích, giúp cải thiện khả năng tổng quát hóa.

**Ví dụ minh họa:** ReLU giống như một cái van một chiều — cho dòng nước dương chảy qua nguyên vẹn, còn chặn đứng hoàn toàn dòng âm (về 0), không làm suy yếu dần như một cái lọc mờ (Sigmoid). Ví dụ số: với đầu vào x = [-3, -0.5, 0, 2, 5], ReLU cho ra [0, 0, 0, 2, 5] — 3/5 giá trị bị "tắt" về 0 (minh họa sparsity), 2 giá trị dương giữ nguyên hoàn toàn không suy giảm. So sánh: Sigmoid(5) ≈ 0.993 và Sigmoid(-3) ≈ 0.047 — cả hai đều rất gần với 0 hoặc 1, nghĩa là đạo hàm tại đó gần như bằng 0, "nghẽn" luồng gradient.

---

### Slide 15 — Max Pooling

**Max Pooling** là phép giảm chiều không gian (downsampling), thường dùng cửa sổ 2×2 với stride 2:

- Trượt cửa sổ 2×2 qua feature map, tại mỗi vị trí chỉ **giữ lại giá trị lớn nhất** trong 4 giá trị, bỏ 3 giá trị còn lại.
- Kết quả: kích thước không gian giảm một nửa mỗi chiều → giảm 75% số điểm dữ liệu.

**Ba lợi ích chính:**
1. **Translation invariance (bất biến dịch chuyển nhỏ):** nếu vật thể trong ảnh dịch chuyển vài pixel, giá trị max trong mỗi cửa sổ thường không đổi → mạng ổn định hơn trước nhiễu vị trí nhỏ.
2. **Giảm chi phí tính toán:** ít điểm dữ liệu hơn ở các lớp sau.
3. **Không có tham số học được:** pooling là phép toán xác định (deterministic), không cần học trọng số, nên không tốn thêm bộ nhớ tham số.

**Ví dụ minh họa:** Giống như việc chấm điểm một đội 4 người bằng cách chỉ lấy điểm của người cao điểm nhất trong nhóm — dù người thấp điểm trong nhóm đó "biến mất" khỏi kết quả cuối, đội vẫn được đại diện đúng bởi thành tích tốt nhất. Ví dụ số: một vùng feature map 2×2 có giá trị [[1, 5], [3, 2]] → sau max pooling chỉ còn lại 1 giá trị duy nhất là 5 (giá trị lớn nhất). Nếu vật thể dịch chuyển khiến giá trị 5 xê dịch sang ô bên cạnh trong cùng vùng 2×2, kết quả max vẫn là 5 — không đổi.

---

### Slide 16 — Hierarchical Feature Representation

Đây là bức tranh tổng hợp cho thấy CNN học đặc trưng theo cấp bậc (hierarchical), càng vào sâu càng trừu tượng:

- **Lớp nông (gần input, giống vùng V1 sinh học ở slide 7):** học được các đặc trưng đơn giản như cạnh có hướng (edges), tương phản màu sắc, các mẫu hình biên cơ bản — tương tự simple cells của Hubel & Wiesel.
- **Lớp trung gian:** kết hợp các cạnh đơn giản thành texture (kết cấu bề mặt), góc, giao điểm, đường cong.
- **Lớp sâu:** receptive field đã đủ lớn để bao trùm toàn bộ vật thể, học được các bộ dò phần vật thể cụ thể (bánh xe, mắt, khuôn mặt...) hoặc thậm chí cả mẫu hình đại diện cho toàn bộ lớp đối tượng (class template).

Đây chính là lý do CNN không cần con người thiết kế đặc trưng thủ công (hand-crafted feature) như thời trước deep learning — mạng tự học phân cấp đặc trưng từ dữ liệu.

**Ví dụ minh họa:** Giống như cách một họa sĩ vẽ chân dung theo từng bước: đầu tiên phác các nét thẳng/cong cơ bản (lớp nông = cạnh), sau đó ghép chúng thành hình mắt, mũi, miệng (lớp giữa = texture/bộ phận), cuối cùng nhìn tổng thể để nhận ra "à, đây là khuôn mặt cụ thể của một người" (lớp sâu = khái niệm hoàn chỉnh). Ví dụ cụ thể với ảnh con mèo: lớp 1 phát hiện các đường viền lông, lớp giữa ghép các đường viền thành hình tai nhọn và mắt tròn, lớp sâu nhận diện "đây là khuôn mặt mèo" dựa trên tổ hợp tai + mắt + mũi đã học được.

---

## PHẦN III — LỊCH SỬ PHÁT TRIỂN KIẾN TRÚC CNN (Slide 17–33)

*Ghi chú chung cho phần này:* mỗi slide milestone theo cùng một mẫu: (1) bối cảnh ra đời, (2) đóng góp cốt lõi, (3) ý nghĩa/ảnh hưởng tới các kiến trúc sau. Khi thuyết trình, nên nhấn mạnh **vấn đề mà kiến trúc đó giải quyết so với kiến trúc trước nó** — đây là mạch logic xuyên suốt cả phần III.

### Slide 17 — Neocognitron (1980)

Neocognitron của Kunihiko Fukushima (1980) là **tổ tiên tính toán trực tiếp** của CNN hiện đại, hiện thực hóa ý tưởng sinh học của Hubel & Wiesel (slide 7):

- **S-cells (Simple cells):** trích xuất đặc trưng cục bộ có hướng — tương đương với "convolution layer" hiện đại.
- **C-cells (Complex cells):** tổng hợp (pool) đầu ra của các S-cells lân cận, tạo khả năng chịu được dịch chuyển nhỏ — tương đương "pooling layer" hiện đại.
- Kiến trúc xen kẽ S-layer và C-layer, huấn luyện bằng cơ chế tự tổ chức (self-organizing), **chưa có backpropagation** (backprop chỉ phổ biến sau này).

Đây là mô hình tính toán đầu tiên có khả năng nhận diện mẫu hình bị biến dạng/dịch chuyển thông qua cấu trúc receptive field phân cấp.

**Ví dụ minh họa:** Giống như bản thiết kế phác thảo đầu tiên của một chiếc ô tô — chưa có động cơ hoàn chỉnh (backpropagation), nhưng đã có đúng hình dáng khung xe (cặp S-layer/C-layer) mà mọi ô tô hiện đại sau này đều dựa trên đó để phát triển. Cụ thể: nếu Neocognitron được huấn luyện nhận diện chữ số viết tay, S-cells ở lớp đầu học phát hiện các nét gạch chéo, gạch ngang cơ bản; C-cells gộp lại để "chấp nhận" một chút xê dịch trong nét chữ của từng người viết khác nhau.

---

### Slide 18 — LeNet-5 (1998)

LeNet-5 của Yann LeCun (1998) là **CNN hiện đại đầu tiên** huấn luyện end-to-end bằng backpropagation:

- Kiến trúc 7 lớp xen kẽ convolution và subsampling (pooling), kết thúc bằng các lớp fully-connected.
- **Điểm đột phá:** thay vì thiết kế đặc trưng thủ công (như các hệ thống thị giác máy tính trước đó) rồi mới phân loại, LeNet-5 học **đồng thời cả trích xuất đặc trưng lẫn phân loại** trong cùng một quá trình tối ưu bằng gradient.
- **Ứng dụng thực tế đã được kiểm chứng:** được các ngân hàng Mỹ triển khai để đọc chữ số viết tay trên séc, xử lý hơn 10% tổng số séc ở Bắc Mỹ vào cuối thập niên 1990 — một trong những ứng dụng thương mại thành công sớm nhất của deep learning.

**Ví dụ minh họa:** Trước LeNet-5, nhận diện chữ viết tay giống như phải thuê 2 nhóm riêng biệt: nhóm 1 tự tay vẽ ra các "quy tắc" nhận diện nét chữ (thiết kế đặc trưng thủ công), nhóm 2 mới dùng các quy tắc đó để phân loại — 2 nhóm không giao tiếp, làm việc kém ăn khớp. LeNet-5 giống như thuê một nhóm duy nhất tự học cả quy tắc lẫn cách phân loại cùng lúc, nên hiệu quả hơn hẳn. Ứng dụng thực tế: khi bạn gửi séc ngân hàng và máy tự động đọc được số tiền viết tay trên đó — đó chính xác là công nghệ LeNet-5 đã làm từ những năm 1990.

---

### Slide 19 — AlexNet (2012)

AlexNet là **cột mốc khởi đầu làn sóng deep learning hiện đại**, chiến thắng cuộc thi ImageNet 2012:

- **Kết quả đột phá:** đạt top-5 error 16.4%, bỏ xa đội về nhì (26.2%) — một khoảng cách chưa từng có trong lịch sử cuộc thi.
- **Bộ công cụ kỹ thuật hiện đại lần đầu kết hợp:**
  - ReLU thay Sigmoid/Tanh → huấn luyện nhanh hơn ~6 lần.
  - Dropout (tỉ lệ 0.5) → giảm overfitting bằng cách ngẫu nhiên "tắt" neuron khi huấn luyện.
  - Data augmentation (tăng cường dữ liệu) → tạo thêm biến thể ảnh training để tăng tính tổng quát.
- **Tận dụng GPU:** huấn luyện song song trên 2 GPU NVIDIA GTX 580 — chứng minh tính khả thi của việc dùng GPU để huấn luyện mạng sâu quy mô lớn, mở đường cho toàn bộ hướng nghiên cứu sau này.

**Ví dụ minh họa:** "top-5 error 16.4%" nghĩa là: cho AlexNet đoán 5 khả năng có thể cho mỗi ảnh, chỉ có 16.4% số ảnh mà đáp án đúng KHÔNG nằm trong 5 khả năng đó — tức đúng 83.6% số lần. Đội về nhì chỉ đúng 73.8% số lần (100%-26.2%) — khoảng cách gần 10 điểm phần trăm là rất lớn trong một cuộc thi mà các năm trước chỉ cải thiện vài phần trăm mỗi năm. Dropout 0.5 giống như trong một lớp học 20 học sinh, mỗi buổi ngẫu nhiên cho 10 em nghỉ học — buộc kiến thức phải phân tán đều, không dồn hết vào một vài "học sinh giỏi" (neuron) duy nhất.

---

### Slide 20 — VGGNet (2014)

VGGNet chứng minh sức mạnh của **sự đơn giản và đồng nhất trong thiết kế**:

- **Chuẩn hóa kernel 3×3:** thay vì dùng nhiều kích thước kernel khác nhau (như AlexNet), VGGNet chỉ dùng kernel 3×3 xuyên suốt, xếp chồng nhiều lớp để đạt receptive field lớn (đã giải thích cơ chế ở slide 11) — tiết kiệm 28% tham số so với dùng kernel lớn trực tiếp.
- **Khối module đồng nhất:** Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2), lặp lại nhiều lần — dễ hiểu, dễ mở rộng, dễ tùy biến độ sâu (VGG-16 có 16 lớp trọng số, VGG-19 có 19 lớp).
- **Ảnh hưởng lâu dài:** VGG-16/19 trở thành backbone tiêu chuẩn cho transfer learning (dùng lại mạng đã huấn luyện sẵn trên ImageNet, rồi fine-tune cho bài toán khác) trong suốt nhiều năm sau đó.

**Ví dụ minh họa:** Giống như xây nhà bằng gạch tiêu chuẩn (chỉ 1 kích cỡ gạch 3×3) thay vì dùng nhiều loại vật liệu khác nhau (gạch, đá, gỗ như AlexNet) — thiết kế đơn giản hơn, dễ lắp ráp thành nhiều tầng, và một khi đã có "bản vẽ gạch chuẩn" thì ai cũng có thể dùng lại để xây các công trình khác (transfer learning). VGG-16 nghĩa là 16 lớp có trọng số xếp chồng — như một tòa nhà 16 tầng, mỗi tầng dùng đúng loại gạch 3×3 giống hệt nhau.

---

### Slide 21 — GoogLeNet / Inception (2014)

GoogLeNet (còn gọi Inception v1) giải quyết bài toán khác: **xử lý đa tỉ lệ (multi-scale) một cách hiệu quả**.

- **Inception module:** thay vì chọn một kích thước kernel cố định, module này chạy **song song nhiều kernel kích thước khác nhau** (1×1, 3×3, 5×5) cùng với một nhánh pooling, rồi **ghép nối (concatenate)** kết quả theo chiều kênh. Nhờ đó mạng tự học được tỉ lệ đặc trưng nào phù hợp nhất cho từng vùng ảnh.
- **Bottleneck 1×1:** trước khi đưa vào các filter tốn kém (3×3, 5×5), dùng một lớp convolution 1×1 để giảm số kênh trước — giảm đáng kể chi phí tính toán mà không mất nhiều thông tin.
- **Global Average Pooling (GAP):** thay vì dùng các lớp fully-connected lớn ở cuối mạng (chiếm phần lớn tham số, như đã thấy ở slide 33), GoogLeNet lấy trung bình toàn bộ mỗi feature map thành 1 giá trị duy nhất — giảm tổng tham số từ 138 triệu (VGG) xuống chỉ còn 5 triệu.

**Ví dụ minh họa:** Giống như một đội trinh sát dùng nhiều loại thiết bị soi cùng lúc — ống nhòm tầm xa (kernel 5×5 để thấy vật thể lớn), kính lúp tầm gần (kernel 1×1 để thấy chi tiết nhỏ), và mắt thường (kernel 3×3 ở giữa) — rồi tổng hợp báo cáo từ cả 3 nguồn thay vì chỉ dùng một loại thiết bị duy nhất. Bottleneck 1×1 giống như việc tóm tắt một báo cáo dài trước khi gửi cho cấp trên đọc kỹ — giảm khối lượng nhưng vẫn giữ ý chính. Con số cụ thể: 138 triệu tham số của VGG xuống còn 5 triệu của GoogLeNet — giảm gần 28 lần, chủ yếu nhờ bỏ các lớp fully-connected cồng kềnh ở cuối.

---

### Slide 22 — ResNet (2015)

ResNet giải quyết một nghịch lý gọi là **bài toán suy thoái (degradation problem)**: khi mạng vượt quá khoảng 20 lớp, độ chính xác **không tăng mà còn giảm** — kể cả trên tập training (không phải overfitting, mà là do vanishing gradient khiến các lớp sâu không học được gì).

**Giải pháp — Residual Learning (học phần dư):**
```
H(x) = F(x) + x
```
Thay vì bắt mỗi khối lớp phải học trực tiếp ánh xạ mong muốn H(x), ResNet để khối đó chỉ học **phần chênh lệch (residual)** F(x) = H(x) − x, rồi cộng lại với đầu vào gốc x thông qua một **kết nối tắt (skip connection / identity connection)**.

**Vì sao điều này giúp gradient:** đạo hàm của H theo x là dH/dx = dF/dx + 1 — luôn có thành phần "+1" đảm bảo gradient không bao giờ về 0 hoàn toàn, cho phép lan truyền ngược trực tiếp về các lớp đầu mà không bị suy giảm qua hàng trăm lớp.

**Kết quả:** ResNet mở rộng được tới 152 lớp, thắng ImageNet 2015 với top-5 error kỷ lục 3.57% — vượt qua cả mức lỗi ước tính của con người (~5.1%).

**Ví dụ minh họa:** Giống như đi thang bộ có thêm một đường trượt song song (skip connection) bên cạnh — nếu bạn mệt (gradient yếu) ở một đoạn cầu thang nào đó, vẫn có đường trượt để đi thẳng xuống, không bị "kẹt cứng" giữa chừng. Về bài toán học: thay vì bắt học sinh phải "vẽ lại từ đầu" một bức tranh giống hệt bản gốc (học H(x) trực tiếp), giáo viên chỉ yêu cầu vẽ thêm phần khác biệt so với bản gốc rồi ghép vào (học F(x) = H(x) − x) — dễ học hơn nhiều nếu phần khác biệt đó nhỏ. Ví dụ số về gradient: nếu không có skip connection, qua 10 lớp mà mỗi lớp gradient chỉ còn 0.5 lần, gradient cuối chỉ còn 0.5^10 ≈ 0.001 (gần như biến mất); có skip connection, mỗi bước luôn cộng thêm ít nhất "1", nên gradient không bao giờ về 0 hoàn toàn.

---

### Slide 23 — DenseNet (2017)

DenseNet đẩy ý tưởng kết nối tắt của ResNet đi xa hơn: thay vì **cộng** (addition) đầu vào vào đầu ra như ResNet, DenseNet **ghép nối** (concatenation) đặc trưng của **tất cả** các lớp trước đó:

```
X_l = H_l([X_0, X_1, ..., X_{l-1}])
```

Mỗi lớp nhận đầu vào là toàn bộ feature map của mọi lớp trước nó (ghép theo chiều kênh), và chỉ cần tạo ra thêm **k kênh đặc trưng mới** (gọi là growth rate, thường k = 12 đến 32) — không cần học lại các đặc trưng đã có sẵn từ các lớp trước.

**Lợi ích:**
- Tránh học lặp lại các bộ lọc dư thừa (redundant), giữ số tham số thấp dù mạng dày đặc kết nối.
- Gradient từ hàm loss có thể truyền trực tiếp đến mọi lớp trước đó, không chỉ qua đường skip đơn lẻ như ResNet — triệt để giải quyết vanishing gradient.

**Ví dụ minh họa:** Giống như một nhóm làm việc chung một tài liệu Google Docs — mỗi thành viên (mỗi lớp) chỉ cần bổ sung phần đóng góp mới của mình vào tài liệu chung, không cần chép lại toàn bộ nội dung người khác đã viết trước đó; ai cũng đọc được toàn bộ lịch sử đóng góp trước mình. Ví dụ số: nếu growth rate k=12, sau 5 lớp DenseNet, lớp thứ 5 sẽ nhận đầu vào có số kênh bằng tổng đóng góp của các lớp trước: kênh gốc + 12 + 12 + 12 + 12 = kênh gốc + 48 — mỗi lớp chỉ thêm 12 kênh mới thay vì phải "gánh" lại toàn bộ từ đầu.

---

### Slide 24 — U-Net (2015)

U-Net chuyển bài toán từ **phân loại toàn ảnh** (image → 1 nhãn) sang **phân đoạn ngữ nghĩa cấp pixel** (image → mặt nạ phân đoạn, mỗi pixel có 1 nhãn riêng) — ứng dụng nhiều trong y tế (phân đoạn khối u), xe tự lái (phân đoạn làn đường)...

**Kiến trúc hình chữ U đối xứng:**
- **Encoder (nhánh co lại):** giống CNN phân loại thông thường, dùng pooling để co dần kích thước không gian, mở rộng receptive field, nắm bắt ngữ cảnh (context) toàn cục.
- **Decoder (nhánh mở rộng):** đối xứng với encoder, dùng upsampling để phóng to dần trở lại kích thước ảnh gốc, khôi phục độ phân giải không gian.
- **Skip connection ngang (horizontal):** copy trực tiếp feature map từ encoder sang decoder ở cùng độ phân giải — giúp khôi phục các chi tiết biên sắc nét mà encoder đã "làm mờ" trong quá trình co lại (vì chỉ dựa vào decoder thì biên vật thể sẽ bị nhòe).

**Ví dụ minh họa:** Giống như việc vẽ bản đồ chi tiết một thành phố: trước tiên bay drone lên cao để nhìn tổng quan bố cục các khu vực (encoder co lại, nắm ngữ cảnh), sau đó bay drone xuống thấp dần để tô lại chi tiết từng con đường (decoder mở rộng) — nhưng nếu chỉ dựa vào lúc bay xuống mà quên hẳn các mốc đã ghi nhận lúc bay cao, bản đồ chi tiết sẽ bị lệch; vì vậy cần "ghim" lại các mốc quan trọng đã thấy ở độ cao tương ứng (skip connection). Ứng dụng thực tế: bác sĩ dùng U-Net để tô màu chính xác đường viền một khối u trên ảnh MRI — mỗi pixel ảnh được gán nhãn "là khối u" hoặc "không phải khối u".

---

### Slide 25 — MobileNet (2017)

MobileNet hướng tới mục tiêu khác hẳn: **hiệu quả tính toán cho thiết bị di động/nhúng**, không phải độ chính xác tối đa.

**Depthwise Separable Convolution (tích chập tách kênh):** tách một phép convolution chuẩn thành 2 bước:
1. **Depthwise convolution:** mỗi kênh đầu vào được lọc **riêng biệt** bằng một kernel không gian (không trộn kênh).
2. **Pointwise convolution (1×1):** dùng convolution 1×1 để trộn (kết hợp) thông tin giữa các kênh.

**Tiết kiệm tính toán:** phép tách này giảm số FLOPs (phép tính dấu phẩy động) khoảng **8-9 lần** so với convolution chuẩn, trong khi độ chính xác chỉ giảm khoảng ~1% — một đánh đổi (trade-off) rất tốt.

**Ứng dụng:** cho phép chạy thị giác máy tính thời gian thực trên điện thoại, robot nhúng, thiết bị IoT biên (edge) có tài nguyên hạn chế.

**Ví dụ minh họa:** Giống như quy trình pha một ly trà sữa: thay vì một người làm tất cả mọi công đoạn cùng lúc (convolution chuẩn — vừa pha trà vừa trộn topping vừa lắc đều), MobileNet tách thành 2 bước tuần tự rõ ràng — một người chuyên pha trà (depthwise, xử lý riêng từng "nguyên liệu"/kênh), một người chuyên trộn các nguyên liệu lại (pointwise 1×1) — tổng thời gian ít hơn nhiều so với 1 người ôm đồm mọi việc. Con số cụ thể: giảm 8-9 lần khối lượng tính toán nhưng chỉ mất khoảng 1% độ chính xác — đây là lý do camera nhận diện khuôn mặt mở khóa điện thoại chạy được mượt mà ngay trên chip điện thoại, không cần gửi dữ liệu lên server.

---

### Slide 26 — EfficientNet (2019)

EfficientNet đặt câu hỏi có hệ thống: **nên mở rộng mạng theo chiều nào để hiệu quả nhất** — độ sâu (depth — số lớp), độ rộng (width — số kênh mỗi lớp), hay độ phân giải đầu vào (resolution)?

**Compound Scaling (mở rộng kết hợp):** thay vì chỉ tăng một chiều duy nhất, EfficientNet mở rộng **cả 3 chiều đồng thời** theo tỉ lệ cân bằng, ràng buộc bởi công thức:

```
α · β² · γ² ≈ 2
```
(với α = hệ số độ sâu, β = hệ số độ rộng, γ = hệ số độ phân giải, kiểm soát chung bởi một hệ số phi φ duy nhất)

**Bằng chứng thực nghiệm:** nhóm tác giả chứng minh mở rộng riêng lẻ một chiều nhanh chóng "bão hòa" (độ chính xác không tăng thêm nữa dù tăng thêm tài nguyên), trong khi mở rộng cân bằng cả 3 chiều cho kết quả tốt hơn ở cùng mức chi phí tính toán.

**Kết quả:** EfficientNet-B7 đạt 84.3% top-1 accuracy trên ImageNet, đồng thời nhỏ hơn 8.4 lần và nhanh hơn 6.1 lần so với các mạng SOTA (state-of-the-art) trước đó.

**Ví dụ minh họa:** Giống như nấu một nồi lẩu ngon — không phải cứ đổ thêm thật nhiều thịt (chỉ tăng 1 chiều) là ngon hơn, mà cần tăng đồng thời cả thịt, rau, và nước dùng theo đúng tỉ lệ cân bằng thì nồi lẩu mới đậm đà hơn hẳn. Tương tự, một tòa nhà chỉ xây cao thêm (depth) mà không mở rộng diện tích sàn (width) và không tăng chất lượng vật liệu (resolution) sẽ nhanh chóng gặp giới hạn — như trần nhà quá thấp so với diện tích, không tận dụng được hết không gian.

---

### Slide 27 — SENet (2018)

SENet giới thiệu cơ chế **attention theo kênh** (channel attention) — cho phép mạng tự động "chú ý" nhiều hơn vào các kênh đặc trưng quan trọng, giảm chú ý các kênh ít liên quan, thông qua khối Squeeze-and-Excitation:

1. **Squeeze (nén):** dùng Global Average Pooling để nén mỗi kênh (một feature map 2D) thành **1 con số duy nhất** đại diện — tóm tắt thông tin không gian toàn cục của kênh đó thành một vector mô tả z.
2. **Excitation (kích hoạt):** đưa vector z qua một MLP nhỏ 2 lớp (FC → ReLU → FC → Sigmoid, với tỉ lệ giảm chiều r=16 ở lớp ẩn) để học mối quan hệ phi tuyến giữa các kênh, cho ra một điểm số s_c ∈ [0,1] cho mỗi kênh.
3. **Recalibration (tái cân chỉnh):** nhân từng kênh gốc với điểm số tương ứng: Y = X ⊙ s — khuếch đại các kênh quan trọng (điểm gần 1), làm yếu các kênh ít liên quan (điểm gần 0).

**Ví dụ minh họa:** Giống như một ban giám khảo cuộc thi ẩm thực chấm 64 món ăn (64 kênh) — đầu tiên nếm thử và tóm tắt mỗi món thành 1 điểm số duy nhất (squeeze), sau đó dựa vào các điểm số này để quyết định món nào xứng đáng được "spotlight" nhiều hơn trên bàn tiệc (excitation cho điểm s_c), rồi thực sự tăng khẩu phần cho món điểm cao, giảm khẩu phần món điểm thấp (recalibration Y = X⊙s). Ví dụ số: nếu một kênh có điểm số s_c = 0.9 (rất quan trọng) và một kênh khác có s_c = 0.1 (ít quan trọng), sau khi nhân lại, kênh đầu gần như được giữ nguyên cường độ (nhân 0.9), kênh sau bị giảm mạnh xuống chỉ còn 10% cường độ ban đầu.

---

### Slide 28 — CBAM (2018)

CBAM (Convolutional Block Attention Module) mở rộng ý tưởng attention của SENet thêm một chiều nữa — không chỉ "kênh nào quan trọng" (what) mà còn "vị trí nào quan trọng" (where):

1. **Channel Attention** (giống cơ chế SENet slide 27): học "cái gì" (what) quan trọng — kênh đặc trưng nào cần chú ý.
2. **Spatial Attention** (nối tiếp sau): học "ở đâu" (where) quan trọng — vùng không gian nào trong feature map cần chú ý, dùng cả Average Pooling (nắm bắt đặc trưng nền/tổng thể) và Max Pooling (nắm bắt đặc trưng vật thể nổi bật) kết hợp.
3. **Plug-and-play:** module này rất nhẹ, tốn rất ít tham số thêm, có thể chèn trực tiếp vào các kiến trúc có sẵn như ResNet hay MobileNet mà không cần thiết kế lại từ đầu.

**Ví dụ minh họa:** Nếu SENet giống như một giám khảo hỏi "món nào ngon nhất" (kênh nào quan trọng), thì CBAM giống như hỏi thêm câu thứ hai "trong đĩa món đó, phần nào ngon nhất — phần thịt hay phần nước sốt" (vị trí nào trong ảnh quan trọng). Ví dụ thực tế: khi CNN nhận diện một con chó trong ảnh có nền cỏ xanh, channel attention giúp mạng chú ý nhiều hơn vào các kênh đặc trưng "lông", "tai" thay vì kênh đặc trưng "màu xanh lá", còn spatial attention giúp mạng khoanh vùng chú ý vào đúng vị trí con chó trong ảnh, bỏ qua vùng nền cỏ xung quanh.

---

### Slide 29 — Vision Transformer (2020)

ViT là bước ngoặt lớn: áp dụng kiến trúc **Transformer** (vốn thành công trong NLP, ví dụ BERT/GPT) cho ảnh, **loại bỏ hoàn toàn convolution**.

**Cơ chế:**
1. **Patch Tokenization:** chia ảnh 224×224×3 thành 196 patch nhỏ không chồng lấn (16×16 mỗi patch), mỗi patch được chiếu tuyến tính (linear projection) thành một token embedding — tương tự cách Transformer trong NLP xử lý từng từ (word) thành token.
2. **Self-Attention thuần túy:** các token tương tác với nhau qua cơ chế Multi-Head Self-Attention:
```
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```
Cơ chế này cho phép **mỗi token "nhìn thấy" mọi token khác ngay từ lớp đầu tiên** — khác hẳn CNN, nơi phải xếp chồng nhiều lớp convolution mới mở rộng được receptive field ra toàn ảnh (như đã thấy ở slide 11, 32).

3. **Đánh đổi:** ViT bỏ hoàn toàn các thiên kiến quy nạp không gian (spatial inductive bias — local connectivity, weight sharing của CNN), đổi lấy khả năng biểu diễn cao hơn — nhưng cần lượng dữ liệu huấn luyện khổng lồ (ví dụ JFT-300M — 300 triệu ảnh) mới vượt được CNN truyền thống.

**Ví dụ minh họa:** Giống như đọc một cuốn truyện tranh bằng cách cắt thành 196 mảnh ghép nhỏ (patch), rồi thay vì đọc lần lượt từ mảnh đầu đến mảnh cuối, mỗi mảnh ghép có thể "nhìn thấy" và tham chiếu tới tất cả các mảnh ghép khác cùng lúc để hiểu ngữ cảnh (self-attention) — khác với CNN vốn giống đọc truyện theo kiểu lật từng trang liên tiếp, phải đọc hết các trang gần mới dần hiểu được toàn cảnh câu chuyện. Ví dụ số: ảnh 224×224 chia thành patch 16×16 sẽ có (224/16)×(224/16) = 14×14 = 196 patch — đúng con số đã nêu trong slide.

---

### Slide 30 — CNN vs. Vision Transformer

Slide tổng hợp so sánh 2 triết lý thiết kế:

- **Thiên kiến quy nạp (Inductive Bias):** CNN "cài cứng" các giả định không gian (locality, weight sharing) vào kiến trúc → học hiệu quả với ít dữ liệu. ViT không có giả định này, phải tự học mọi quan hệ từ dữ liệu thô → cần rất nhiều dữ liệu.
- **Độ phức tạp tính toán:** convolution có độ phức tạp tuyến tính theo số pixel O(K²HW) — kernel size cố định. Self-attention chuẩn có độ phức tạp **bậc hai** theo số token O(N²) — vì mỗi token phải so sánh với mọi token khác, tốn kém hơn nhiều khi ảnh/chuỗi dài.
- **Xu hướng hội tụ hiện đại:** các kiến trúc mới nhất kết hợp cả hai thế giới. **ConvNeXt** hiện đại hóa CNN thuần túy bằng cách áp dụng các thiết kế học từ Transformer (ví dụ kernel lớn hơn, chuẩn hóa layer norm). **MaxViT** pha trộn convolution cục bộ với self-attention toàn cục trong cùng một kiến trúc.

**Ví dụ minh họa:** CNN giống như một học sinh được thầy cô cho sẵn công thức và mẹo làm bài (inductive bias) — học nhanh, cần ít bài tập luyện để giỏi. ViT giống như một học sinh phải tự mày mò tìm ra công thức từ hàng nghìn bài tập mẫu, không được gợi ý trước — ban đầu chậm hơn nhưng nếu có đủ bài tập (dữ liệu khổng lồ), học sinh này có thể tự tìm ra những công thức tối ưu hơn cả công thức thầy cô dạy. Ví dụ số về độ phức tạp: với ảnh có N=196 token (patch), self-attention cần tính N² = 196² ≈ 38,416 phép so sánh cặp — nếu ảnh lớn hơn thành N=784 (patch nhỏ hơn), số phép so sánh nhảy vọt lên 784² ≈ 614,656 — tăng gấp 16 lần dù số token chỉ tăng 4 lần.

---

### Slide 31 — The CNN Pipeline

Đây là slide tổng kết lại **pipeline chuẩn** mà mọi CNN cổ điển (từ LeNet đến VGG) đều tuân theo, sau khi đã học xong tất cả thành phần riêng lẻ:

1. **Feature Extraction Backbone (khung trích xuất đặc trưng):** các khối Conv → ReLU → Pooling xen kẽ lặp lại nhiều lần, trích xuất đặc trưng ngày càng trừu tượng (như đã thấy ở slide 16).
2. **Chuyển tiếp sang phân loại:** dùng Global Pooling hoặc Flatten để nén tensor đặc trưng nhiều chiều (C×H×W) thành một vector 1D.
3. **Decision Head (đầu ra quyết định):** các lớp fully-connected, kết thúc bằng Softmax để tính xác suất cho từng lớp (class) đầu ra.

Slide này đóng vai trò "chốt lại" toàn bộ Phần II và III trước khi đi vào phân tích chi tiết luồng dữ liệu (tensor shape) ở 2 slide tiếp theo.

**Ví dụ minh họa:** Giống như quy trình khám bệnh tổng quát: bước 1 là các xét nghiệm chuyên sâu để thu thập nhiều chỉ số khác nhau (feature extraction — máu, nước tiểu, X-quang...), bước 2 là tổng hợp toàn bộ chỉ số thành một bệnh án tóm tắt (flatten thành vector 1D), bước 3 là bác sĩ dựa vào bệnh án đó để đưa ra chẩn đoán cuối cùng dưới dạng xác suất cho từng loại bệnh (Softmax — ví dụ 70% khả năng cảm cúm, 20% viêm họng, 10% dị ứng).

---

### Slide 32 — Tensor Shape Across Depth

Đây là quy luật thiết kế phổ biến trong hầu hết CNN cổ điển: **không gian co lại, kênh mở rộng** khi đi sâu vào mạng:

- **Co lại về không gian:** 224 → 112 → 56 → 28 → 7 (qua các lớp pooling/stride, như đã học ở slide 10, 15).
- **Mở rộng về kênh:** 3 → 64 → 128 → 256 → 512 (số filter tăng dần ở mỗi khối).

**Ý nghĩa:** ở các lớp nông, mạng cần độ phân giải không gian cao để nắm chi tiết cục bộ, nhưng chưa cần nhiều "loại" đặc trưng khác nhau (ít kênh). Càng vào sâu, độ phân giải không gian không còn quan trọng bằng khả năng biểu diễn đa dạng các khái niệm trừu tượng — nên đánh đổi độ phân giải lấy số lượng kênh. Đây là sự "đánh đổi thông tin": bỏ bớt dư thừa pixel-level, đổi lấy các vector đặc trưng ngữ nghĩa trừu tượng, bất biến với dịch chuyển.

**Ví dụ minh họa:** Giống như quá trình biên tập một bộ phim tài liệu: đoạn thô ban đầu (224×224, 3 kênh màu) có độ phân giải rất cao nhưng thông tin ý nghĩa còn thô sơ (chỉ là điểm ảnh màu). Qua từng vòng biên tập (mỗi khối conv-pool), độ phân giải hình ảnh giảm dần (không cần xem từng khung hình chi tiết nữa) nhưng số lượng "lớp chú thích ý nghĩa" tăng lên (âm thanh, phụ đề, hiệu ứng, bình luận...) — cuối cùng đổi lấy một sản phẩm cô đọng, giàu ý nghĩa hơn nhiều so với đoạn thô ban đầu dù kích thước file (không gian) nhỏ hơn.

---

### Slide 33 — FLOPs vs. Parameters

Một điểm rất dễ hiểu nhầm: **nơi tốn tham số** và **nơi tốn phép tính** trong CNN là 2 vị trí khác nhau:

- **Compute bottleneck (nút thắt tính toán — FLOPs):** các lớp **convolution** chiếm hơn 90% tổng số phép tính dấu phẩy động, vì phải trượt kernel qua toàn bộ lưới không gian có độ phân giải cao.
- **Memory bottleneck (nút thắt bộ nhớ — Parameters):** các lớp **fully-connected** chiếm hơn 90% tổng số trọng số, vì đây là phép nhân ma trận đặc (dense) kết nối mọi neuron với mọi neuron.

**Giải pháp hiện đại:** thay lớp fully-connected cồng kềnh bằng Global Average Pooling (đã thấy ở GoogLeNet, slide 21) — loại bỏ hàng triệu tham số dư thừa mà không ảnh hưởng nhiều đến độ chính xác.

**Ví dụ minh họa:** Giống như một nhà máy có 2 khu vực riêng biệt gây tốn kém khác nhau — khu vực dây chuyền sản xuất (convolution) tốn nhiều điện năng vận hành liên tục (FLOPs cao vì phải "quét" qua từng sản phẩm), trong khi khu vực kho lưu trữ nguyên liệu (fully-connected) lại tốn nhiều diện tích mặt bằng để chứa đồ (tham số cao) dù không tốn nhiều điện khi hoạt động. Hai vấn đề khác nhau cần hai giải pháp khác nhau — không thể giải quyết "chi phí điện" bằng cách "giảm diện tích kho" và ngược lại.

---

## PHẦN IV — DEMO CODE & KIỂM CHỨNG (Slide 34–39)

### Slide 34 — Three Levels of Implementation

Slide này giới thiệu 3 mức độ trừu tượng hóa (abstraction) khi cài đặt CNN trong thực tế, sẽ được minh họa chi tiết ở các slide sau:

1. **NumPy (từ đầu — Level 1):** tự viết vectorized im2col cho forward pass, tự viết col2im cho backward pass (backprop thủ công bằng tay). Giá trị chính là **tính minh bạch tuyệt đối** — thấy rõ từng phép toán bên trong, phù hợp học tập nhưng chậm.
2. **Keras (khai báo — Level 2):** dùng các lớp cấp cao như `layers.Conv2D`, `layers.MaxPooling2D`, đồ thị tính toán (forward + backward) được biên dịch tự động. Ưu tiên tốc độ phát triển (rapid prototyping).
3. **PyTorch (mệnh lệnh — Level 3):** thực thi forward pass theo kiểu "chạy ngay" (eager execution, imperative), kết hợp tự động vi phân (autograd) — cân bằng giữa khả năng debug trực quan kiểu Python và hiệu năng cao, là chuẩn phổ biến trong nghiên cứu hiện nay.

**Ví dụ minh họa:** Giống như 3 cách nấu một món ăn: cách 1 là tự trồng rau, tự xay bột, tự làm mọi thứ từ đầu (NumPy — hiểu rõ từng nguyên liệu nhưng rất lâu); cách 2 là dùng bộ nguyên liệu đóng gói sẵn, chỉ cần làm theo công thức in trên hộp (Keras — nhanh nhưng không thấy được quy trình bên trong hộp); cách 3 là dùng nguyên liệu tươi mua ở chợ nhưng tự tay chế biến theo ý mình, vừa kiểm soát được chất lượng vừa không mất công tự trồng trọt (PyTorch — cân bằng).

---

### Slide 35 — Cross-Framework Rosetta Stone

Bảng đối chiếu trực tiếp cùng một khái niệm được viết như thế nào ở 3 framework — mục đích là chứng minh **các framework khác nhau về cú pháp (syntax) nhưng giống nhau về bản chất toán học (semantics)**:

| Khái niệm | NumPy | Keras | PyTorch |
|---|---|---|---|
| Convolution 2D | `Y_col = np.dot(W_row, X_col)` | `layers.Conv2D(32, (3,3), padding='same')` | `nn.Conv2d(in_ch, out_ch, kernel_size=3)` |
| Backward pass | `dX_col = np.dot(W_row.T, dY_col)` | `tape.gradient(loss, vars)` | `loss.backward()` |
| Cập nhật trọng số | `W -= lr * dW` | `optimizer.apply_gradients(...)` | `optimizer.step()` |

Mỗi dòng trong bảng đều là cùng một phép toán, chỉ khác cách gọi API.

**Ví dụ minh họa:** Giống như "Rosetta Stone" thật trong lịch sử — tấm đá khắc cùng một đoạn văn bản bằng 3 loại chữ viết khác nhau (chữ tượng hình Ai Cập, chữ Demotic, chữ Hy Lạp), giúp các nhà khảo cổ nhận ra rằng dù ký tự khác nhau hoàn toàn, chúng đang diễn đạt đúng một nội dung. Tương tự, `W -= lr * dW` (NumPy), `optimizer.apply_gradients(...)` (Keras), và `optimizer.step()` (PyTorch) trông rất khác nhau về mặt chữ viết code, nhưng đều thực hiện đúng một phép toán: lấy trọng số cũ trừ đi (tốc độ học × đạo hàm).

---

### Slide 36 — Forward Pass: im2col + GEMM

Đây là kỹ thuật **tăng tốc convolution bằng cách biến nó thành phép nhân ma trận** — vì phần cứng (CPU/GPU) tối ưu cực tốt cho nhân ma trận (GEMM — General Matrix Multiply) nhưng không tối ưu trực tiếp cho vòng lặp trượt cửa sổ.

**Kỹ thuật im2col (image to column):**
1. "Duỗi" (unfold) mỗi vùng receptive field (mỗi vị trí kernel sẽ trượt qua) thành **một cột** trong một ma trận lớn `x_col`.
2. Duỗi các trọng số kernel thành ma trận hàng `w_row`.
3. Tính convolution bằng **một phép nhân ma trận duy nhất**: `out = w_row @ x_col + b` — thay vì hàng nghìn phép tính vòng lặp nhỏ lẻ, giờ chỉ là 1 phép GEMM lớn, tận dụng được thư viện BLAS đã tối ưu sẵn (nhanh hơn rất nhiều).
4. Reshape kết quả trở lại thành tensor đầu ra 4 chiều.

**Ví dụ minh họa:** Giống như thay vì đi chợ mua từng món riêng lẻ nhiều lần trong ngày (vòng lặp nhỏ lẻ, chậm), ta viết ra một danh sách mua sắm đầy đủ rồi đi chợ một lần duy nhất mua hết (gộp thành một phép GEMM lớn) — tốn công chuẩn bị danh sách (bước duỗi im2col) nhưng tổng thời gian đi lại nhanh hơn hẳn. Ví dụ số nhỏ: một ảnh 4×4 với kernel 2×2, stride 1 sẽ có 3×3=9 vị trí trượt; im2col "duỗi" 9 vùng đó thành 9 cột trong 1 ma trận, rồi chỉ cần 1 phép nhân ma trận duy nhất để tính hết cả 9 vị trí, thay vì 9 vòng lặp riêng lẻ.

---

### Slide 37 — Backward Pass: col2im

Đây là phần lan truyền ngược (backpropagation) tương ứng với forward pass ở slide 36 — tính gradient để cập nhật trọng số:

1. **Gradient của bias (db):** cộng tổng gradient đầu ra theo chiều batch và không gian.
2. **Gradient của trọng số (dw):** nhân ma trận gradient đầu ra với ma trận `x_col` đã lưu từ forward pass — cũng là một phép GEMM.
3. **Gradient của đầu vào (dx):** phức tạp hơn — vì mỗi pixel đầu vào có thể tham gia vào **nhiều vùng receptive field chồng lấn** (overlapping) khi kernel trượt qua, nên **col2im** (ngược với im2col) phải **cộng dồn (accumulate)** gradient từ tất cả các vị trí đã dùng đến pixel đó, chứ không đơn thuần "gán ngược" lại.

**Ví dụ minh họa:** Giống như tính tổng tiền hoa hồng của một nhân viên bán hàng làm việc ở nhiều chi nhánh trong tháng — nếu nhân viên đó xuất hiện trong doanh số của 3 chi nhánh khác nhau (3 vị trí receptive field chồng lấn dùng cùng 1 pixel), tổng hoa hồng cuối tháng phải **cộng dồn** cả 3 khoản lại, không thể chỉ lấy một khoản duy nhất rồi bỏ qua 2 khoản còn lại. Ví dụ cụ thể: với kernel 2×2 stride 1 trên ảnh, một pixel ở giữa ảnh có thể nằm trong tới 4 vùng receptive field khác nhau (khi kernel trượt qua 4 vị trí lân cận) — gradient của pixel đó khi lan truyền ngược phải là tổng gradient đóng góp từ cả 4 vị trí này cộng lại.

---

### Slide 38 — PyTorch Implementation (nn.Module)

Slide minh họa cách viết một CNN hoàn chỉnh bằng PyTorch theo phong cách hướng đối tượng (`nn.Module`), tương phản với cách viết thủ công ở slide 36-37:

- `self.features`: khối trích xuất đặc trưng gồm Conv2d → BatchNorm2d → ReLU → MaxPool2d, lặp 2 lần (giống pipeline chuẩn đã học ở slide 31).
- `self.classifier`: khối phân loại gồm Flatten → Linear → ReLU → Dropout → Linear.
- `forward()`: chỉ cần gọi lần lượt `self.features(x)` rồi `self.classifier(x)` — toàn bộ logic forward/backward phức tạp (im2col, col2im) đã được PyTorch tự động xử lý ẩn bên dưới.

**Ba lợi thế nổi bật của PyTorch:**
1. **Dynamic computation graph:** đồ thị tính toán được xây dựng ngay trong lúc chạy forward (không cần khai báo trước như TensorFlow 1.x cũ) — cho phép logic điều khiển linh hoạt (if/else, vòng lặp) và debug dễ dàng bằng công cụ Python thông thường.
2. **BatchNorm ở chế độ production:** trong lúc huấn luyện dùng thống kê của mini-batch hiện tại; lúc đánh giá (evaluation) dùng thống kê tổng thể đã tích lũy (running statistics) — đảm bảo tính nhất quán khi triển khai thực tế.
3. **Automatic differentiation (autograd):** engine tự động theo dõi mọi phép toán trên tensor để dựng đồ thị lan truyền ngược, người dùng không cần viết code đạo hàm thủ công như ở slide 37.

**Ví dụ minh họa:** So với việc tự tay viết im2col/col2im ở 2 slide trước (giống tự lắp ráp một chiếc xe đạp từ từng con ốc vít), viết CNN bằng PyTorch `nn.Module` giống như mua một chiếc xe đạp lắp ráp sẵn theo module — chỉ cần chọn khung xe (features) và bộ phanh/yên xe (classifier), rồi lắp chúng nối tiếp nhau (forward). Autograd giống như một trợ lý ghi chép tự động mọi bước bạn đã làm khi lắp xe, để nếu cần tháo ra sửa lại (backward), trợ lý đó tự biết chính xác phải tháo theo thứ tự ngược lại thế nào mà bạn không cần tự nhớ.

---

### Slide 39 — Verification & Inference Dashboard

Slide tổng kết Phần IV, chứng minh tính đúng đắn (correctness) của cả 3 cách cài đặt bằng thực nghiệm cụ thể:

- **Sự hội tụ của bộ lọc học được (Learned Filter Convergence):** các kernel 3×3 sau khi huấn luyện (không có giám sát/nhãn cho việc này — mạng tự học), hội tụ về các bộ dò cạnh, tương phản màu, texture — khớp với lý thuyết đã trình bày ở slide 16 (hierarchical feature learning) và slide 7 (Hubel & Wiesel).
- **Đồng thuận dự đoán thời gian thực (Real-Time Prediction Consensus):** khi chạy cùng một mẫu dữ liệu qua cả 3 phiên bản model (NumPy, Keras, PyTorch), độ tin cậy dự đoán đạt trên 96% và **khớp nhau giữa cả 3 framework** — bằng chứng thực nghiệm cho luận điểm "cùng toán học, khác cách viết" xuyên suốt cả Phần IV.
- Đây là điểm chốt logic (closing argument) trước khi chuyển sang Phần V — nối lý thuyết với 3 bài thực hành thực tế của nhóm.

**Ví dụ minh họa:** Giống như 3 học sinh khác nhau được yêu cầu giải cùng một bài toán bằng 3 phương pháp khác nhau (giải tay, dùng máy tính bỏ túi, dùng Excel) — nếu cả 3 đều ra cùng một đáp số, đó là bằng chứng thuyết phục rằng phương pháp giải đúng, không phụ thuộc công cụ nào được dùng. Con số cụ thể: >96% độ tin cậy nghĩa là khi đưa 1 ảnh mẫu vào, cả 3 model đều tự tin ("chắc chắn hơn 96%") rằng đó là đúng lớp cần nhận diện — và quan trọng hơn, cả 3 model đều cùng chọn MỘT đáp án giống nhau.

---

## PHẦN V — TỪ LÝ THUYẾT ĐẾN THỰC HÀNH: 3 BÀI TẬP THỰC TẾ (Slide 40–45)

*Ghi chú chung:* phần này là "bằng chứng sống" — nhóm đã tự tay lặp lại toàn bộ lý thuyết ở Phần II-IV trên 3 bài toán thực tế khác nhau, dùng cả 3 cách cài đặt (Scratch/Keras/PyTorch), và đối chiếu kết quả.

### Slide 40 — From Theory to Practice

Giới thiệu 3 bài toán mà nhóm đã thực hành, cố tình chọn để bao phủ đủ các trường hợp đã học ở Phần II:

1. **Diabetes (MLP):** bài toán phân loại nhị phân trên **dữ liệu dạng bảng** (tabular — không có cấu trúc không gian 2D, đúng như slide 3 đã giải thích tại sao MLP phù hợp ở đây), 50 đặc trưng đã chuẩn hóa, model MLP với 2,177 tham số.
2. **Fashion-MNIST (CNN):** phân loại ảnh **xám 1 kênh** (grayscale), kích thước 1×28×28, 10 lớp, dùng CNN 2 khối với 421,642 tham số.
3. **CIFAR-10 (CNN):** phân loại ảnh **màu RGB 3 kênh**, kích thước 3×32×32, 10 lớp, dùng CNN 2 khối với 545,098 tham số.

Ba bài toán này cố tình bao phủ: dữ liệu bảng (không cần CNN) → ảnh 1 kênh → ảnh 3 kênh, để kiểm chứng toàn bộ lý thuyết đã trình bày trước đó trên các trường hợp thực tế khác nhau.

**Ví dụ minh họa:** Giống như một bộ 3 đề thi được thiết kế để kiểm tra 3 mức độ khác nhau của cùng một kiến thức — đề dễ (Diabetes, dữ liệu bảng không cần CNN, dùng để xác nhận MLP vẫn là lựa chọn đúng khi không có cấu trúc không gian), đề trung bình (Fashion-MNIST, ảnh xám đơn giản), đề khó (CIFAR-10, ảnh màu thực tế phức tạp hơn). Bộ dữ liệu Diabetes ví dụ có các đặc trưng như "chỉ số đường huyết", "tuổi", "BMI" — mỗi đặc trưng là một cột độc lập trong bảng, không có khái niệm "cột này nằm cạnh cột kia" như pixel trong ảnh.

---

### Slide 41 — Fashion-MNIST: Grayscale CNN in Practice

Truy vết (trace) shape của tensor qua từng lớp trong CNN thực tế cho bài Fashion-MNIST — áp dụng trực tiếp công thức tính kích thước đã học ở slide 10:

```
Input (B, 1, 28, 28)
  → Conv(1→32, k=3, pad=1) → (B, 32, 28, 28)   [same padding giữ nguyên kích thước]
  → MaxPool(2) → (B, 32, 14, 14)                [pooling giảm một nửa]
  → Conv(32→64, k=3, pad=1) → (B, 64, 14, 14)
  → MaxPool(2) → (B, 64, 7, 7)
  → Flatten → vector 3,136 chiều (= 64 × 7 × 7)
  → Linear(3136→128) → Linear(128→10)
```

Tổng số tham số: 421,642 — và điểm quan trọng nhất: con số này **khớp chính xác** giữa cả 3 cách cài đặt (NumPy Scratch, Keras, PyTorch) — minh chứng thực nghiệm cho công thức đếm tham số ở slide 13 (không phụ thuộc framework, chỉ phụ thuộc kiến trúc).

**Ví dụ minh họa:** Đây là ví dụ áp dụng "thực chiến" của công thức H_out = ⌊(H_in−K+2P)/S⌋+1 đã học ở slide 10: với H_in=28, K=3, P=1 (same padding), S=1: H_out = ⌊(28−3+2)/1⌋+1 = 28 — ảnh giữ nguyên 28×28 sau convolution. Sau đó MaxPool(2) với stride ngầm định 2: H_out = 28/2 = 14. Lặp lại tương tự cho khối thứ 2 để ra 7×7. Con số 3,136 = 64×7×7 chính là "kích thước hộp" chứa toàn bộ đặc trưng đã học được, trước khi "đổ" vào các lớp fully-connected để ra quyết định phân loại cuối cùng.

---

### Slide 42 — CIFAR-10: RGB CNN in Practice

Tương tự slide 41 nhưng cho ảnh màu RGB — minh họa sự khác biệt khi đầu vào có 3 kênh thay vì 1:

```
Input (B, 3, 32, 32)
  → Conv(3→32, k=3, pad=1) → (B, 32, 32, 32)
  → MaxPool(2) → (B, 32, 16, 16)
  → Conv(32→64, k=3, pad=1) → (B, 64, 16, 16)
  → MaxPool(2) → (B, 64, 8, 8)
  → Flatten → vector 4,096 chiều (= 64 × 8 × 8)
  → Linear(4096→128) → Linear(128→10)
```

So sánh với Fashion-MNIST: kiến trúc gần như giống hệt (2 khối conv-pool, cùng số kênh 32→64), chỉ khác ở **kênh đầu vào** (3 thay vì 1) và **kích thước ảnh gốc** (32×32 thay vì 28×28) — dẫn đến tổng tham số 545,098, nhiều hơn Fashion-MNIST một chút (chủ yếu do lớp Conv đầu tiên có 3 kênh input thay vì 1, và kích thước vector Flatten khác nhau).

**Ví dụ minh họa:** So sánh trực tiếp với slide 41 — áp dụng công thức đếm tham số ở slide 13 cho lớp Conv đầu tiên: Fashion-MNIST có Params = (3×3×1+1)×32 = 320 (vì C_in=1, ảnh xám); CIFAR-10 có Params = (3×3×3+1)×32 = 896 (vì C_in=3, ảnh màu) — chênh lệch 576 tham số chỉ từ một lớp Conv đầu tiên, đúng như dự đoán vì công thức phụ thuộc trực tiếp vào C_in. Đây là bằng chứng số học cụ thể cho việc "kênh đầu vào khác nhau dẫn đến tổng tham số khác nhau" đã nêu ở phần giải thích.

---

### Slide 43 — Same Math, Three Frameworks

Bảng đối chiếu tương tự slide 35 (Rosetta Stone), nhưng lần này áp dụng **trực tiếp lên code thực tế nhóm đã viết** cho 3 bài toán ở trên — không còn là ví dụ minh họa lý thuyết, mà là bằng chứng thực nghiệm cụ thể từ chính bài làm của nhóm:

| Thành phần | Scratch (NumPy) | Keras | PyTorch |
|---|---|---|---|
| Tiền xử lý | `/255.0`, chuyển sang CHW | `/255.0`, giữ nguyên HWC | `/255.0`, chuyển sang CHW |
| Lớp Convolution | `conv2d_forward` (im2col) | `layers.Conv2D()` | `nn.Conv2d()` |
| Lan truyền ngược | Hàm đạo hàm viết tay | Autodiff trong `fit()` | `loss.backward()` (Autograd) |
| Cập nhật trọng số | `W -= lr * dW` | `optimizer='sgd'` | `torch.optim.SGD()` |

**Ví dụ minh họa:** Giống như 3 người bạn cùng dịch một cuốn sách sang 3 ngôn ngữ khác nhau (code Scratch, Keras, PyTorch) nhưng đều dựa trên đúng một bản gốc (thuật toán CNN). Chi tiết thú vị đáng nhắc khi thuyết trình: `/255.0` xuất hiện ở cả 3 cột đầu tiên — đây là bước chuẩn hóa giá trị pixel (từ khoảng 0-255 về khoảng 0-1) để mạng học ổn định hơn, một bước tiền xử lý đơn giản nhưng bắt buộc ở cả 3 framework.

---

### Slide 44 — Empirical Results Across Frameworks

Kết quả thực nghiệm tổng hợp trên cả 3 bài toán, đối chiếu 3 framework:

- **Độ chính xác tương đồng (Accuracy Parity):** Diabetes 73-78%, Fashion-MNIST 86-90%, CIFAR-10 57-61% — chênh lệch giữa các framework chỉ trong khoảng vài phần trăm, chủ yếu do khác biệt ngẫu nhiên (random seed khởi tạo trọng số, thứ tự batch...), không phải do bản chất toán học khác nhau.
- **Số tham số khớp tuyệt đối:** 2,177 / 421,642 / 545,098 — khớp **chính xác tuyệt đối** giữa cả 3 cách cài đặt (không phải "gần giống" mà là giống hệt), vì kiến trúc và công thức đếm tham số là bất biến với framework.
- **Chênh lệch thời gian huấn luyện rất lớn:** bản Scratch (NumPy thuần) chậm hơn Keras/PyTorch **15-20 lần** trên các bài CNN — vì NumPy không có tối ưu phần cứng chuyên biệt (BLAS đã tối ưu, cuDNN trên GPU...) như các framework production.

**Ý nghĩa:** đây chính là minh chứng thực nghiệm hoàn chỉnh cho luận điểm cốt lõi xuyên suốt cả bài — "cùng toán học, khác tốc độ thực thi tùy mức trừu tượng hóa" (đã nêu ở slide 34).

**Ví dụ minh họa:** Hình dung việc "chậm hơn 15-20 lần" bằng thời gian thực tế: nếu bản Keras/PyTorch huấn luyện xong một model trong 5 phút, bản NumPy thuần làm việc tương tự sẽ mất từ 75 đến 100 phút — đủ để đi ăn một bữa cơm trưa trọn vẹn và quay lại vẫn chưa xong. Đây là lý do trong thực tế công nghiệp gần như không ai dùng NumPy thuần để huấn luyện model quy mô lớn, dù về mặt kết quả toán học nó hoàn toàn chính xác như các framework khác.

---

### Slide 45 — Closing Synthesis

Ba kết luận (key finding) tổng hợp toàn bộ bài trình bày:

1. **Mathematical Equivalence (Tương đương toán học):** khi giữ nguyên kiến trúc, tập dữ liệu, và cách chia dữ liệu, cả 3 framework hội tụ về độ chính xác gần như nhau — vì các phép tính tensor và công thức gradient là bất biến, không phụ thuộc cú pháp framework.

2. **Computational Efficiency (Hiệu quả tính toán):** các framework đã biên dịch (compiled) như Keras/PyTorch vượt trội NumPy thuần tới 15-20 lần, vì tận dụng thư viện BLAS/CUDA đã tối ưu sâu ở tầng C++/GPU — phần lớn khối lượng tính toán (>90% FLOPs, như đã thấy ở slide 33) nằm ở convolution, nơi các tối ưu này phát huy tác dụng lớn nhất.

3. **Abstraction Spectrum (Phổ trừu tượng hóa):** ba framework phục vụ ba mục đích khác nhau — NumPy cho mục đích **giáo dục** (hiểu cơ chế), Keras cho **tốc độ phát triển** (prototype nhanh), PyTorch cho **nghiên cứu** (cân bằng kiểm soát và hiệu năng). Không có framework nào "tốt nhất tuyệt đối" — lựa chọn phụ thuộc vào mục tiêu.

**Ví dụ minh họa:** Ba kết luận này giống như ba bài học rút ra từ việc học lái 3 loại xe khác nhau để đi cùng một quãng đường (mathematical equivalence — đều đến đích): xe đạp tự đạp (NumPy) giúp hiểu rõ cơ chế bánh xe, xích, phanh hoạt động thế nào nhưng đi chậm; xe máy (Keras) đi nhanh và tiện nhưng không thấy động cơ bên trong hoạt động ra sao; ô tô số sàn (PyTorch) vừa đi nhanh vừa cho phép người lái can thiệp linh hoạt (chuyển số, phanh tay) khi cần. Không có phương tiện nào "tốt nhất tuyệt đối" cho mọi tình huống — chọn xe đạp khi cần học, chọn xe máy khi cần tốc độ tối đa với ít điều khiển, chọn ô tô số sàn khi vừa cần tốc độ vừa cần kiểm soát chi tiết.
