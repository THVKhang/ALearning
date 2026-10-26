# Phân công cố định — Nhóm CR7

Môn CO3133 Deep Learning and Its Applications, Semester-261.
Tài liệu này chốt **role cố định** của ba thành viên và liệt kê **công việc bắt buộc của A1, A2, A3**
theo handbook (`docs/handbook-ene-v2.pdf`). Mọi mục đều ghi kèm số section của handbook để khi cần
tra thì tra thẳng vào đó, không phải nhớ.

---

## 0. Ba role cố định

Đây đúng là ba role đã công bố trên landing page (yêu cầu của §2.1: "main role or contribution of
each member"). Giữ nguyên, không đổi giữa các assignment — role đi theo người, không đi theo bài.

| Thành viên | MSSV | Role cố định | Sở hữu xuyên suốt A1–A3 |
|---|---|---|---|
| Trần Hoàng Vỹ Khang | 2352502 | **Data & Reproducibility (Lead)** | Dataset, EDA, split/leakage, Dataset+DataLoader, preprocessing, seed/config/dependency/hardware, `README.md`, kỷ luật repo (branch, PR, `.gitignore`), checkpoint & hướng dẫn tái tạo, chốt hạn nộp LMS |
| Đỗ Đăng Khoa | 2352554 | **Experiments & Analysis, Delivery** | Chạy thí nghiệm, bảng metric, learning curve, confusion matrix / ví dụ đúng-sai, error analysis, compute-cost, trang assignment (GitHub Pages), slide, dựng video YouTube |
| Dương Đăng Khoa | 2352551 | **Modeling & Methods** | Code model, vòng train/eval của từng model, phần Methodology của báo cáo, sơ đồ pipeline, thiết kế ablation / controlled experiment |

**Ba luật chung, áp cho cả ba assignment**

1. **Video:** mọi thành viên phải trình bày một phần (§2.3). Không ai được vắng mặt trong video.
   Tiêu đề thống nhất: `CO3133-Semester-261 – Group [ID] – Assignment [1/2/3]`, Public hoặc Unlisted.
2. **AI disclosure:** phần nào do AI sinh ra thì **người sở hữu role đó** phải đọc lại từng dòng, tự
   chạy lại, rồi mới ghi "đã review" vào `AI_USAGE.md` (§5.5 — sinh viên chịu trách nhiệm cuối cùng).
   Chưa review thì entry phải ghi rõ là *outstanding*, không được ghi là đã duyệt.
3. **Merge:** không ai merge nhánh của chính mình vào `main`. Mở PR, một người khác đọc rồi mới merge.

---

## 1. Việc giống nhau ở cả ba assignment

| # | Deliverable (§4.1) | Người chịu trách nhiệm |
|---|---|---|
| 1 | Trang assignment trên GitHub Pages, đủ các mục của §2.2 | Đỗ Khoa |
| 2 | Source-code repository | Khang |
| 3 | Báo cáo theo §3 (Part 1 dữ liệu / Part 2 phương pháp / Part 3 kết quả) | Part 1 Khang · Part 2 Dương Khoa · Part 3 Đỗ Khoa |
| 4 | Slide thuyết trình | Đỗ Khoa (nội dung method: Dương Khoa) |
| 5 | Video YouTube | Đỗ Khoa dựng, **cả ba cùng nói** |
| 6 | AI Usage Disclosure ở **4 nơi**: landing page + trang assignment + báo cáo + `AI_USAGE.md` | Mỗi người tự ghi phần của mình; Khang gộp |
| 7 | Checkpoint hoặc hướng dẫn tái tạo checkpoint | Khang |
| 8 | PDF báo cáo nộp LMS: `CR7_A1_Report.pdf`, `CR7_A2_Report.pdf`, `CR7_A3_Report.pdf` — do **người đại diện** nộp | Khang nộp |

§2.2 yêu cầu trang assignment có đủ: tiêu đề · thành viên · giảng viên · problem statement · dataset +
EDA · methodology · experimental setup · results · comparison & discussion · error analysis ·
limitations & conclusion · link source code · link checkpoint · link report/slides · link video · AI
disclosure riêng của bài đó.

§4.2 yêu cầu repo phải có: `README.md` (install / chuẩn bị dữ liệu / train / evaluate), config file,
seed, phiên bản dependency, thông tin hardware, checkpoint, link báo cáo + trang assignment,
`AI_USAGE.md`. Và **mọi số liệu báo cáo phải truy được** về config + split + checkpoint + log/experiment
ID + commit tương ứng. Nộp notebook đã chạy sẵn mà không có hướng dẫn tái tạo là **không hợp lệ**.

Trễ hạn: trừ **20% điểm milestone đó mỗi tuần** (kể cả lẻ tuần), quá 7 ngày là **0** điểm milestone (§7.2).

---

## 2. Assignment 1 — Image Classification (40% điểm đồ án)

| Milestone | Hạn 23:59 GMT+7 | Trọng số trong A1 | Nội dung tối thiểu (§7.4) |
|---|---|---|---|
| M1 Draft | 23 Sep 2026 | 25% | EDA · Dataset/DataLoader · vòng train/val · Linear + MLP chạy được (CNN tuỳ chọn) |
| M2 Final | **21 Oct 2026** | 75% | **Đủ 5 model** · so sánh đầy đủ · báo cáo · slide · video · trang assignment · checkpoint |

### 2.1 Năm model bắt buộc (§11.1) — chủ: Dương Khoa

1. **Linear / softmax**: flatten ảnh, một linear layer ra logits, cross-entropy.
   Lưu ý: **không đặt softmax trước `CrossEntropyLoss`**.
2. **MLP**: ít nhất một hidden layer; phải *giải thích* activation và regularization đã chọn.
3. **CNN tự thiết kế**: **không được chỉ dùng model pretrained**; phải giải thích convolution,
   pooling, feature map.
4. **LSTM hoặc GRU**: coi ảnh là chuỗi (hàng / cột / patch); phải nói rõ timestep là gì, input size
   là gì, hidden representation là gì.
5. **Transformer**: token embedding/projection + positional encoding; giải thích attention nhận gì,
   trả ra gì.

§11.3: PyTorch là framework chính; phải là **project tái lập được**, không phải một notebook chạy một
lần; backbone pretrained chỉ được dùng như extension (§11.2), không thay thế model bắt buộc.

### 2.2 Kết quả so sánh bắt buộc (§12.1) — chủ: Đỗ Khoa

accuracy · macro-F1 · số parameter · thời gian train · thời gian inference · learning curve train/val ·
confusion matrix · ví dụ đúng **và** ví dụ sai · phân tích inductive bias của từng họ model.
Handbook ghi thẳng: **không được kết luận model nào tốt hơn chỉ dựa vào accuracy**.

§12.2 — công bằng: cùng split, cùng protocol, cùng metric; phải báo cáo hardware, phiên bản software,
seed, và quy tắc chọn checkpoint. (Chủ: Khang.)

### 2.3 EDA bắt buộc (§13) — chủ: Khang

Phân bố lớp · kích thước input · phân tích mất cân bằng · ảnh mẫu đại diện.

### 2.4 Rubric A1 (§14) — ai mất điểm ở đâu

| Thành phần | Trọng số | Role chịu |
|---|---|---|
| Problem & data understanding | 15% | Khang |
| Data & training pipeline | 15% | Khang |
| Methods & implementation | 30% | Dương Khoa |
| Experimental design & results | 20% | Đỗ Khoa |
| Analysis & error analysis | 10% | Đỗ Khoa |
| Reproducibility, report & presentation | 10% | cả ba |

### 2.5 Còn thiếu gì cho A1 M2 Final

- [x] Merge nhánh `a1-m2-models` vào `main` (6 model đã chạy deterministic, README + trang A1 đã cập nhật)
- [x] `assignment1/scripts/compare.py` — đã hoàn thiện, sinh đủ 4 bảng so sánh §12.1 từ `outputs/*/metrics_test.json` — **Đỗ Khoa**
- [ ] Báo cáo `CR7_A1_Report.pdf` theo §3, nộp LMS — **cả ba, Khang nộp**
- [ ] Slide — **Đỗ Khoa**
- [ ] Video YouTube, cả ba người nói — **Đỗ Khoa dựng**
- [x] Checkpoint hoặc hướng dẫn tái tạo (§4.1 mục 7) — **Khang**
- [x] `assignment1/AI_USAGE.md` — toàn bộ 3 Entry đã thống nhất vai trò review code + kiểm soát architecture và ghi rõ đã kiểm tra (§5.5)

---

## 3. Assignment 2 — Monocular Depth Estimation (30% điểm đồ án)

| Milestone | Hạn 23:59 GMT+7 | Trọng số trong A2 | Nội dung tối thiểu (§7.5) |
|---|---|---|---|
| M1 Proposal | 07 Oct 2026 | 15% | Dataset Proposal: task, data, split, metric, baseline, compute estimate. **Chỉ được chạy thí nghiệm chính sau khi được Approved / Approved with conditions** |
| M2 Draft | **28 Oct 2026** | 25% | Baseline đơn giản + model pretrained đang chạy · EDA · metric sơ bộ |
| M3 Final | 11 Nov 2026 | 60% | Controlled experiment · error analysis · báo cáo · slide · video · trang assignment · checkpoint |

### 3.1 Dataset Proposal — 11 trường bắt buộc (§18) — chủ: Khang

Tên dataset, nguồn, version, license · định nghĩa task/input/output · số sample + loại annotation ·
phân tích phân bố sơ bộ · kế hoạch split · **đơn vị split chống leakage** · metric · kế hoạch baseline ·
model pretrained dự kiến · ước lượng compute · quy tắc chọn subset (nếu có).

§19.6 (track depth): **≥ ~5,000 cặp RGB–depth**; phải mô tả depth scale, invalid pixel, và cách chia
scene; **không để cùng một scene nằm ở cả train và test**.

### 3.2 Yêu cầu kỹ thuật (§20)

| # | Yêu cầu | Role chịu |
|---|---|---|
| 1 | Một baseline đơn giản (U-Net tự viết) | Dương Khoa |
| 2 | ≥ 1 model pretrained/modern (MiDaS) | Dương Khoa |
| 3 | Quy trình fine-tune rõ ràng | Dương Khoa |
| 4 | ≥ 1 controlled experiment: freeze vs full fine-tune, có/không augmentation, backbone A vs B, loss A vs B, resolution A vs B… | Dương Khoa thiết kế · Đỗ Khoa chạy |
| 5 | Đánh giá định lượng | Đỗ Khoa |
| 6 | Đánh giá định tính | Đỗ Khoa |
| 7 | Error analysis | Đỗ Khoa |
| 8 | Compute-cost analysis | Đỗ Khoa |

Metric cho depth (§20.1): **Abs Rel, RMSE, δ accuracy**.

§21 — mỗi thí nghiệm báo cáo phải nêu: **hypothesis · yếu tố thay đổi · yếu tố giữ cố định · metric ·
tiêu chí quyết định**. Ít nhất một ablation là **bắt buộc**. Kèm hardware, phiên bản software, seed,
quy tắc chọn checkpoint, thời gian train, chi phí inference.

§22 — EDA cho depth: **phân bố độ sâu · invalid pixel · phân bố scene**.

### 3.3 Rubric A2 (§23)

Problem & data 15% (Khang) · pipeline 15% (Khang) · Methods & implementation 25% (Dương Khoa) ·
Experimental design & results 20% (Đỗ Khoa) · Analysis & error analysis 15% (Đỗ Khoa) ·
Reproducibility/report/presentation 10% (cả ba).

### 3.4 Việc tiếp theo của A2 (trước 28 Oct)

- [x] Tải lại `.ipynb` bản cuối từ Kaggle ghi đè `assignment2/notebooks/` — **Khang**
- [x] `assignment2/AI_USAGE.md` — cả 13 Entry đã thống nhất vai trò review code + kiểm soát architecture và ghi rõ đã kiểm tra theo §5.5
- [x] U-Net baseline chạy được + metric sơ bộ Abs Rel / RMSE / δ — **Dương Khoa**
- [x] MiDaS zero-shot + full fine-tune + frozen encoder trên subset nearest-neighbour — **Đỗ Khoa & Dương Khoa**
- [x] Controlled experiment (`freeze_encoder: false` vs `true`) đã có hypothesis + decision criterion và kết quả 1 seed cho M2 Draft — **Dương Khoa & Đỗ Khoa**

---

## 4. Assignment 3 — Multimodal Deep Learning (30% điểm đồ án)

| Milestone | Hạn 23:59 GMT+7 | Trọng số trong A3 | Nội dung tối thiểu (§7.6) |
|---|---|---|---|
| M1 Proposal | 18 Nov 2026 | 15% | Dataset Proposal: modality, pair validity, đơn vị split, metric, baseline. Chỉ chạy thí nghiệm chính sau khi được approve |
| M2 Draft | 25 Nov 2026 | 25% | Unimodal A, unimodal B, simple fusion — **có số liệu định lượng** |
| M3 Final | 02 Dec 2026 | 60% | Fusion cải tiến · ablation · báo cáo · slide · video · trang assignment · checkpoint |

Đề bài: hệ thống dùng **ít nhất hai modality thật sự liên quan nhau** (cùng entity, event, scene,
hoặc timestamp).

### 4.1 Tính hợp lệ của cặp dữ liệu (§28.1) — chủ: Khang

- Mỗi sample phải có ≥ 2 modality gắn với nhau bởi cùng entity/event/scene/timestamp.
- **Không được ghép ngẫu nhiên** dữ liệu từ hai dataset độc lập.
- Phải kiểm tra cặp bị thiếu, bị nhiễu, bị lệch.
- Split theo entity/event/scene; **một entity có nhiều record thì tất cả phải ở cùng một split**.

Ngưỡng kích thước (§28.2): multimodal classification ≥ 5 lớp và ~5,000 cặp · image–text retrieval
~5,000 cặp · captioning ~5,000 ảnh · VQA ~10,000 cặp câu hỏi–ảnh · **RGB–depth ~5,000 cặp aligned**.

### 4.2 Bốn model bắt buộc (§29.1) — chủ: Dương Khoa

1. Baseline chỉ dùng **modality A**.
2. Baseline chỉ dùng **modality B**.
3. Một **simple fusion** baseline.
4. Một **multimodal / fusion cải tiến**.

Multimodal **không bắt buộc phải thắng** unimodal. Nếu không thắng thì phải phân tích nguyên nhân
bằng tính chất dữ liệu và bằng chứng thí nghiệm.

### 4.3 Nội dung phương pháp bắt buộc (§29.2) — chủ: Dương Khoa

Encoder của từng modality · cách biểu diễn từng modality · thủ tục align dữ liệu · tầng fusion (early /
intermediate / late) · phép fusion · loss từng modality và loss tổng · quy trình train · xử lý khi thiếu
modality (nếu có) · chiến lược pretrain + fine-tune · **≥ 1 ablation bỏ hoặc đổi một modality / một
thành phần fusion**.

### 4.4 Nội dung kết quả bắt buộc (§30) — chủ: Đỗ Khoa

So sánh unimodal vs multimodal · đánh giá định lượng · ví dụ định tính · **trường hợp hai modality bổ
trợ nhau** · **trường hợp hai modality xung đột nhau** · error analysis · ablation · compute-cost.
Mỗi thí nghiệm nêu hypothesis, yếu tố thay đổi, yếu tố cố định, metric, tiêu chí kết luận.

§31 — EDA multimodal phải có: số cặp · tỷ lệ thiếu modality · độ dài text (nếu có) · chất lượng cặp.
Sơ đồ pipeline gợi ý: `raw paired data → preprocessing → data loader → encoders → fusion → loss →
optimization → prediction → evaluation`.

§29.3 — metric: classification accuracy + macro-F1 · retrieval R@1/R@5/R@10 + median rank ·
captioning CIDEr/BLEU + định tính · VQA accuracy · detection/segmentation mAP hoặc mIoU ·
RGB–depth dùng metric depth tương ứng.

### 4.5 Rubric A3 (§32)

Problem & data (pair validity, EDA, leakage-aware split) 15% (Khang) · pipeline multimodal 15%
(Khang) · Methods & implementation 25% (Dương Khoa) · Experimental design & results 20% (Đỗ Khoa) ·
Analysis (support/conflict, ablation, limitations) 15% (Đỗ Khoa) · Reproducibility/report/presentation
10% (cả ba).

### 4.6 Trạng thái hoàn thành A3

- [x] Chọn task + hai modality (6-Class Multimodal Environment & Viewing-Geometry Classification trên 7,195 cặp RGB + 3D LiDAR Depth của DIODE, đạt ngưỡng §28.2: 6 lớp ≥ 5 lớp, 7,195 cặp ≥ 5,000 cặp, 0% scene overlap) — **cả ba**
- [x] Điền cột Role trong `assignment3/index.html` theo bảng ở mục 0 — **Đỗ Khoa**
- [x] Cài đặt đủ 4 mô hình bắt buộc (`rgb_only`, `depth_only`, `simple_fusion`, `improved_fusion`) + 2 ablation (`ablation_no_gate`, `ablation_no_aux`) + đánh giá missing-modality — **Dương Khoa & Đỗ Khoa**
- [x] Phân tích định lượng, định tính, trường hợp bổ trợ / xung đột giữa hai modality, error analysis, compute cost, trang web `assignment3/index.html`, `README.md`, `AI_USAGE.md`, và `CR7_A3_Report.md` — **cả ba**

---

## 5. Contributor ghi trên web: một bảng duy nhất cho cả ba assignment

Đã chốt: **cả bốn trang web đều ghi contributor theo đúng ba role ở mục 0**, không trang nào ghi
khác. Role là *phần việc mình chịu trách nhiệm*, không phải bảng kê khối lượng của một milestone
riêng lẻ. §2.1 chỉ đòi "main role or contribution of each member", nên ghi theo role là hợp chuẩn.

| Trang | Trạng thái |
|---|---|
| `index.html` (landing) | đã có đủ ba role — giữ nguyên |
| `assignment1/index.html` | đã thêm cột Role (trước đó chỉ có tên + MSSV); nhân đây sửa luôn lỗi chính tả "Đàng" → "Đăng" ở hai cái tên |
| `assignment2/index.html` | dòng "No M1 contribution" của Dương Đăng Khoa **đã bỏ**, thay bằng role Modeling & Methods (U-Net baseline, MiDaS fine-tune, methodology, thiết kế controlled experiment + ablation) |
| `assignment3/index.html` | cột Role trước đây để trống, nay đã điền đủ ba người |

Nội dung role trên mỗi trang được viết theo đúng việc của bài đó, nhưng **nhãn role thì giống nhau
tuyệt đối**: Data & Reproducibility (Lead) · Experiments & Analysis, Delivery · Modeling & Methods.
Sửa nhãn ở một trang thì phải sửa cả bốn trang và cả file này.

**Ranh giới cần giữ.** Bảng role nói *ai chịu phần nào*. Riêng `AI_USAGE.md` thì vẫn ghi *ai thực sự
đã làm từng mục đã log*, vì §5.5 buộc từng thành viên tự chịu trách nhiệm cho phần AI hỗ trợ mà
mình đứng tên. Hai thứ đó không được trộn: không được sửa `AI_USAGE.md` cho khớp bảng role.
Ghi chú attribution ở `assignment2/index.html` đã nói rõ điều này.

Căn cứ đóng góp A1 có thật trong git, để dùng khi cần trả lời giảng viên:

| Tác giả git | Số commit |
|---|---|
| GabrielHCMUT + Gabriel Tr (Khang) | 22 |
| Đỗ Khoa | 12 |
| Khoa4444 (Dương Khoa) | 2 — nhánh `mlp-cnn`: `src/models/mlp.py`, `src/models/cnn.py`, `train.py`, `default.yaml` |

Hai commit trên nhánh `mlp-cnn` là đóng góp A1 thật của Dương Khoa, nay đã được ghi công ở cột Role
của `assignment1/index.html`. Nhánh đó không được merge vì CNN trong đó có 421,642 parameter so với
50,378 của bản đã dùng, nên không dùng được cho kết luận so sánh — nhưng việc không merge không
đồng nghĩa với không có đóng góp.
