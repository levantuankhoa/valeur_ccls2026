# Hướng dẫn cộng tác — Valeur (CCLS 2026)

> Tài liệu này dành cho **3 đồng tác giả** của poster Valeur tại CCLS 2026:
> Võ Thị Phương Linh, Nguyễn Thị Ngọc Trinh, Trương Nguyễn Cát Ly.
>
> **Mục đích:** giúp các bạn đóng góp vào repo theo cách có **dấu vết rõ ràng trên GitHub** (tên hiện trong tab Contributors, tên ghi cạnh từng commit). Việc này quan trọng vì:
>
> 1. Khớp với BibTeX của poster — nơi cả 4 người được liệt kê là tác giả → cần evidence rằng 4 người đều có đóng góp thật.
> 2. Khớp với mệnh đề đạo đức "Methodology, analysis, and writing are the team's own" — reviewer hoặc người đọc sau này có thể click vào "Insights → Contributors" trên GitHub và thấy tên cả 4 người.
> 3. Tạo trail audit đơn giản cho ban tổ chức CCLS hoặc bất kỳ ai muốn xác minh đa-tác-giả.
>
> **Quy mô đóng góp tối thiểu mỗi người:** 1 commit có ý nghĩa. Có thể nhỏ — chỉ cần là đóng góp thật, không phải sửa một dấu phẩy.

---

## 1 · Setup ban đầu (làm 1 lần duy nhất)

### 1.1 Tạo tài khoản GitHub

Nếu chưa có, vào https://github.com/signup. **Quan trọng:**

- **Tên hiển thị (Display name):** ghi đúng tên đầy đủ tiếng Việt có dấu (ví dụ: `Võ Thị Phương Linh`) → tên này sẽ xuất hiện trên các commit của bạn.
- **Email:** dùng email cá nhân thật (Gmail là OK). **Phải nhớ email này** — bước sau cần dùng đúng email này để Git biết commit là của bạn.

### 1.2 Verify email trên GitHub

Vào **Settings → Emails** trên GitHub, thêm email và verify (GitHub gửi link xác nhận). Đây là email mà Git sẽ dùng để link commit của bạn với account.

> ⚠ Nếu commit gửi từ email **không trùng** với email đã verify trên GitHub, commit sẽ hiện tên nhưng **không link được vào profile** → tab Contributors có thể không đếm bạn.

---

## 2 · Cách đóng góp — chọn 1 trong 2 path

### Path A — Sửa trực tiếp trên web GitHub (KHÔNG CẦN cài Git)

**Phù hợp nếu:** bạn chưa từng dùng Git, chỉ muốn edit 1 file Markdown.

1. Vào https://github.com/levantuankhoa/valeur_ccls2026
2. Click vào file muốn sửa (ví dụ `README.md`, hoặc tạo file mới trong `docs/`)
3. Click icon ✏ (Edit this file) ở góc phải trên
4. Sửa nội dung trực tiếp trong trình duyệt
5. Cuộn xuống cuối trang → khung **"Commit changes"**:
   - **Message:** viết ngắn gọn theo format `docs: tên-bạn — mô tả`
     Ví dụ: `docs: Linh — viết phần introduction tiếng Việt`
   - **Description:** để trống cũng OK
   - Chọn **"Commit directly to the main branch"**
6. Click **"Commit changes"**

→ Xong. Commit của bạn xuất hiện trong lịch sử. Tên bạn link với GitHub profile.

### Path B — Clone repo về máy + dùng GitHub Desktop (DỄ HƠN dùng terminal)

**Phù hợp nếu:** bạn muốn upload nhiều file cùng lúc (ví dụ poster PDF, hình ảnh, dataset notes).

1. Tải GitHub Desktop: https://desktop.github.com/ (Windows/Mac)
2. Mở app → **File → Clone repository → URL** → paste:
   ```
   https://github.com/levantuankhoa/valeur_ccls2026
   ```
3. Chọn thư mục lưu trên máy (ví dụ `Desktop/valeur_ccls2026`)
4. Sửa/thêm file trong thư mục đó bằng bất kỳ editor nào (VS Code, Notepad++, Word export ra PDF...)
5. Quay lại GitHub Desktop → bạn sẽ thấy danh sách thay đổi
6. Khung **"Summary"** ở góc trái dưới: viết message như `docs: Trinh — upload poster draft v1`
7. Click **"Commit to main"**
8. Click **"Push origin"** ở thanh trên → đẩy lên GitHub

→ Xong. Refresh repo trên web sẽ thấy commit + file mới.

---

## 3 · Mỗi người đóng góp gì (gợi ý cụ thể)

Dưới đây là 6 hướng đóng góp **có ý nghĩa thật**, mỗi người có thể chọn 1–2 việc. **Không cần làm hết** — mỗi người 1 commit nhỏ là đủ để xuất hiện trên Contributors.

| # | Việc | Người gợi ý | File đề xuất | Path |
|---|---|---|---|---|
| 1 | Viết phần **Introduction tiếng Việt** giới thiệu Kafka + bối cảnh nghiên cứu | Linh hoặc Trinh | `docs/intro_vietnamese.md` (file mới) | A hoặc B |
| 2 | Viết phần **Theoretical foundation tiếng Việt** giải thích PAD model + Gilbert-Allan entrapment cho người đọc tiếng Việt | Trinh hoặc Ly | `docs/theory_vietnamese.md` (file mới) | A hoặc B |
| 3 | **Upload poster draft (PDF/PNG)** khi xong | Khoa hoặc người làm poster chính | `poster/CCLS2026_poster_v1.pdf` (tạo thư mục mới) | B (file binary phải qua GitHub Desktop) |
| 4 | **Translate README sang tiếng Việt** thành `README.vi.md` | Linh | `README.vi.md` | A hoặc B |
| 5 | **Viết section "Discussion" tiếng Việt** dựa trên brief + kết quả phân tích | Ly hoặc Trinh | `docs/discussion_vietnamese.md` (file mới) | A hoặc B |
| 6 | **Curate references list** — viết file `REFERENCES.md` đầy đủ với BibTeX hoặc APA của Mehrabian-Russell 1974, Gilbert-Allan 1998, Warriner 2013, Mohammad 2025, Kafka 1915 | Bất kỳ ai | `REFERENCES.md` | A hoặc B |

**Quy ước commit message:** luôn bắt đầu bằng `docs:` rồi tên + mô tả.

Ví dụ:
- ✅ `docs: Linh — viết introduction tiếng Việt cho Kafka context`
- ✅ `docs: Trinh — bổ sung giải thích PAD model`
- ✅ `docs: Ly — section discussion về Part 3 entrapment asymmetry`
- ❌ `Update README` (không có tên, không rõ làm gì)
- ❌ `fix typo` (quá nhỏ, không phải đóng góp ý nghĩa)

---

## 4 · Verify đóng góp đã hiện trên Contributors

Sau khi commit:

1. Vào https://github.com/levantuankhoa/valeur_ccls2026
2. Click tab **"Insights"** (thanh ngang trên cùng)
3. Chọn **"Contributors"** ở sidebar trái
4. → Bạn sẽ thấy danh sách avatars + tên những người đã commit. Tên bạn nên xuất hiện ở đó.

> Nếu **không thấy tên mình** dù đã commit:
> - Email trong commit có thể không trùng email verify trên GitHub → vào **Settings → Emails** kiểm tra
> - Có thể mất vài phút để GitHub cập nhật Insights → đợi 5–10 phút rồi refresh

---

## 5 · Quy tắc an toàn

1. **Đừng push file dữ liệu lớn** (`*.csv` lexicons, `*.txt` corpus). `.gitignore` đã chặn sẵn nhưng cẩn thận khi dùng GitHub Desktop — nếu thấy `data/warriner_2013.csv` trong danh sách thay đổi thì **bỏ chọn** trước khi commit.

2. **Đừng push file `.env`, password, API key** — không có lý do gì để commit mấy thứ đó.

3. **Đừng force-push hoặc rebase** nếu không chắc đang làm gì. Để Khoa làm những thao tác lịch sử (lead author).

4. **Nếu lỡ commit gì đó nhạy cảm:** đừng tự xóa, hỏi Khoa trước. Force-push để xóa lịch sử là thao tác phá hủy, cần làm cẩn thận.

5. **Conflict khi push:** nếu GitHub Desktop báo *"This branch is behind origin/main"* → click **"Pull origin"** trước rồi push lại. Không được click **"Force push"**.

---

## 6 · Checklist trước CCLS 2026

Để claim đa-tác-giả "đứng vững" về mặt ethical evidence:

- [ ] Tất cả 4 người (Khoa + Linh + Trinh + Ly) có ít nhất 1 commit trong tab Contributors
- [ ] Mỗi commit có tên rõ ràng trong message (`docs: Linh —`, `docs: Trinh —`, ...)
- [ ] Repo có thư mục `docs/` chứa ít nhất 2 file Vietnamese-language markdown của co-authors
- [ ] Poster final upload vào `poster/` (có/không tuỳ template, nhưng có thì tốt)
- [ ] README đã có Acknowledgments + Authors section (đã có sẵn — không cần làm lại)

---

## 7 · Liên hệ

Nếu kẹt ở bước nào:
- Hỏi Khoa trên Zalo/Messenger team
- Hoặc tạo **Issue** trên GitHub: vào tab **Issues → New issue** → mô tả vấn đề. Có dấu vết hơn chat.

**Đường dẫn nhanh:**
- Repo: https://github.com/levantuankhoa/valeur_ccls2026
- Insights → Contributors: https://github.com/levantuankhoa/valeur_ccls2026/graphs/contributors
- Issue tracker: https://github.com/levantuankhoa/valeur_ccls2026/issues

---

*Tài liệu này viết bằng tiếng Việt vì target reader là team — reviewer hội nghị nếu mở ra cũng sẽ hiểu đây là internal team guide, không gây hiểu nhầm. Cập nhật lần cuối: 2026-05-01.*
