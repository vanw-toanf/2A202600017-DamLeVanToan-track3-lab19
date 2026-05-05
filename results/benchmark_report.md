# Báo cáo Benchmark GraphRAG vs Flat RAG (Lab 19)

## 1. Phân tích chi phí xây dựng đồ thị (Graph Construction)

- **Tổng số bài viết xử lý:** 60
- **Tổng số bộ ba (triples) trích xuất:** 1090
- **Số lượng Node trong Neo4j:** 1099
- **Số lượng Relationship trong Neo4j:** 1081
- **Tổng số Token LLM sử dụng (GPT-4o-mini):** 102,421
- **Chi phí ước tính:** $0.0154

## 2. Bảng so sánh kết quả 20 câu hỏi Benchmark

Dưới đây là thống kê tổng quan so sánh hiệu năng giữa hai hệ thống:

| Tiêu chí | Flat RAG | GraphRAG |
|----------|----------|----------|
| Số câu thắng (LLM Judge) | **13** | **4** |
| Số câu hòa | \multicolumn{2}{c|}{3} |
| Độ trễ trung bình (s) | 2.42 | **2.47** |
| Tổng Token sử dụng | 56,953 | **5,157** |
| Điểm đánh giá trung bình (/30) | 23.2 | 16.1 |
| Ước tính chi phí truy vấn | $0.0085 | **$0.0008** |

> **🏆 Hệ thống chiến thắng chung cuộc:** Flat RAG

## 3. Kết luận ngắn gọn

- **Về độ chính xác:** Flat RAG thường trả lời đầy đủ và bao quát hơn vì truy xuất được nguyên văn đoạn text thô, hữu ích nếu thực thể chưa được trích xuất hoàn thiện vào GraphRAG. GraphRAG lại rất mạnh ở các câu hỏi yêu cầu liên kết phức tạp.
- **Về chi phí và tốc độ:** GraphRAG tốn chi phí ban đầu để xây dựng đồ thị, nhưng ở pha truy vấn (Querying), GraphRAG **nhanh hơn đáng kể** và **tiết kiệm token rất nhiều** so với Flat RAG do chỉ truyền các quan hệ đồ thị thu gọn thay vì nguyên văn nhiều đoạn văn bản lớn.
