# Lab 19: GraphRAG với Tech Company Corpus

Dự án này là bài thực hành tổng hợp về việc xây dựng hệ thống **GraphRAG** (Retrieval-Augmented Generation dựa trên Đồ thị Tri thức) và so sánh hiệu năng của nó với **Flat RAG** truyền thống (Vector RAG). 

## 🧠 Các Kiến Thức Cốt Lõi Trong Lab

### 1. Flat RAG (Vector RAG) vs. GraphRAG
- **Flat RAG**: Sử dụng vector embeddings (ChromaDB, FAISS) để tìm kiếm độ tương đồng về ngữ nghĩa.
  - *Ưu điểm:* Dễ cài đặt, giữ lại được nguyên văn ngữ cảnh thô (raw context), trả lời tốt các câu hỏi trực tiếp.
  - *Nhược điểm:* Khó khăn khi trả lời các câu hỏi đòi hỏi "Multi-hop reasoning" (suy luận nhiều bước) hoặc cần tổng hợp thông tin rời rạc từ nhiều nguồn khác nhau.
- **GraphRAG**: Sử dụng Đồ thị Tri thức (Knowledge Graph - Neo4j) để lưu trữ thông tin dưới dạng các bộ ba (Subject - Relation - Object).
  - *Ưu điểm:* Thể hiện rõ ràng các mối quan hệ phức tạp, giúp LLM nhìn thấy bức tranh tổng thể về các thực thể được liên kết với nhau. Giảm thiểu chi phí token vì context truyền vào LLM rất súc tích (chỉ chứa các quan hệ).
  - *Nhược điểm:* Quá trình trích xuất đồ thị (Entity Extraction) tốn kém, phụ thuộc lớn vào chất lượng của LLM trích xuất. Nếu đồ thị bị thiếu node/quan hệ, câu trả lời sẽ bị sai.

### 2. Pipeline Xây Dựng GraphRAG
Hệ thống được xây dựng qua 4 bước chính:
1. **Crawling (Thu thập dữ liệu):** Sử dụng `wikipedia` API để cào nội dung các bài báo về các công ty AI/Công nghệ.
2. **Entity & Relation Extraction (Trích xuất tri thức):** Sử dụng Prompt Engineering với LLM (`gpt-4o-mini`) để trích xuất văn bản thô thành các bộ ba tri thức (Triples). Xử lý khéo léo lỗi định dạng trả về của LLM (Markdown JSON fences).
3. **Graph Construction (Xây dựng đồ thị):** Đưa các bộ ba thu được vào Database đồ thị chuẩn công nghiệp **Neo4j**, quản lý index và merge node để tránh trùng lặp dữ liệu (Deduplication).
4. **Multi-hop Querying (Truy vấn đa bước):** Khi người dùng đặt câu hỏi, dùng LLM xác định các "Seed Entities" -> Truy xuất đồ thị con (2-hop subgraph) xung quanh seed đó -> Biến đổi đồ thị thành text (Textualization) để làm Context cho LLM trả lời.

### 3. Đánh Giá Hiệu Năng (Benchmark)
Phương pháp **LLM-as-a-Judge** được sử dụng để chấm điểm tự động.
- Qua thử nghiệm thực tế, Flat RAG cho độ chính xác bao quát tốt hơn nếu thông tin chưa được trích xuất kỹ vào Graph.
- Ngược lại, **GraphRAG vượt trội hơn ở các câu hỏi truy vấn cấu trúc/quan hệ** (VD: "Mối quan hệ giữa cty A và cty B là gì?") và **tiết kiệm chi phí token lên tới 10 lần** ở khâu truy vấn so với nhồi toàn bộ raw text như Flat RAG.

## 🚀 Hướng Dẫn Chạy Project

1. Chạy file cào dữ liệu:
```bash
python 01_crawl_corpus.py
```
2. Trích xuất thực thể vào Neo4j (đảm bảo Neo4j Docker đang chạy):
```bash
python 02_extract_entities.py
```
3. Xây dựng baseline Flat RAG (ChromaDB):
```bash
python 03_build_flat_rag.py
```
4. Chạy file đánh giá (Benchmark so sánh 2 hệ thống):
```bash
python 05_benchmark.py
```
