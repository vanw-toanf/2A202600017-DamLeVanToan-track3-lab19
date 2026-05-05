import json
from pathlib import Path

def generate_report():
    with open("results/benchmark_summary.json") as f:
        summary = json.load(f)
        
    with open("results/extraction_summary.json") as f:
        extraction = json.load(f)

    report = f"""# Báo cáo Benchmark GraphRAG vs Flat RAG (Lab 19)

## 1. Phân tích chi phí xây dựng đồ thị (Graph Construction)

- **Tổng số bài viết xử lý:** {extraction['total_articles']}
- **Tổng số bộ ba (triples) trích xuất:** {extraction['total_triples']}
- **Số lượng Node trong Neo4j:** {extraction['neo4j_nodes']}
- **Số lượng Relationship trong Neo4j:** {extraction['neo4j_relationships']}
- **Tổng số Token LLM sử dụng (GPT-4o-mini):** {extraction['total_tokens']:,}
- **Chi phí ước tính:** ${extraction['estimated_cost_usd']:.4f}

## 2. Bảng so sánh kết quả {summary['total_questions']} câu hỏi Benchmark

Dưới đây là thống kê tổng quan so sánh hiệu năng giữa hai hệ thống:

| Tiêu chí | Flat RAG | GraphRAG |
|----------|----------|----------|
| Số câu thắng (LLM Judge) | **{summary['flat_rag_wins']}** | **{summary['graph_rag_wins']}** |
| Số câu hòa | \multicolumn{{2}}{{c|}}{{{summary['ties']}}} |
| Độ trễ trung bình (s) | {summary['avg_latency_flat']:.2f} | **{summary['avg_latency_graph']:.2f}** |
| Tổng Token sử dụng | {summary['total_tokens_flat']:,} | **{summary['total_tokens_graph']:,}** |
| Điểm đánh giá trung bình (/30) | {summary['avg_score_flat']:.1f} | {summary['avg_score_graph']:.1f} |
| Ước tính chi phí truy vấn | ${summary['estimated_cost_flat_usd']:.4f} | **${summary['estimated_cost_graph_usd']:.4f}** |

> **🏆 Hệ thống chiến thắng chung cuộc:** {summary['overall_winner']}

## 3. Kết luận ngắn gọn

- **Về độ chính xác:** Flat RAG thường trả lời đầy đủ và bao quát hơn vì truy xuất được nguyên văn đoạn text thô, hữu ích nếu thực thể chưa được trích xuất hoàn thiện vào GraphRAG. GraphRAG lại rất mạnh ở các câu hỏi yêu cầu liên kết phức tạp.
- **Về chi phí và tốc độ:** GraphRAG tốn chi phí ban đầu để xây dựng đồ thị, nhưng ở pha truy vấn (Querying), GraphRAG **nhanh hơn đáng kể** và **tiết kiệm token rất nhiều** so với Flat RAG do chỉ truyền các quan hệ đồ thị thu gọn thay vì nguyên văn nhiều đoạn văn bản lớn.
"""
    
    with open("results/benchmark_report.md", "w") as f:
        f.write(report)
        
    print("Report generated at results/benchmark_report.md")

if __name__ == "__main__":
    generate_report()
