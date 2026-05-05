"""
LAB 19 - Step 5: Benchmark - GraphRAG vs Flat RAG
Compares GraphRAG (Neo4j Knowledge Graph) vs Flat RAG (ChromaDB) across 10 questions.
Measures: answer quality, latency, token usage.
"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 10 benchmark questions (multi-hop / relational) ──────────────────────────
BENCHMARK_QUESTIONS = [
    "Who founded OpenAI and what is their current role?",
    "Which AI companies were co-founded by former Google or DeepMind employees?",
    "What products has NVIDIA developed that are used in AI research?",
    "Who is the CEO of Anthropic and what is their background?",
    "What is the relationship between Google Brain and Google DeepMind?",
    "Which companies has Sam Altman been involved with as a founder or investor?",
    "What large language models have been developed by Meta AI?",
    "How does GPT-4 relate to OpenAI and what companies use it?",
    "What AI research institutions are associated with Stanford University?",
    "Which autonomous vehicle companies have received investment from major tech firms?",
    "What is the role of Demis Hassabis at Google DeepMind?",
    "Which institutions or labs has Yoshua Bengio been affiliated with?",
    "How was Google Brain formed and who were the key people involved?",
    "What are some notable AI products or models developed by Google?",
    "What is the relationship between Microsoft and OpenAI?",
    "Which companies are competitors to OpenAI in the large language model space?",
    "Who is Yann LeCun and what is his role at Meta?",
    "What technologies or models are associated with Stability AI?",
    "Which AI researchers are considered the 'Godfathers of AI'?",
    "What is the main product or focus of Hugging Face?",
]


# ─── Flat RAG answer ──────────────────────────────────────────────────────────

def answer_flat_rag(question: str) -> dict:
    """Answer using ChromaDB flat vector RAG."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    t0 = time.time()

    chroma_client = chromadb.PersistentClient(path="data/chromadb")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        collection = chroma_client.get_collection(
            name="lab19_flat_rag",
            embedding_function=embed_fn,
        )
    except Exception as e:
        return {
            "question": question,
            "method": "FlatRAG",
            "answer": f"[ChromaDB not available: {e}]",
            "context_chunks": 0,
            "usage": {},
            "latency_s": 0,
        }

    results = collection.query(query_texts=[question], n_results=5)

    context_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(context_chunks)

    t_retrieve = time.time() - t0

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise AI assistant. Answer the question based ONLY "
                    "on the provided document excerpts. Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )

    latency = time.time() - t0
    answer = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "retrieve_s": round(t_retrieve, 3),
        "latency_s": round(latency, 3),
    }

    return {
        "question": question,
        "method": "FlatRAG",
        "answer": answer,
        "context_chunks": len(context_chunks),
        "usage": usage,
        "latency_s": round(latency, 3),
    }


# ─── GraphRAG answer ──────────────────────────────────────────────────────────

def answer_graphrag(question: str) -> dict:
    """Answer using GraphRAG (Neo4j knowledge graph traversal)."""
    import sys
    sys.path.insert(0, ".")
    from four_graphrag_query import GraphRAGEngine  # noqa

    engine = GraphRAGEngine()
    result = engine.query(question, verbose=False)
    engine.close()

    return {
        "question": question,
        "method": "GraphRAG",
        "answer": result["answer"],
        "entities_extracted": result["entities_extracted"],
        "nodes_found": result["nodes_found"],
        "triples_count": result["triples_count"],
        "usage": result["usage"],
        "latency_s": result["total_latency_s"],
    }


# ─── LLM Judge ────────────────────────────────────────────────────────────────

def judge_answers(question: str, flat_answer: str, graph_answer: str) -> dict:
    """Use GPT-4o-mini as judge to score both answers."""
    prompt = f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Question: {question}

Answer A (Flat RAG): {flat_answer}

Answer B (GraphRAG): {graph_answer}

Evaluate both answers on:
1. Accuracy (0-10): Factual correctness
2. Completeness (0-10): How fully it answers the question
3. Relevance (0-10): How relevant the info is to the question

Return ONLY valid JSON:
{{
  "flat_rag": {{"accuracy": X, "completeness": X, "relevance": X, "total": X, "comment": "..."}},
  "graph_rag": {{"accuracy": X, "completeness": X, "relevance": X, "total": X, "comment": "..."}},
  "winner": "flat_rag|graph_rag|tie",
  "reason": "brief reason"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fair and precise answer evaluator. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    import re
    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "flat_rag": {"accuracy": 0, "completeness": 0, "relevance": 0, "total": 0, "comment": "parse error"},
        "graph_rag": {"accuracy": 0, "completeness": 0, "relevance": 0, "total": 0, "comment": "parse error"},
        "winner": "tie",
        "reason": "Could not parse judge response",
    }


# ─── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("🏆  LAB 19 – GraphRAG vs Flat RAG Benchmark")
    print("=" * 70)
    print(f"📋  Questions: {len(BENCHMARK_QUESTIONS)}")
    print()

    # Try to import GraphRAG engine
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("graphrag", "04_graphrag_query.py")
        mod = importlib.util.load_from_spec(spec)  # just check import
    except Exception:
        pass

    benchmark_results = []

    # Aggregate stats
    flat_wins = 0
    graph_wins = 0
    ties = 0
    flat_latencies = []
    graph_latencies = []
    flat_tokens = []
    graph_tokens = []
    flat_scores = []
    graph_scores = []

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(BENCHMARK_QUESTIONS)}] {question}")
        print("-" * 60)

        # ── Flat RAG ──
        print("  🔍 Running Flat RAG...")
        try:
            flat_result = answer_flat_rag(question)
        except Exception as e:
            flat_result = {
                "question": question,
                "method": "FlatRAG",
                "answer": f"[Error: {e}]",
                "usage": {},
                "latency_s": 0,
            }
        print(f"  ✅ Flat RAG ({flat_result['latency_s']:.2f}s): {flat_result['answer'][:120]}...")

        # ── GraphRAG ──
        print("  🔍 Running GraphRAG...")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("graphrag_module", "04_graphrag_query.py")
            graphrag_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(graphrag_mod)
            engine = graphrag_mod.GraphRAGEngine()
            graph_result_raw = engine.query(question, verbose=False)
            engine.close()
            graph_result = {
                "question": question,
                "method": "GraphRAG",
                "answer": graph_result_raw["answer"],
                "entities_extracted": graph_result_raw["entities_extracted"],
                "nodes_found": graph_result_raw["nodes_found"],
                "triples_count": graph_result_raw["triples_count"],
                "usage": graph_result_raw["usage"],
                "latency_s": graph_result_raw["total_latency_s"],
            }
        except Exception as e:
            graph_result = {
                "question": question,
                "method": "GraphRAG",
                "answer": f"[Error: {e}]",
                "usage": {},
                "latency_s": 0,
            }
        print(f"  ✅ GraphRAG ({graph_result['latency_s']:.2f}s): {graph_result['answer'][:120]}...")

        # ── Judge ──
        print("  ⚖️  Judging answers...")
        judgment = judge_answers(question, flat_result["answer"], graph_result["answer"])
        winner = judgment.get("winner", "tie")
        if winner == "flat_rag":
            flat_wins += 1
        elif winner == "graph_rag":
            graph_wins += 1
        else:
            ties += 1
        print(f"  🏅 Winner: {winner.upper()} | Reason: {judgment.get('reason', '')}")

        # Collect stats
        flat_latencies.append(flat_result["latency_s"])
        graph_latencies.append(graph_result["latency_s"])
        flat_tokens.append(flat_result.get("usage", {}).get("total_tokens", 0))
        graph_tokens.append(graph_result.get("usage", {}).get("total_tokens", 0))

        fr_scores = judgment.get("flat_rag", {})
        gr_scores = judgment.get("graph_rag", {})
        flat_scores.append(fr_scores.get("total", 0))
        graph_scores.append(gr_scores.get("total", 0))

        benchmark_results.append({
            "question_id": i,
            "question": question,
            "flat_rag": flat_result,
            "graph_rag": graph_result,
            "judgment": judgment,
        })

        time.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊  BENCHMARK SUMMARY")
    print("=" * 70)

    avg_flat_lat = sum(flat_latencies) / len(flat_latencies) if flat_latencies else 0
    avg_graph_lat = sum(graph_latencies) / len(graph_latencies) if graph_latencies else 0
    total_flat_tok = sum(flat_tokens)
    total_graph_tok = sum(graph_tokens)
    avg_flat_score = sum(flat_scores) / len(flat_scores) if flat_scores else 0
    avg_graph_score = sum(graph_scores) / len(graph_scores) if graph_scores else 0

    print(f"\n{'Metric':<35} {'Flat RAG':>12} {'GraphRAG':>12}")
    print("-" * 60)
    print(f"{'Wins (LLM Judge)':<35} {flat_wins:>12} {graph_wins:>12}  (ties: {ties})")
    print(f"{'Avg Latency (s)':<35} {avg_flat_lat:>12.2f} {avg_graph_lat:>12.2f}")
    print(f"{'Total Tokens Used':<35} {total_flat_tok:>12,} {total_graph_tok:>12,}")
    print(f"{'Avg LLM Score (/30)':<35} {avg_flat_score:>12.1f} {avg_graph_score:>12.1f}")

    cost_flat = total_flat_tok / 1_000_000 * 0.15
    cost_graph = total_graph_tok / 1_000_000 * 0.15
    print(f"{'Estimated Cost (USD)':<35} ${cost_flat:>11.4f} ${cost_graph:>11.4f}")

    overall_winner = "GraphRAG" if graph_wins > flat_wins else ("Flat RAG" if flat_wins > graph_wins else "Tie")
    print(f"\n🏆  Overall Winner: {overall_winner}")

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "total_questions": len(BENCHMARK_QUESTIONS),
        "flat_rag_wins": flat_wins,
        "graph_rag_wins": graph_wins,
        "ties": ties,
        "avg_latency_flat": round(avg_flat_lat, 3),
        "avg_latency_graph": round(avg_graph_lat, 3),
        "total_tokens_flat": total_flat_tok,
        "total_tokens_graph": total_graph_tok,
        "avg_score_flat": round(avg_flat_score, 2),
        "avg_score_graph": round(avg_graph_score, 2),
        "estimated_cost_flat_usd": round(cost_flat, 6),
        "estimated_cost_graph_usd": round(cost_graph, 6),
        "overall_winner": overall_winner,
    }

    with open("results/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("results/benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to:")
    print(f"   📄 results/benchmark_summary.json")
    print(f"   📄 results/benchmark_results.json")

    return summary, benchmark_results


if __name__ == "__main__":
    run_benchmark()
