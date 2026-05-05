"""
LAB 19 - Step 4: GraphRAG Query Engine
Multi-hop traversal over Neo4j knowledge graph + LLM answer generation.
"""
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lab19password")

client = OpenAI(api_key=OPENAI_API_KEY)


class GraphRAGEngine:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    def close(self):
        self.driver.close()

    # ── 1. Entity extraction from question ────────────────────────────────────
    def extract_entities_from_question(self, question: str) -> list[str]:
        """Use GPT to identify key entities in the question."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract the key named entities (companies, people, products, technologies) from the question. Return ONLY a JSON array of strings. Example: [\"OpenAI\", \"Sam Altman\"]"},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()
        try:
            import json
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        # Fallback: simple word extraction
        words = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', question)
        return list(set(words))[:5]

    # ── 2. Find closest nodes in Neo4j ────────────────────────────────────────
    def find_nodes(self, entity_names: list[str]) -> list[str]:
        """Find node names that fuzzy-match the given entities."""
        found = []
        with self.driver.session() as session:
            for name in entity_names:
                # Exact match first
                result = session.run(
                    "MATCH (n) WHERE toLower(n.name) = toLower($name) RETURN n.name LIMIT 3",
                    name=name
                )
                rows = [r["n.name"] for r in result]
                
                if not rows:
                    # Partial match
                    result = session.run(
                        "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) RETURN n.name LIMIT 3",
                        name=name
                    )
                    rows = [r["n.name"] for r in result]
                
                found.extend(rows)
        return list(set(found))

    # ── 3. 2-hop graph traversal ───────────────────────────────────────────────
    def traverse_2hop(self, node_names: list[str]) -> list[dict]:
        """Get all paths within 2 hops from the seed nodes."""
        if not node_names:
            return []
        
        triples = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (seed)
                WHERE seed.name IN $names
                MATCH path = (seed)-[r1*1..2]-(neighbor)
                WITH 
                    startNode(relationships(path)[0]).name AS s,
                    type(relationships(path)[0]) AS rel,
                    endNode(relationships(path)[0]).name AS o
                RETURN DISTINCT s, rel, o
                LIMIT 150
                """,
                names=node_names,
            )
            for row in result:
                triples.append({
                    "subject": row["s"],
                    "relation": row["rel"],
                    "object": row["o"],
                })
        return triples

    # ── 4. Textualize graph context ────────────────────────────────────────────
    def textualize(self, triples: list[dict], seed_nodes: list[str]) -> str:
        """Convert triples to readable text for LLM."""
        if not triples:
            return "No relevant knowledge graph data found."
        
        lines = [f"Knowledge Graph Context (seed nodes: {', '.join(seed_nodes)}):\n"]
        
        # Group by subject
        from collections import defaultdict
        by_subject = defaultdict(list)
        for t in triples:
            by_subject[t["subject"]].append(f"{t['relation']} → {t['object']}")
        
        for subj, rels in by_subject.items():
            lines.append(f"\n{subj}:")
            for r in rels[:10]:  # limit per node
                lines.append(f"  - {r}")
        
        return "\n".join(lines)

    # ── 5. LLM answer generation ───────────────────────────────────────────────
    def answer_with_graph(self, question: str, context: str) -> tuple[str, dict]:
        """Generate answer using graph context."""
        t0 = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a precise AI assistant using knowledge graph data.
Answer the question based ONLY on the provided knowledge graph context.
If the context doesn't contain enough info, say so honestly.
Be concise but complete. Cite the graph relationships."""},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
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
            "latency_s": round(latency, 3),
        }
        return answer, usage

    # ── 6. Full GraphRAG pipeline ──────────────────────────────────────────────
    def query(self, question: str, verbose: bool = True) -> dict:
        """Full GraphRAG pipeline: question → graph traversal → LLM answer."""
        t_start = time.time()
        
        if verbose:
            print(f"\n🔍 GraphRAG Query: {question}")
        
        # Step 1: Extract entities
        entities = self.extract_entities_from_question(question)
        if verbose:
            print(f"  📌 Entities: {entities}")
        
        # Step 2: Find nodes
        node_names = self.find_nodes(entities)
        if verbose:
            print(f"  🔵 Found nodes: {node_names}")
        
        # Step 3: 2-hop traversal
        triples = self.traverse_2hop(node_names)
        if verbose:
            print(f"  🔗 Retrieved {len(triples)} triples from graph")
        
        # Step 4: Textualize
        context = self.textualize(triples, node_names)
        
        # Step 5: Answer
        answer, usage = self.answer_with_graph(question, context)
        
        total_latency = time.time() - t_start
        
        if verbose:
            print(f"  ✅ Answer: {answer[:200]}...")
            print(f"  ⏱️  Latency: {total_latency:.2f}s | Tokens: {usage['total_tokens']}")
        
        return {
            "question": question,
            "method": "GraphRAG",
            "entities_extracted": entities,
            "nodes_found": node_names,
            "triples_count": len(triples),
            "context": context,
            "answer": answer,
            "usage": usage,
            "total_latency_s": round(total_latency, 3),
        }

    # ── 7. Cypher for visualization ────────────────────────────────────────────
    def get_visualization_cypher(self, node_names: list[str]) -> str:
        """Return Cypher query to visualize subgraph in Neo4j Browser."""
        names_str = str(node_names).replace("'", '"')
        return f"""// Subgraph visualization for: {node_names}
MATCH (seed)
WHERE seed.name IN {names_str}
MATCH path = (seed)-[r*1..2]-(neighbor)
RETURN path
LIMIT 100"""


if __name__ == "__main__":
    engine = GraphRAGEngine()
    
    # Demo query 1
    result1 = engine.query("What is OpenAI?")
    print("\n" + "="*60)
    print("ANSWER:", result1["answer"])
    
    # Demo query 2 (multi-hop)
    result2 = engine.query("Which AI companies were co-founded by former Google employees?")
    print("\n" + "="*60)
    print("ANSWER:", result2["answer"])
    
    engine.close()
