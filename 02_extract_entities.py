"""
LAB 19 - Step 2: Entity & Relation Extraction → Neo4j
Uses GPT-4o-mini to extract (subject, relation, object) triples
and stores them in Neo4j knowledge graph.
"""
import os
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lab19password")

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Neo4j helpers ────────────────────────────────────────────────────────────

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_all(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️  Cleared all nodes and relationships")
    
    def create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)")
            session.run("CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)")
        print("📑 Created indexes")
    
    def merge_triple(self, subject: str, relation: str, obj: str,
                     subject_type: str = "Entity", obj_type: str = "Entity",
                     source_article: str = ""):
        """MERGE a triple into the graph."""
        relation_clean = re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_"))
        if not relation_clean:
            return
        
        cypher = f"""
        MERGE (s:{subject_type} {{name: $subject}})
        ON CREATE SET s.type = $subject_type, s.sources = [$source]
        ON MATCH SET s.sources = CASE 
            WHEN $source IN s.sources THEN s.sources 
            ELSE s.sources + $source 
        END

        MERGE (o:{obj_type} {{name: $obj}})
        ON CREATE SET o.type = $obj_type, o.sources = [$source]
        ON MATCH SET o.sources = CASE 
            WHEN $source IN o.sources THEN o.sources 
            ELSE o.sources + $source 
        END

        MERGE (s)-[r:{relation_clean}]->(o)
        ON CREATE SET r.source = $source, r.count = 1
        ON MATCH SET r.count = r.count + 1
        """
        with self.driver.session() as session:
            session.run(cypher, subject=subject, obj=obj,
                       subject_type=subject_type, obj_type=obj_type,
                       source=source_article)
    
    def get_stats(self):
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        return nodes, rels


# ─── LLM Entity Extraction ────────────────────────────────────────────────────

EXTRACTION_PROMPT_PREFIX = """You are a knowledge graph extraction expert. 
Extract entities and relations from the text to form RDF-style triples.

Rules:
1. Focus on: companies, people, products, technologies, locations, dates, events
2. Relations should be clear verbs: FOUNDED_BY, WORKS_AT, ACQUIRED, INVESTED_IN, DEVELOPED, LOCATED_IN, FOUNDED_IN, CEO_OF, CO_FOUNDED_BY, COMPETES_WITH, PARTNERED_WITH, PART_OF, RESEARCH_AREA, etc.
3. Normalize entity names (use full names, avoid pronouns)
4. Classify entities: Company | Person | Product | Technology | Location | Year | Organization
5. Extract 15-30 high-quality triples per text chunk

Output ONLY valid JSON array:
[
  {
    "subject": "entity name",
    "subject_type": "Company|Person|Product|Technology|Location|Year|Organization",
    "relation": "RELATION_TYPE",
    "object": "entity name",
    "object_type": "Company|Person|Product|Technology|Location|Year|Organization"
  }
]

Text:
"""

def extract_triples_llm(text: str, article_title: str) -> list[dict]:
    """Use GPT-4o-mini to extract triples from text."""
    # Truncate if too long
    text = text[:3000]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledge graph extraction assistant. Always output valid JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT_PREFIX + text}
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        
        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()
        
        # Extract JSON array from response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            triples = json.loads(match.group())
            return triples, response.usage
        return [], response.usage
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error for '{article_title}': {e}")
        return [], None
    except Exception as e:
        print(f"  ❌ LLM error for '{article_title}': {e}")
        return [], None


def main():
    corpus_dir = Path("data/corpus")
    triples_dir = Path("data/triples")
    triples_dir.mkdir(exist_ok=True)
    
    # Connect to Neo4j
    print(f"🔗 Connecting to Neo4j: {NEO4J_URI}")
    graph = Neo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    graph.clear_all()
    graph.create_indexes()
    
    articles = sorted(corpus_dir.glob("*.json"))
    print(f"📚 Found {len(articles)} articles to process\n")
    
    total_triples = 0
    total_tokens = 0
    all_triples_data = []
    
    for i, article_path in enumerate(articles, 1):
        with open(article_path, encoding="utf-8") as f:
            article = json.load(f)
        
        title = article["title"]
        content = article.get("content", "")
        summary = article.get("summary", "")
        
        # Use summary + first part of content
        text_to_process = f"Title: {title}\n\nSummary: {summary}\n\n{content[:2000]}"
        
        print(f"[{i:3d}/{len(articles)}] Extracting from: {title[:60]}")
        
        # Check if already processed
        triple_path = triples_dir / article_path.name
        if triple_path.exists():
            with open(triple_path) as f:
                saved = json.load(f)
            triples = saved.get("triples", [])
            tokens_used = saved.get("tokens", 0)
            print(f"  📁 Loaded {len(triples)} cached triples")
        else:
            triples, usage = extract_triples_llm(text_to_process, title)
            tokens_used = usage.total_tokens if usage else 0
            
            # Save triples
            with open(triple_path, "w", encoding="utf-8") as f:
                json.dump({
                    "article": title,
                    "triples": triples,
                    "tokens": tokens_used,
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ Extracted {len(triples)} triples ({tokens_used} tokens)")
            time.sleep(0.5)  # Rate limit
        
        # Push to Neo4j
        for triple in triples:
            try:
                graph.merge_triple(
                    subject=triple["subject"],
                    relation=triple["relation"],
                    obj=triple["object"],
                    subject_type=triple.get("subject_type", "Entity"),
                    obj_type=triple.get("object_type", "Entity"),
                    source_article=title,
                )
            except Exception as e:
                pass  # Skip malformed triples
        
        total_triples += len(triples)
        total_tokens += tokens_used if tokens_used else 0
        all_triples_data.extend(triples)
    
    # Final stats
    nodes, rels = graph.get_stats()
    
    print(f"\n{'='*60}")
    print(f"✅ Knowledge Graph Built!")
    print(f"   📊 Total triples extracted : {total_triples:,}")
    print(f"   🔵 Nodes in Neo4j          : {nodes:,}")
    print(f"   🔗 Relationships in Neo4j  : {rels:,}")
    print(f"   🪙  Total tokens used       : {total_tokens:,}")
    
    cost_usd = total_tokens / 1_000_000 * 0.15  # gpt-4o-mini ~$0.15/1M tokens
    print(f"   💰 Estimated cost           : ${cost_usd:.4f}")
    print(f"{'='*60}")
    
    # Save summary
    summary_data = {
        "total_articles": len(articles),
        "total_triples": total_triples,
        "neo4j_nodes": nodes,
        "neo4j_relationships": rels,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
    }
    with open("results/extraction_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    graph.close()
    return summary_data


if __name__ == "__main__":
    main()
