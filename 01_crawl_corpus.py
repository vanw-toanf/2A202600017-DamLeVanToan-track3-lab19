"""
LAB 19 - Step 1: Crawl Wikipedia corpus about AI/Tech companies
Crawls ~100 articles and saves them to data/corpus/
"""
import wikipedia
import json
import os
import time
from pathlib import Path

DATA_DIR = Path("data/corpus")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# List of AI / Tech companies and related topics for the corpus
AI_COMPANY_TOPICS = [
    # Major AI Labs
    "OpenAI", "Anthropic (company)", "Google DeepMind", "Meta AI",
    "Mistral AI", "Cohere (company)", "Inflection AI", "Adept AI",
    "Character.AI", "Stability AI", "Midjourney", "Hugging Face",
    "Scale AI", "Runway (company)", "Perplexity AI", "xAI (company)",
    
    # Big Tech AI divisions
    "Google Brain", "Microsoft Research", "Amazon Web Services",
    "Apple Inc.", "IBM Research", "Baidu", "Alibaba Group",
    "Tencent", "ByteDance", "NVIDIA", "Intel", "AMD",
    "Qualcomm", "Samsung Electronics",
    
    # Classic AI / ML companies
    "DeepMind", "OpenAI", "Boston Dynamics", "iRobot",
    "Palantir Technologies", "Databricks", "DataRobot",
    "C3.ai", "UiPath", "Automation Anywhere",
    
    # Cloud & Infrastructure
    "Microsoft Azure", "Google Cloud Platform",
    "Oracle Corporation", "Snowflake Inc.", "MongoDB",
    
    # Robotics & Autonomous
    "Tesla", "Waymo", "Cruise (autonomous vehicle)", "Aurora Innovation",
    "Nuro", "Zoox", "Mobileye",
    
    # NLP / Language
    "OpenAI", "Cohere (company)", "AI21 Labs", "Aleph Alpha",
    "LightOn", "EleutherAI", "Together AI",
    
    # Key People / Founders (for relation extraction)
    "Sam Altman", "Elon Musk", "Greg Brockman", "Ilya Sutskever",
    "Demis Hassabis", "Shane Legg", "Mustafa Suleyman",
    "Yann LeCun", "Geoffrey Hinton", "Yoshua Bengio",
    "Andrew Ng", "Fei-Fei Li", "Jeff Dean", "Sanjay Ghemawat",
    "Larry Page", "Sergey Brin", "Mark Zuckerberg",
    "Satya Nadella", "Jensen Huang",
    "Dario Amodei", "Daniela Amodei",
    "Aidan Gomez", "Nick Frosst",
    "Emad Mostaque", "Robin Li",
    
    # Research institutions
    "Stanford Artificial Intelligence Laboratory",
    "MIT Computer Science and Artificial Intelligence Laboratory",
    "Berkeley Artificial Intelligence Research Lab",
    "Vector Institute",
    "Allen Institute for AI",
    
    # Notable AI systems / products
    "ChatGPT", "GPT-4", "BERT (language model)",
    "LaMDA", "Gemini (language model)",
    "Claude (language model)", "Llama (language model)",
    "Stable Diffusion", "DALL-E", "Midjourney",
    "AlphaGo", "AlphaFold",
    
    # AI topics for context
    "Artificial general intelligence", "Large language model",
    "Transformer (deep learning architecture)",
    "Reinforcement learning from human feedback",
    "Generative artificial intelligence",
]

# Deduplicate
AI_COMPANY_TOPICS = list(dict.fromkeys(AI_COMPANY_TOPICS))

wikipedia.set_lang("en")

def crawl_article(topic: str, save_dir: Path) -> dict | None:
    """Crawl a single Wikipedia article and save it."""
    safe_name = topic.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    filepath = save_dir / f"{safe_name}.json"
    
    # Skip if already crawled
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    
    try:
        page = wikipedia.page(topic, auto_suggest=True, redirect=True)
        content = {
            "title": page.title,
            "topic_query": topic,
            "url": page.url,
            "content": page.content[:8000],  # Limit to 8000 chars
            "summary": page.summary,
            "categories": page.categories[:20],
            "links": [l for l in page.links[:50] if len(l) < 50],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Saved: {page.title}")
        return content
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            content = {
                "title": page.title,
                "topic_query": topic,
                "url": page.url,
                "content": page.content[:8000],
                "summary": page.summary,
                "categories": page.categories[:20],
                "links": [l for l in page.links[:50] if len(l) < 50],
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            print(f"  ✅ (disambig) Saved: {page.title}")
            return content
        except Exception as ex:
            print(f"  ⚠️  Disambig failed for '{topic}': {ex}")
            return None
    except wikipedia.exceptions.PageError:
        print(f"  ❌ Not found: {topic}")
        return None
    except Exception as e:
        print(f"  ❌ Error for '{topic}': {e}")
        return None


def main():
    articles = []
    print(f"🚀 Starting crawl of {len(AI_COMPANY_TOPICS)} topics...")
    
    for i, topic in enumerate(AI_COMPANY_TOPICS, 1):
        print(f"[{i:3d}/{len(AI_COMPANY_TOPICS)}] Fetching: {topic}")
        art = crawl_article(topic, DATA_DIR)
        if art:
            articles.append(art)
        time.sleep(0.3)  # Be polite to Wikipedia API
    
    print(f"\n✅ Crawled {len(articles)} articles → saved to {DATA_DIR}/")
    
    # Save summary index
    index = [{"title": a["title"], "url": a["url"]} for a in articles]
    with open("data/corpus_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"📋 Index saved: data/corpus_index.json")
    return articles


if __name__ == "__main__":
    main()
