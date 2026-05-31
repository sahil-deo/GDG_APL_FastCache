import os
import json
import time
import urllib.request
import urllib.error
import streamlit as st

from fastcache import SemanticCache
from fastcache.embedders.gemini import GeminiEmbedder

# Define the LLM fallback function
def call_gemini_llm(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("FASTCACHE_GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable is not set."
        
    model = "gemini-3.1-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"API Error {e.code}: {e.read().decode('utf-8')}"
    except Exception as e:
        return f"Request failed: {str(e)}"

# Initialize cache (cached by Streamlit so it persists across reruns)
@st.cache_resource
def get_cache():
    return SemanticCache(
        embedder=GeminiEmbedder(model="gemini-embedding-2")
    )

def main():
    st.set_page_config(page_title="FastCache Demo", page_icon="⚡")
    st.title("⚡ FastCache LLM Demo")
    st.markdown("This app uses **gemini-3.1-flash-lite** for generation and **gemini-embedding-2** for semantic caching.")
    
    # Get cache instance
    try:
        cache = get_cache()
    except Exception as e:
        st.error(f"Failed to initialize cache: {e}")
        st.info("Make sure you have GEMINI_API_KEY set in your environment.")
        return

    # Check for API key early
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("FASTCACHE_GEMINI_API_KEY")):
        st.warning("⚠️ GEMINI_API_KEY is not set. The application will not be able to call the Gemini API.")

    # Sidebar stats
    with st.sidebar:
        st.header("Cache Stats")
        stats = cache.stats
        st.metric("Total Queries", stats.total_queries)
        st.metric("Cache Hits", stats.cache_hits)
        st.metric("Hit Rate", f"{stats.hit_rate*100:.1f}%")
        st.metric("Avg Latency (ms)", f"{stats.avg_total_latency_ms:.1f}")
        
        st.markdown("---")
        if st.button("Clear Cache"):
            cache.invalidate(all=True)
            st.rerun()

    prompt = st.text_area("Enter your prompt:", height=150)
    
    if st.button("Submit", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return
            
        with st.spinner("Processing..."):
            start_time = time.time()
            
            # Query the cache
            try:
                response = cache.query(prompt, call_gemini_llm)
                latency_ms = (time.time() - start_time) * 1000
                
                st.success(f"Response (took {latency_ms:.1f}ms):")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

if __name__ == "__main__":
    main()
