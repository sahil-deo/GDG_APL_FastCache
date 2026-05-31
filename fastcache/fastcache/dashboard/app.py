import streamlit as st
import pandas as pd
import numpy as np

# This will be injected by serve_dashboard()
_global_cache = None

def get_cache():
    return _global_cache

def main():
    st.set_page_config(page_title="FastCache Dashboard", layout="wide")
    
    cache = get_cache()
    if not cache:
        st.error("Cache instance not found! Please run via SemanticCache.serve_dashboard().")
        return
        
    st.title("FastCache Dashboard")
    
    stats = cache.stats
    
    # 1. Overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", stats.total_queries)
    col2.metric("Hit Rate", f"{stats.hit_rate*100:.1f}%")
    col3.metric("Avg Latency (ms)", f"{stats.avg_total_latency_ms:.1f}")
    col4.metric("Entries in Cache", stats.total_entries)
    
    st.markdown("---")
    
    # 3. Stats Charts
    st.subheader("Performance Metrics")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Latency Distribution**")
        df_lat = pd.DataFrame({
            "Type": ["Cache Hits", "Cache Misses (LLM)"],
            "Latency (ms)": [stats.avg_hit_latency_ms, stats.avg_miss_latency_ms]
        })
        st.bar_chart(df_lat.set_index("Type"))
        
    with c2:
        st.markdown("**Entries by Namespace**")
        if stats.entries_by_namespace:
            df_ns = pd.DataFrame(list(stats.entries_by_namespace.items()), columns=["Namespace", "Count"])
            st.bar_chart(df_ns.set_index("Namespace"))
        else:
            st.info("No entries yet.")
            
    st.markdown("---")
    
    # 4. Cache Explorer
    st.subheader("Cache Explorer")
    if hasattr(cache.store, "_entries") and cache.store._entries:
        all_entries = []
        for ns, entries in cache.store._entries.items():
            for e in entries:
                all_entries.append({
                    "ID": e.id[:8],
                    "Namespace": e.namespace,
                    "Query": e.query[:50] + "..." if len(e.query) > 50 else e.query,
                    "Hit Count": e.hit_count,
                    "Age (s)": round(e.age_seconds),
                    "TTL": "No expiry" if e.ttl == 0 else round(e.ttl_remaining)
                })
        
        df_explorer = pd.DataFrame(all_entries)
        st.dataframe(df_explorer, use_container_width=True)
    else:
        st.info("Cache is empty or store is not InMemoryStore (explorer not supported for Redis yet).")

if __name__ == "__main__":
    main()
