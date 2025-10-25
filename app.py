import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os
import re
from query_helpers import (
    get_highest_wheat_punjab, 
    get_wheat_production_multi_year,
    get_highest_crop_gujarat,
    get_highest_rice_karnataka,
    get_annual_rainfall_subdivision,
    detect_query_type,
    list_crops_gujarat,
    compare_wheat_punjab_gujarat,
    get_average_rainfall_subdivision,
    get_rainfall_trend_subdivision,
    get_highest_crop_gujarat,
    get_highest_rice_karnataka,
    get_annual_rainfall_subdivision,
    detect_query_type
)

from advanced_helpers import handle_complex_query

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not set in .env!")
    st.stop()

st.set_page_config(page_title="Project Samarth", page_icon="🌾", layout="wide")

@st.cache_resource
def load_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.load_local(
        "vectorstore/samarth_vectorstore", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.1, 
        google_api_key=GEMINI_API_KEY
    )
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), 
        return_source_documents=True
    )

def handle_aggregate_query(prompt):
    """Handle queries that need code-based computation"""
    prompt_lower = prompt.lower()
    
    # Wheat Punjab - Highest production
    if 'highest' in prompt_lower and 'wheat' in prompt_lower and 'punjab' in prompt_lower:
        year_match = re.search(r'\b(20\d{2})\b', prompt)
        if year_match:
            year = year_match.group(1)
            district, value = get_highest_wheat_punjab(year)
            if district:
                return f"The district with the highest wheat production in Punjab in {year} was **{district}** with a production of **{value}**.\n\n**Source:** Wheat Punjab dataset (computed from data/wheat_punjab_clean.csv)"
    
    # List wheat production multi-year
    if 'list' in prompt_lower and 'wheat' in prompt_lower and ('last' in prompt_lower or 'past' in prompt_lower):
        district_match = re.search(r'(gurdaspur|pathankot|amritsar|ludhiana)', prompt_lower)
        if district_match:
            district = district_match.group(1).title()
            years = [str(y) for y in range(2022, 2017, -1)]  # Last 5 years
            data = get_wheat_production_multi_year(district, years)
            if data:
                result = f"**Wheat production for {district} (last 5 years):**\n\n"
                for year, value in sorted(data.items(), reverse=True):
                    result += f"- **{year}**: {value}\n"
                result += "\n**Source:** Wheat Punjab dataset"
                return result
    
    # List crops in Gujarat
    if ('crops' in prompt_lower or 'crop' in prompt_lower) and 'gujarat' in prompt_lower and ('produce' in prompt_lower or 'grown' in prompt_lower or 'what' in prompt_lower):
        crops = list_crops_gujarat()
        if crops:
            result = f"**Crops produced in Gujarat:**\n\n"
            for crop in crops:
                result += f"- {crop}\n"
            result += "\n**Source:** Gujarat Crops dataset"
            return result
    
    # Gujarat - Highest crop
    if 'highest' in prompt_lower and ('gujarat' in prompt_lower or 'crop' in prompt_lower):
        if 'yield' in prompt_lower:
            crop, value = get_highest_crop_gujarat('yield')
            metric = 'yield'
        elif 'area' in prompt_lower:
            crop, value = get_highest_crop_gujarat('area')
            metric = 'area'
        else:
            crop, value = get_highest_crop_gujarat('production')
            metric = 'production'
        
        if crop:
            return f"The crop with the highest {metric} in Gujarat is **{crop}** with a {metric} of **{value}**.\n\n**Source:** Gujarat Crops dataset"
    
    # Compare wheat production Punjab vs Gujarat
    if 'compare' in prompt_lower and 'wheat' in prompt_lower and ('punjab' in prompt_lower or 'gujarat' in prompt_lower):
        year_match = re.search(r'\b(20\d{2})\b', prompt)
        year = year_match.group(1) if year_match else '2022'
        
        punjab_total, gujarat_prod = compare_wheat_punjab_gujarat(year)
        if punjab_total is not None and gujarat_prod is not None:
            result = f"**Wheat production comparison for {year}:**\n\n"
            result += f"- **Punjab (Total)**: {punjab_total:.1f}\n"
            result += f"- **Gujarat**: {gujarat_prod}\n"
            result += f"\n**Source:** Wheat Punjab dataset, Gujarat Crops dataset"
            return result
    
    # Karnataka Rice - Highest production
    if 'highest' in prompt_lower and 'rice' in prompt_lower and 'karnataka' in prompt_lower:
        district, value = get_highest_rice_karnataka('all_seasons_production')
        if district:
            return f"The district with the highest rice production in Karnataka is **{district}** with an all-seasons production of **{value}** tonnes.\n\n**Source:** Rice Karnataka dataset"
    
    # Average rainfall
    if 'average' in prompt_lower and 'rainfall' in prompt_lower:
        subdivision_keywords = ['andaman', 'punjab', 'gujarat', 'karnataka', 'saurashtra']
        subdivision = None
        for keyword in subdivision_keywords:
            if keyword in prompt_lower:
                subdivision = keyword
                break
        
        if subdivision:
            result_data = get_average_rainfall_subdivision(subdivision)
            if result_data:
                avg, count = result_data
                result = f"**Average annual rainfall in {subdivision.title()} (2010-2017):**\n\n"
                result += f"- **Average**: {avg:.1f} mm\n"
                result += f"- **Years analyzed**: {count}\n"
                result += "\n**Source:** Rainfall Subdivision dataset"
                return result
    
    # Rainfall trends
    if 'trend' in prompt_lower and 'rainfall' in prompt_lower:
        subdivision_keywords = ['andaman', 'punjab', 'gujarat', 'karnataka', 'india']
        subdivision = None
        for keyword in subdivision_keywords:
            if keyword in prompt_lower:
                subdivision = keyword if keyword != 'india' else 'andaman'
                break
        
        if subdivision:
            trend_data = get_rainfall_trend_subdivision(subdivision)
            if trend_data is not None:
                result = f"**Rainfall trend for {subdivision.title()} (2010-2017):**\n\n"
                for _, row in trend_data.iterrows():
                    result += f"- **{int(row['year'])}**: {row['annual']:.1f} mm\n"
                result += "\n**Source:** Rainfall Subdivision dataset"
                return result
    
    # Annual rainfall query
    if 'rainfall' in prompt_lower and 'annual' in prompt_lower:
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', prompt)
        subdivision_keywords = ['andaman', 'punjab', 'gujarat', 'karnataka', 'saurashtra']
        subdivision = None
        for keyword in subdivision_keywords:
            if keyword in prompt_lower:
                subdivision = keyword
                break
        
        if year_match and subdivision:
            year = year_match.group(1)
            rainfall = get_annual_rainfall_subdivision(subdivision, year)
            if rainfall:
                return f"The annual rainfall in **{subdivision.title()}** subdivision in **{year}** was **{rainfall} mm**.\n\n**Source:** Rainfall Subdivision dataset"
    
    return None

st.title("🌾 Project Samarth - Agricultural & Climate Q&A")
st.markdown("*Intelligent Q&A system powered by real government data from data.gov.in*")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.info("""
    This system answers questions about:
    - Wheat production (Punjab)
    - Rice production (Karnataka)  
    - Crop statistics (Gujarat)
    - Rainfall patterns (India)
    """)
    
    st.markdown("---")
    st.subheader("💡 Sample Questions")
    
    samples = [
        "What was the wheat production in Amritsar in 2022?",
        "Which district had the highest wheat production in Punjab in 2021?",
        "Which crop had the highest yield in Gujarat?",
        "List wheat production for Gurdaspur for the last 5 years",
        "What was the annual rainfall in Andaman & Nicobar Islands in 2010?"
    ]
    
    for sample in samples:
        if st.button(f"📌 {sample[:45]}...", key=sample, use_container_width=True):
            st.session_state.sample_query = sample

if "messages" not in st.session_state:
    st.session_state.messages = []

qa_chain = load_chain()

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Get input from chat field
prompt = st.chat_input("Ask about crop production, rainfall...")

# Check for sample query
if "sample_query" in st.session_state:
    prompt = st.session_state.sample_query
    del st.session_state.sample_query

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing datasets..."):
            result = handle_complex_query(prompt)
            if result and isinstance(result, dict) and 'summary' in result:
                st.markdown(result['summary'])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['summary']
                })
            else:
                # fallback to your LLM/vector pipeline
                # Try aggregate query first
                query_type = detect_query_type(prompt)
                aggregate_response = None

                if query_type in ['aggregate', 'timeseries']:
                    aggregate_response = handle_aggregate_query(prompt)

                if aggregate_response:
                    st.markdown(aggregate_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": aggregate_response
                    })
                else:
                    # Fall back to RAG
                    try:
                        result = qa_chain({"question": prompt})
                        response = result.get("answer", "I couldn't find information about that in the datasets.")

                        if result.get("sources"):
                            sources = "\n\n**📊 Sources:**\n" + "\n".join(
                                f"- {src}" for src in result["sources"].split(", ")[:3]
                            )
                            response += sources

                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("🌾 Project Samarth | Data sources: data.gov.in (Punjab Wheat, Karnataka Rice, Gujarat Crops, IMD Rainfall)")
