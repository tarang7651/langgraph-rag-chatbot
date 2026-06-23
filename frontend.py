import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from backend import workflow, process_pdf

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📄 Document Upload")
    st.markdown("Upload a PDF to use RAG capabilities.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                file_bytes = uploaded_file.getvalue()
                process_pdf(file_bytes)
                st.success("PDF processed successfully! You can now ask questions about it.")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Main layout */
.block-container {
    max-width: 100%;
    margin: auto;
    padding-top: 2rem;
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Chat message spacing */
.stChatMessage {
    padding: 0.75rem;
    border-radius: 12px;
}

/* Welcome screen */
.welcome {
    text-align:center;
    margin-top:15vh;
}

.welcome h1 {
    font-size:3rem;
    margin-bottom:0.5rem;
}

.welcome p {
    color:#9ca3af;
    font-size:1.1rem;
}

/* Top bar */
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# =========================
# TOP BAR
# =========================
col1, col2 = st.columns([8, 1])

with col1:
    st.markdown("## 🤖 AI Assistant")

with col2:
    if st.button("🗑️"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
# =========================
# WELCOME SCREEN
# =========================
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome">
        <h1>🤖 AI Assistant</h1>
        <p>Ask anything. Powered by LangGraph.</p>
    </div>
    """, unsafe_allow_html=True)
# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
# =========================
# CHAT INPUT
# =========================
prompt = st.chat_input("Message AI Assistant...")
if prompt:
    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    # Assistant message
    with st.chat_message("assistant"):

        try:

            with st.spinner("Thinking..."):

                result = workflow.invoke(
                    {
                        "messages": [
                            HumanMessage(content=prompt)
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )
                response = result["messages"][-1].content
            st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )
        except Exception as e:
            error_msg = f"❌ {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg
                }
            )