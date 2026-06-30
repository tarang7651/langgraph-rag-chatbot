import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import workflow, process_pdf

def get_new_thread_id():
    return str(uuid.uuid4())

def get_conversation(thread_id):
    print(thread_id)
    conversation = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    return conversation.values.get("messages", False)
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
# SESSION STATE
# =========================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = [st.session_state.thread_id]
if "messages" not in st.session_state:
    st.session_state.messages = []

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

with st.sidebar:
    # ── Brand Header ──
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🤖</div>
        <div class="brand-text">AI Assistant</div>
        <div class="brand-tagline">Powered by LangGraph</div>
    </div>
    """, unsafe_allow_html=True)

    # ── New Chat Button ──
    if st.button("✨  New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.thread_id = get_new_thread_id()
        st.session_state.thread_ids.append(st.session_state.thread_id)
        st.session_state.messages = []
        st.rerun()

    # ── Chat History Section ──
    st.markdown('<p class="sidebar-section-label">💬 Chat History</p>', unsafe_allow_html=True)

    for idx, thread_id in enumerate(st.session_state.thread_ids):
        is_active = (thread_id == st.session_state.thread_id)
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{'🟢' if is_active else '💬'}  Chat {idx + 1}"
            if st.button(label, key=f"btn_{thread_id}", use_container_width=True):
                st.session_state.thread_id = thread_id
                st.session_state.messages = []
                messages = get_conversation(thread_id)
                if messages:
                    for message in messages:
                        if isinstance(message, HumanMessage):
                            st.session_state.messages.append({"role": "user", "content": message.content})
                        elif isinstance(message, AIMessage):
                            st.session_state.messages.append({"role": "assistant", "content": message.content})
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{thread_id}", help="Delete this chat", use_container_width=True):
                st.session_state.thread_ids.remove(thread_id)
                if st.session_state.thread_id == thread_id:
                    st.session_state.messages = []
                    if st.session_state.thread_ids:
                        st.session_state.thread_id = st.session_state.thread_ids[-1]
                        messages = get_conversation(st.session_state.thread_id)
                        if messages:
                            for message in messages:
                                if isinstance(message, HumanMessage):
                                    st.session_state.messages.append({"role": "user", "content": message.content})
                                elif isinstance(message, AIMessage):
                                    st.session_state.messages.append({"role": "assistant", "content": message.content})
                    else:
                        st.session_state.thread_id = get_new_thread_id()
                        st.session_state.thread_ids.append(st.session_state.thread_id)
                st.rerun()

    # ── Document Upload Section ──
    st.markdown('<p class="sidebar-section-label">📄 Document Upload</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-hint">Upload a PDF to enable RAG-powered answers.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("⬆️  Process PDF", use_container_width=True, key="process_pdf_btn"):
            with st.spinner("Processing PDF..."):
                file_bytes = uploaded_file.getvalue()
                process_pdf(file_bytes)
                st.success("✅ PDF processed! Ask questions about it.")

    # ── Footer ──
    st.markdown("""
    <div class="sidebar-footer">
        Built with <span style="color:#ef4444">♥</span> using LangChain & Streamlit
    </div>
    """, unsafe_allow_html=True)

# =========================
# CUSTOM CSS
# =========================
import os
css_path = os.path.join(os.path.dirname(__file__), "styles", "style.css")
try:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    pass


# =========================
# TOP BAR
# =========================
col1, col2 = st.columns([8, 1])

with col1:
    st.markdown("## 🤖 AI Assistant")

with col2:
    if st.button("🗑️", help="Clear current conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = get_new_thread_id()
        st.session_state.thread_ids.append(st.session_state.thread_id)
        st.rerun()
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
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            response = ""

            for chunk in workflow.stream(
                {
                    "messages": [
                        HumanMessage(content=prompt)
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                },
                stream_mode="messages",
            ):
                message, metadata = chunk

                if hasattr(message, "content") and message.content:
                    response += message.content
                    placeholder.markdown(response)

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