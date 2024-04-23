import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="💬", layout="wide", page_title="Mafoya1er Goes Brrrrrrrr...")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🏎️")

st.subheader("Mafoya1er Chat App", divider="rainbow", anchor=False)

client = Groq(
    #api_key=st.secrets["GROQ_API_KEY"],
    api_key="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"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define model details
model_option = "mixtral-8x7b-32768"
max_tokens_range = 32768  # Fixed token size for the Mistral model

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Define the function to generate chat responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Sidebar for document upload
st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "md"])

if uploaded_file is not None:
    # Do something with the uploaded file
    st.sidebar.success('File uploaded successfully!')
    # Here you can add your code to process the uploaded file

# Chat input
if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens_range,
            stream=True,
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response}
        )
