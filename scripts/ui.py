import streamlit as st
import requests

st.title("💬 RAG Chat")

user_input = st.text_input("Ask a question:")

if st.button("Send"):
    if user_input:
        response = requests.post(
            "http://127.0.0.1:8000/query",
            json={"question": user_input}
        )

        answer = response.json().get("answer", "Error")

        st.write("### Answer:")
        st.write(answer)