
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="SmartAgentic Travel",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ SmartAgentic Travel")
st.write("Ask a question and get travel information from the travel knowledge base.")

question = st.text_area(
    "Your travel question",
    placeholder="Example: What are the best places to visit in Hyderabad?",
    height=100,
)

if st.button("Get travel recommendations", type="primary"):
    if not question.strip():
        st.warning("Please enter a travel question.")
    else:
        try:
            with st.spinner("Finding relevant travel information..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question.strip()},
                    timeout=120,
                )

            response.raise_for_status()
            result = response.json()

            st.subheader("Answer")
            st.write(result.get("answer", "No answer was returned."))

            sources = result.get("sources", [])
            if sources:
                st.subheader("Sources")
                for source in sources:
                    st.write(f"- {source}")

        except requests.exceptions.Timeout:
            st.error("The request took too long. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. "
                "Check that the FastAPI server is running."
            )
        except requests.exceptions.HTTPError:
            st.error(
                f"The API returned an error ({response.status_code}). "
                "Check the FastAPI terminal for details."
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
