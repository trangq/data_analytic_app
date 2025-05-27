import streamlit as st
import requests
from bs4 import BeautifulSoup


# Constants
MODEL = "llama3.2"

# A class to represent a Webpage
class Website:
    """
    A utility class to represent a Website that we have scraped
    """
    url: str
    title: str
    text: str

    def __init__(self, url):
        """
        Create this Website object from the given URL using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# System prompt for the model
system_prompt = (
    "You are an assistant that analyzes the contents of a website "
    "and provides a short summary, ignoring text that might be navigation related. "
    "Respond in markdown in vietnamese."
)

# Function to create the user prompt
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}. "
    user_prompt += (
        "The contents of this website are as follows; "
        "please provide a short summary of this website in markdown. "
        "If it includes news or announcements, then summarize these too.\n\n"
    )
    user_prompt += website.text
    return user_prompt

# Function to create messages for the model
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]

def query_ollama(messages, model="llama3"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False  # ✅ thêm dòng này
        }
    )
    return response.json()


# Function to summarize a URL
def summarize(url):
    try:
        website = Website(url)
        messages = messages_for(website)
        response = query_ollama(messages, model=MODEL)
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"



# Streamlit app
st.title("Website Summary App by Trangqh")
st.markdown(
    "Điền link 1 website public để tóm tắt nội dung chính"
)

# Input URL
url = st.text_input("Enter a URL:", "")

if st.button("Summarize"):
    if url:
        with st.spinner("Summarizing..."):
            summary = summarize(url)
        st.markdown(summary)
    else:
        st.warning("Please enter a valid URL.")