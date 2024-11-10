import streamlit as st
import requests
st.title("LinkedOut")
menu = ["Scrape Job Description", "Upload Resume", "User Login"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Scrape Job Description":
    st.subheader("Scrape Job Description")
    url = st.text_input("Enter the Job Post URL")
    if st.button("Scrape"):
        response = requests.post("http://localhost:8000/scrape", params = {"url" : url})
        if response.status_code == 200:
            data = response.json()
            st.write("Job Description Scraped Successfullyl")
            st.write("Keywords:", data["keywords"])
        else:
            st.write("Error scraping job description")
elif choice == "Upload Resume":
    st.subheader("Upload your Resume")
    file = st.file_uploader("Choose a PDF or DOCX file", type = ["pdf", "docx"])
    if st.button("Upload"):
        if file.type == 'application/pdf':
            response = requests.post("http://localhost:8000/upload_pdf", files = {"file": file})
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            response = requests.post("http://localhost:8000/upload_docx", files = {"file": file})
        if response.status_code == 200:
            st.success(response.json()['message'])
        else:
            st.write("Please upload a valid PDF or DOCX file")
elif choice == "User Login":
    st.subheader("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type = "password")
    email = st.text_input("Email")
    if st.button("Register"):
        user = {"username": username, "password": password, "email": email}
        response = requests.post("http://localhost:8000/register", json = user)
        if response.status_code == 200:
            st.success("YOu have been registered")
        else:
            st.error("Registration failed")
    