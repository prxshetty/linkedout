import streamlit as st
import requests
import time
import json
st.title("LinkedOut")
menu = ["Home", "Login"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Home":
    sub_menu = st.sidebar.selectbox(
        "Select Action",
        ["Scrape Job Description", "Upload Resume"]
    )

    if sub_menu == "Scrape Job Description":
        st.subheader("Job Description Analysis")
        url = st.text_input("Enter the Job Post URL")
        if st.button("Scrape URL"):
            if url:
                with st.spinner("Scraping Job Description...."):
                    response = requests.post("http://localhost:8000/scrape", params = {"url" : url})
                    time.sleep(1)
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Job Description Scraped Successfullyl")
                        keywords = data.get("keywords", [])

                        if keywords:
                            st.write("Keywords:", keywords)
                            for kw in keywords:
                                st.write(f"- {kw}")
                            
                            confirm = st.checkbox("Confirm Keywords?")
                            if confirm:
                                st.session_state.confirmed_keywords = keywords
                                st.success("Keywords confirmed!")
                        else:
                            st.warning("No keywords were extracted.")
                    else:
                        st.error("Error scraping job description")
                         
        st.markdown("---") 
        post_text = st.text_area("Or Enter the Job Post Description")
        num_keywords = st.number_input(
            "Number of keywords to extract",
            min_value=5,
            max_value=20,
            value=10,
            help="Select how many keywords you want to extract"
        )
        
        if st.button("Analyze Text"):
            if not post_text:
                st.error("Please enter some text to analyze")
            else:
                with st.spinner("Analyzing Job Post description..."):
                    response = requests.post(
                        "http://localhost:8000/analyze_ai", 
                        json={
                            "description": post_text,
                            "num_keywords": num_keywords
                        }
                    )
            if response.status_code == 200:
                data = response.json()
                if data["keywords"]:
                    st.success("Job Description analyzed Successfully!")
                    st.write(f"Top {num_keywords} Keywords:")
                    for i, keyword in enumerate(data["keywords"][:num_keywords], 1):
                        st.write(f"{i}. {keyword}")
                    
                    confirm = st.checkbox("Confirm Keywords")
                    if confirm:
                        st.session_state.confirmed_keywords = data['keywords'][:num_keywords]
                        st.success("Keywords confirmed and saved")
                else:
                    st.warning("No keywords were extracted. Please check the input text.")
            else:
                st.error(f"Error analyzing job description: {response.text}")

    elif sub_menu == "Upload Resume":
        st.subheader("Upload your Resume")
        upload_type = st.radio("Select upload type", ["Resume File", "Overleaf Code"])

        if upload_type == "Resume File":
            file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
            if st.button("Upload Resume"):
                if file is not None:
                    with st.spinner("Uploading Resume >>>"):
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        if file.type == 'application/pdf':
                            response = requests.post("http://localhost:8000/upload_pdf", files = file)
                        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                            response = requests.post("http://localhost:8000/upload_docx", files = file)
                        if response.status_code == 200:
                            st.success(response.json()['message'])
                        else:
                            st.write("Upload Failed: {response.text}")
                else:
                    st.write("Please upload a file first")

        elif upload_type == "Overleaf Code":
            st.markdown("""
            ### Instructions:
            1. Paste your complete LaTeX code (from \\documentclass to \\end{document})
            2. Enter keywords separated by new lines
            """)
            
            latex_code = st.text_area(
                "Enter your complete LaTeX Code",
                height=300,
                help="Paste your entire LaTeX document, including \\documentclass and \\end{document}"
            )
            
            keywords = st.text_area(
                "Enter keywords (one per line)",
                height=100,
                help="Enter each keyword on a new line"
            )
            
            optimization_level = st.slider(
                "Select Optimization Level",
                min_value = 1,
                max_value = 10,
                value = 5,
                help = "1: Minimal changes, 10: Maximum Optimization"
            )
            st.markdown("""
            **Optimization Level Guide:**
            - 1-3: Subtle changes, minimal keyword insertion
            - 4-7: Balanced optimization with natural keyword integration
            - 8-10: Aggressive optimization with maximum keyword presence
            """)

            if st.button("Optimize Resume"):
                if not latex_code or not keywords:
                    st.error("Please provide both LaTeX code and keywords")
                    st.stop()
                    
                if not latex_code.strip().endswith('\\end{document}'):
                    st.error("LaTeX code must end with \\end{document}")
                    st.stop()

                with st.spinner("Optimizing Resume..."):
                    response = requests.post(
                        "http://localhost:8000/optimize_sections",
                        json={
                            "latex_code": latex_code,
                            "keywords": keywords.split('\n'),
                            "optimization_level" : optimization_level
                        }
                    )
                    time.sleep(1)
                    
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        optimized_latex_result = data.get("optimized_latex")
                        st.success("Resume optimized successfully!")
                        st.text_area("Optimized LaTeX Code", optimized_latex_result, height=300)
                        if st.button("Download Optimized LaTeX"):
                            st.download_button(
                                "Download LaTeX file",
                                optimized_latex_result,
                                file_name="optimized_resume.tex",
                                mime="text/plain"
                            )
                    else:
                        st.error(f"Optimization failed: {data.get('message', 'Unknown error')}")
                else:
                    st.error(f"Request failed: {response.text}")


elif choice == "Login":
    st.subheader("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")
    if st.button("Register"):
        if username and password and email:
            user = {"username": username, "password": password, "email": email}
            with st.spinner("Registering..."):
                response = requests.post("http://localhost:8000/register", json=user)
                time.sleep(1)  
            if response.status_code == 200:
                st.success("You have been registered")
            else:
                st.error("Registration failed")
        else:
            st.error("Please fill in all fields")
        