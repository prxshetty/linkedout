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
    post_text = st.text_area("Enter the Job Post Description")
    if st.button("Analyze"):
        if not post_text:
            st.error("Please enter some text to analyze")
            
        response = requests.post(
            "http://localhost:8000/analyze", 
            json={"description": post_text}
        )
        if response.status_code == 200:
            data = response.json()
            if data["keywords"]:
                st.write("Job Description analyzed Successfully!")
                st.write("Keywords:", data["keywords"])
            else:
                st.warning("No keywords were extracted. Please check the input text.")
        else:
            st.error(f"Error analyzing job description: {response.text}")

    if st.button("Analyze with AI"):
        if not post_text:
            st.error("Please enter some text to analyze")
        else:
            st.spinner("Analyzing with AI...")
            response = requests.post(
                "http://localhost:8000/analyze_ai", 
                json={"description": post_text}
            )
            if response.status_code == 200:
                data = response.json()
                print("api response", data)
                if data["keywords"]:
                    st.success("AI Analysis Complete!")
                    st.write("Required Technical Skills")
                    for skill in data["keywords"]:
                        st.write(f"{skill}")
                else:
                    st.warning("No keywords were extracted. Please check the input text.")
            else:
                st.error(f"Error in AI Analysis: {response.text}")

elif choice == "Upload Resume":
    st.subheader("Upload your Resume")
    upload_type = st.radio("Select upload type", ["Resume File", "Overleaf Code"])


    if upload_type == "Resume File":
        file = st.file_uploader("Choose a PDF or DOCX file", type = ["pdf", "docx"])
        if st.button("Upload Resume"):
            if file is not None:
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
                
            # if not latex_code.strip().startswith('\\documentclass'):
            #     st.error("LaTeX code must start with \\documentclass")
            #     st.stop()
                
            if not latex_code.strip().endswith('\\end{document}'):
                st.error("LaTeX code must end with \\end{document}")
                st.stop()

            response = requests.post(
                "http://localhost:8000/optimize_sections",
                json={
                    "latex_code": latex_code,
                    "keywords": keywords.split('\n'),
                    "optimization_level" : optimization_level
                }
            )
                
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
    