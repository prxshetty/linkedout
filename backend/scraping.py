import requests
from bs4 import BeautifulSoup

def scrape_job_description(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"failed to load page {response.status_code}")
        soup = BeautifulSoup(response.text, 'html.parser')
        job_desc_div = soup.find('div', {'class': 'job-description'})
        if job_desc_div == None:
            raise Exception("Element not found")
        job_desc = job_desc_div.get_text()
        return job_desc
    except Exception as e:
        print(f"Error scraping job description: {e}")
    
url = 'https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4072874183'
job_d = scrape_job_description(url)
print(job_d)
