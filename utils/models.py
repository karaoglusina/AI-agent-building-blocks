"""
Pydantic models for job post data.
These models represent the structure of our job posting data.
"""

from pydantic import BaseModel


class SearchParams(BaseModel):
    """Parameters used when scraping the job."""
    keyword_used: str
    batch_scrape: bool
    batch_timestamp: str


class JobPost(BaseModel):
    """A single job posting from LinkedIn."""
    id: str
    title: str
    description: str
    companyName: str
    location: str
    publishedAt: str
    
    # Optional fields (may be empty)
    salary: str = ""
    jobUrl: str = ""
    companyUrl: str = ""
    postedTime: str = ""
    applicationsCount: str = ""
    descriptionHtml: str = ""
    contractType: str = ""
    experienceLevel: str = ""
    workType: str = ""
    sector: str = ""
    applyType: str = ""
    applyUrl: str = ""
    companyId: str = ""
    benefits: str = ""
    posterProfileUrl: str = ""
    posterFullName: str = ""
    search_params: SearchParams | None = None


if __name__ == "__main__":
    # Quick validation test
    from data_loader import load_sample_jobs
    
    jobs = load_sample_jobs(1)
    job = JobPost(**jobs[0])
    print(f"✓ Validated: {job.title} at {job.companyName}")
