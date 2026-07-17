# Data Setup

## Dataset Overview

The curriculum uses a dataset of **1,318 job postings** for practical examples. The dataset (`sample_job_data.json`) is about 14 MB and ships in this repo — a plain `git clone` gets it, with no Git LFS or download step. It contains LinkedIn job postings with the following fields:

- `id`: Unique job identifier
- `title`: Job title
- `description`: Full job description
- `companyName`: Company name
- `location`: Job location
- `publishedAt`: Publication date
- Plus optional fields: salary, sector, experienceLevel, workType, etc.

## Setup Options

### Option 1: Use the included data (default — nothing to do)

`sample_job_data.json` is already here after cloning. The scripts load it automatically via `utils/data_loader.py`. Skip to "Data Loading" below.

### Option 2: Use Your Own Data

The scripts work with any job posting dataset in JSON format. Replace `sample_job_data.json` in this directory with the following structure:

```json
[
  {
    "id": "job-1",
    "title": "Software Engineer",
    "description": "Job description here...",
    "companyName": "Acme Corp",
    "location": "San Francisco, CA",
    "publishedAt": "2024-01-15",
    "sector": "Technology",
    "experienceLevel": "Mid-Senior level"
  }
]
```

**Minimum required fields:** `id`, `title`, `description`, `companyName`, `location`

### Option 3: Generate a tiny synthetic set

If you want something smaller than the included 1,318 postings for a quick test:

   ```bash
   python -c "
   import json
   sample_jobs = [
       {
           'id': f'job-{i}',
           'title': 'Sample Job Title',
           'description': 'This is a sample job description for testing purposes.',
           'companyName': 'Sample Company',
           'location': 'Sample City',
           'publishedAt': '2024-01-01'
       }
       for i in range(10)
   ]
   with open('data/sample_job_data.json', 'w') as f:
       json.dump(sample_jobs, f, indent=2)
   "
   ```

### Option 4: Scrape Your Own Dataset

If you want to build your own dataset, consider scraping from:
- LinkedIn Jobs API (requires authentication)
- Public job boards with APIs (Indeed, GitHub Jobs, etc.)
- Company career pages

**Note:** Always respect website terms of service and rate limits when scraping.

## Data Loading

The curriculum provides utilities in `utils/data_loader.py`:

```python
from utils.data_loader import load_jobs, load_sample_jobs, get_job_by_id

# Load all jobs
all_jobs = load_jobs()

# Load first 10 jobs for quick testing
sample = load_sample_jobs(10)

# Get specific job
job = get_job_by_id("job-123")
```

## Validate Your Data

Test that your data works with the curriculum:

```bash
python -c "from utils.data_loader import load_sample_jobs; jobs = load_sample_jobs(3); print(f'✅ Loaded {len(jobs)} jobs successfully')"
```

## Privacy & Legal

- The included `sample_job_data.json` is a 1,318-posting excerpt of a larger scrape, committed to this repo. The full ~10k dataset is not published.
- **Do not commit real job data** with personally identifiable information
- If sharing your own data, ensure you have the right to do so
- Anonymize or use synthetic data for public repositories
