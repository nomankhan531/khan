"""
CV Analyzer & Job Matching System
=================================
STAGE 1 - FOUNDATION

What this stage does
--------------------
1. Accepts a CV upload (PDF, DOCX, TXT) and validates it *before* parsing it.
2. Extracts the text entirely in memory (nothing is written to disk).
3. Detects, using transparent rules only (no AI, no network):
   contact details, CV sections, known skills, education lines, dated work entries.
4. Shows the results in a dashboard, clearly labelling:
       DETECTED  = copied/matched directly from the CV text
       ESTIMATE  = inferred by a heuristic (can be wrong)
5. Offers a JSON export of the detected facts.

What this stage does NOT do (yet)
---------------------------------
Job description analysis, match percentages, recommendations, PDF reports and
semantic NLP arrive in later stages, after approval of this one.

Run:        streamlit run app.py
Self-test:  python app.py --selftest

File layout (one file, organised in numbered sections):
    1 Configuration      2 Data models      3 Errors
    4 Validation + text extraction          5 Text normalisation
    6 Contact detection  7 Section detection
    8 Skill detection    9 Experience + education
   10 Observations      11 Analysis pipeline
   12 Streamlit UI      13 Self-tests       14 Entry point
"""

from __future__ import annotations

import functools
import hashlib
import html
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Limits protect a small machine (8 GB RAM) and reduce attack surface.

APP_TITLE = "CV Analyzer"
APP_VERSION = "0.1.0 - Stage 1 (Foundation)"

ALLOWED_EXTENSIONS = ("pdf", "docx", "txt")
MAX_FILE_BYTES = 5 * 1024 * 1024                 # 5 MB upload limit
MAX_PDF_PAGES = 15                               # a CV rarely exceeds this
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024   # guards against "zip bombs"
MIN_TEXT_CHARS = 80                              # below this we assume extraction failed

# =============================================================================
# 2. DATA MODELS
# =============================================================================
# Dataclasses give every result a named, typed shape. This makes the code
# self-documenting and easy to test, instead of passing around loose dicts.


@dataclass
class ContactInfo:
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    linkedin: list[str] = field(default_factory=list)
    github: list[str] = field(default_factory=list)
    other_links: list[str] = field(default_factory=list)


@dataclass
class SkillHit:
    """A skill found in the CV text. 'Found' means the words appear - nothing more."""

    name: str
    category: str
    mentions: int
    sections: list[str]
    in_skills_section: bool
    in_work_context: bool  # appears under experience or projects


@dataclass
class DateRange:
    raw: str
    start_month: int  # months since year 0 (year * 12 + month - 1)
    end_month: int    # exclusive end, same unit


@dataclass
class Observation:
    level: str    # "warning" or "info"
    message: str


@dataclass
class CVProfile:
    filename: str
    file_type: str
    word_count: int
    char_count: int
    text: str
    name_guess: str | None
    contact: ContactInfo
    sections: dict[str, str]
    skills: list[SkillHit]
    education_lines: list[str]
    experience_lines: list[str]
    date_ranges: list[DateRange]
    experience_months: int | None
    checklist: list[tuple[str, bool]]
    observations: list[Observation] = field(default_factory=list)


# =============================================================================
# 3. ERRORS
# =============================================================================
# Custom exceptions let the UI show a friendly message for expected problems
# while unexpected bugs are handled separately.


class CVAnalyzerError(Exception):
    """Base class for expected, user-facing errors."""


class FileValidationError(CVAnalyzerError):
    """The uploaded file failed a safety or format check."""


class TextExtractionError(CVAnalyzerError):
    """The file looked valid but its text could not be extracted."""


# =============================================================================
# 4. VALIDATION AND TEXT EXTRACTION
# =============================================================================


def validate_upload(filename: str, data: bytes) -> str:
    """Check size, extension and file signature. Return the lower-case extension.

    Why not trust the extension alone? Anyone can rename 'virus.exe' to 'cv.pdf'.
    Checking the first bytes ("magic bytes") confirms the content matches the type.
    """
    if not data:
        raise FileValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        raise FileValidationError(
            f"File is too large ({size_mb:.1f} MB). The limit is "
            f"{MAX_FILE_BYTES // (1024 * 1024)} MB."
        )

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            "Unsupported file type. Please upload a PDF, DOCX or TXT file."
        )

    if extension == "pdf" and b"%PDF-" not in data[:1024]:
        raise FileValidationError("This file is not a valid PDF document.")
    if extension == "docx":
        _validate_docx_container(data)
    if extension == "txt" and b"\x00" in data[:4096]:
        raise FileValidationError("This file looks binary, not plain text.")
    return extension


def _validate_docx_container(data: bytes) -> None:
    """A .docx is a ZIP archive. Verify it is a sane one before opening it."""
    if data[:4] != b"PK\x03\x04":
        raise FileValidationError("This file is not a valid DOCX document.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise FileValidationError("The DOCX file expands to an unsafe size.")
            if "word/document.xml" not in {entry.filename for entry in entries}:
                raise FileValidationError("This DOCX file has no readable document body.")
    except zipfile.BadZipFile as exc:
        raise FileValidationError("This DOCX file is corrupted.") from exc


def extract_text(extension: str, data: bytes) -> str:
    """Dispatch to the extractor for the validated file type."""
    if extension == "pdf":
        return _extract_pdf(data)
    if extension == "docx":
        return _extract_docx(data)
    return _extract_txt(data)


def _extract_pdf(data: bytes) -> str:
    """Read text from the first MAX_PDF_PAGES pages using pypdf (pure Python).

    Limitation: scanned PDFs are images, so there is no text to extract.
    OCR would need heavy dependencies, which we avoid on purpose.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise TextExtractionError(
            "The 'pypdf' package is missing. Run: pip install -r REQUIREMENTS.TXT"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")  # many "protected" PDFs use an empty password
            except Exception:
                unlocked = 0
            if not unlocked:
                raise TextExtractionError("This PDF is password-protected.")
        page_texts: list[str] = []
        for index in range(min(len(reader.pages), MAX_PDF_PAGES)):
            try:
                page_texts.append(reader.pages[index].extract_text() or "")
            except Exception:
                page_texts.append("")  # one bad page should not kill the whole CV
        return "\n".join(page_texts)
    except TextExtractionError:
        raise
    except Exception as exc:
        raise TextExtractionError("The PDF could not be read. It may be corrupted.") from exc


def _extract_docx(data: bytes) -> str:
    """Read paragraphs and tables in document order (so sections stay in order)."""
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise TextExtractionError(
            "The 'python-docx' package is missing. Run: pip install -r REQUIREMENTS.TXT"
        ) from exc

    try:
        document = Document(io.BytesIO(data))
        lines: list[str] = []
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                lines.append(Paragraph(child, document).text)
            elif child.tag.endswith("}tbl"):
                for row in Table(child, document).rows:
                    seen_cells = set()
                    cells: list[str] = []
                    for cell in row.cells:
                        if cell._tc in seen_cells:  # merged cells repeat; skip repeats
                            continue
                        seen_cells.add(cell._tc)
                        cells.append(cell.text.strip())
                    lines.append(" | ".join(text for text in cells if text))
        return "\n".join(lines)
    except Exception as exc:
        raise TextExtractionError("The DOCX file could not be read.") from exc


def _extract_txt(data: bytes) -> str:
    """Decode text trying common encodings. latin-1 never fails, so it is the last resort."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


# =============================================================================
# 5. TEXT NORMALISATION
# =============================================================================


def normalize_text(raw: str) -> str:
    """Clean extracted text so every later rule sees consistent input.

    Steps: unify Unicode forms, unify line endings, map odd bullet glyphs to one
    bullet, remove control characters, collapse spaces, limit blank lines.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\uf000-\uf8ff]", "•", text)         # private-use bullets (Symbol fonts)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)  # control characters
    text = re.sub(r"[^\S\n]+", " ", text)                 # spaces/tabs/nbsp -> one space
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# =============================================================================
# 6. CONTACT DETECTION
# =============================================================================

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# Candidate phone: starts with + ( or digit, contains digits/spaces/()-. , ends with a digit.
# Candidates are then filtered by digit count and "looks like a year range" rules.
PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)\+?\(?\d[\d ().\-]{7,}\d(?!\w)")
YEARS_ONLY_RE = re.compile(r"(?:(?:19|20)\d{2}[\s\-]*)+")
LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company)/[A-Za-z0-9\-_%/]+", re.I
)
GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_]+(?:/[A-Za-z0-9\-_.]+)?", re.I
)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]\"']+", re.I)


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicates (case-insensitive) while keeping the original order."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def extract_contact_info(text: str) -> ContactInfo:
    """Find emails, phones and links with regular expressions (facts, not guesses)."""
    emails = _dedupe(EMAIL_RE.findall(text))

    phones: list[str] = []
    seen_digits: set[str] = set()
    for match in PHONE_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if not 9 <= len(digits) <= 15:
            continue
        if YEARS_ONLY_RE.fullmatch(candidate):  # e.g. "2019 2021 2022"
            continue
        if digits not in seen_digits:
            seen_digits.add(digits)
            phones.append(candidate)

    def clean_links(pattern: re.Pattern[str]) -> list[str]:
        return _dedupe([m.group(0).rstrip(".,;:") for m in pattern.finditer(text)])

    linkedin = clean_links(LINKEDIN_RE)
    github = clean_links(GITHUB_RE)
    other = [
        url
        for url in clean_links(URL_RE)
        if "linkedin.com" not in url.lower() and "github.com" not in url.lower()
    ]
    return ContactInfo(emails, phones, linkedin, github, other)


NON_NAME_WORDS = {"curriculum", "vitae", "resume", "cv", "profile", "summary", "contact"}


def guess_name(text: str) -> str | None:
    """HEURISTIC (an estimate): the first short, capitalised 2-4 word line near the top.

    Names are impossible to detect with certainty by rules, so the UI labels this
    as an estimate and returns None rather than inventing a name.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()][:6]
    for line in lines:
        if len(line) > 40 or "@" in line or any(ch.isdigit() for ch in line):
            continue
        if _normalize_heading(line) in _heading_lookup():
            continue
        words = line.split()
        if not 2 <= len(words) <= 4:
            continue
        if any(word.lower().strip(".,") in NON_NAME_WORDS for word in words):
            continue
        # Unicode-aware: letters from any script, optional apostrophes/dots/hyphens.
        if all(re.fullmatch(r"[^\W\d_][\w'’.\-]*", word) for word in words) and not any(
            word[0].islower() for word in words
        ):
            return line
    return None


# =============================================================================
# 7. SECTION DETECTION
# =============================================================================

SECTION_ALIASES: dict[str, set[str]] = {
    "summary": {"summary", "professional summary", "profile", "professional profile",
                "about me", "about", "objective", "career objective", "personal statement"},
    "experience": {"experience", "work experience", "professional experience",
                   "employment history", "work history", "employment", "career history",
                   "internships", "internship experience"},
    "education": {"education", "academic background", "academic qualifications",
                  "education and training", "educational background"},
    "skills": {"skills", "technical skills", "key skills", "core skills",
               "core competencies", "skills and tools", "competencies", "technologies",
               "tools and technologies", "soft skills", "technical proficiency"},
    "projects": {"projects", "personal projects", "academic projects",
                 "selected projects", "key projects", "side projects"},
    "certifications": {"certifications", "certificates", "licenses and certifications",
                       "courses", "training", "courses and certifications"},
    "languages": {"languages", "language skills"},
    "awards": {"awards", "honors", "honours", "achievements", "awards and honors"},
    "interests": {"interests", "hobbies", "hobbies and interests"},
    "references": {"references"},
}


def _normalize_heading(line: str) -> str:
    """Lower-case, drop punctuation/digits, turn '&' into 'and' ("Skills & Tools:" -> "skills and tools")."""
    cleaned = re.sub(r"[^a-z& ]", "", line.lower()).replace("&", " and ")
    return re.sub(r"\s+", " ", cleaned).strip()


@functools.lru_cache(maxsize=1)
def _heading_lookup() -> dict[str, str]:
    """Map every normalised alias to its canonical section name (built once, then cached)."""
    lookup: dict[str, str] = {}
    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_heading(alias)] = canonical
    return lookup


def split_into_sections(text: str) -> dict[str, str]:
    """Split the CV into sections by recognising short heading lines.

    Text before the first heading is stored under 'header' (usually name + contacts).
    A line counts as a heading only if it is short and matches a known alias exactly,
    which avoids treating sentences like "Experience with Python" as headings.
    """
    lookup = _heading_lookup()
    buckets: dict[str, list[str]] = defaultdict(list)
    current = "header"
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) <= 40:
            canonical = lookup.get(_normalize_heading(stripped))
            if canonical:
                current = canonical
                buckets.setdefault(current, [])
                continue
        buckets[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in buckets.items() if "".join(lines).strip()}


# =============================================================================
# 8. SKILL DETECTION
# =============================================================================
# Design: a catalog of canonical skills -> alternative spellings ("aliases").
# The canonical name is added automatically as an alias.
# Single-letter names (C, R, Go) are deliberately omitted: they match ordinary
# text far too often. A later stage will handle them using context.

SKILL_CATALOG: dict[str, dict[str, tuple[str, ...]]] = {
    "Programming Languages": {
        "Python": ("python3",), "Java": (), "JavaScript": ("ecmascript", "es6"),
        "TypeScript": (), "C++": (), "C#": ("c sharp",), "PHP": (), "Ruby": (),
        "Swift": (), "Kotlin": (), "Rust": (), "Golang": (), "Scala": (), "MATLAB": (),
        "Dart": (), "Perl": (), "Bash": ("shell scripting",), "PowerShell": (), "VBA": (),
        "SQL": ("t-sql", "pl/sql"),
    },
    "Web & Frontend": {
        "HTML": ("html5",), "CSS": ("css3",), "React": ("react.js", "reactjs"),
        "Angular": ("angularjs",), "Vue.js": ("vue", "vuejs"), "Next.js": ("nextjs",),
        "Svelte": (), "Tailwind CSS": ("tailwind",), "Bootstrap": (), "jQuery": (),
        "Responsive Design": ("responsive web design",),
    },
    "Backend & Frameworks": {
        "Django": (), "Flask": (), "FastAPI": (), "Streamlit": (), "Node.js": ("nodejs", "node js"),
        "Express.js": ("expressjs",), "Spring Boot": ("spring",), "Laravel": (),
        ".NET": ("dotnet", ".net core", "asp.net"), "REST API": ("restful", "rest apis", "restful api"),
        "GraphQL": (), "Microservices": ("microservice",),
    },
    "Data & AI": {
        "Pandas": (), "NumPy": (), "Scikit-learn": ("sklearn", "scikit learn"),
        "TensorFlow": (), "PyTorch": (), "Keras": (), "OpenCV": (),
        "Machine Learning": ("ml models",), "Deep Learning": (),
        "NLP": ("natural language processing",), "spaCy": (), "NLTK": (),
        "Hugging Face": ("huggingface",), "LLM": ("large language models",),
        "Data Analysis": ("data analytics",), "Data Visualization": ("data visualisation",),
        "Statistics": ("statistical analysis",), "ETL": (), "Apache Spark": ("pyspark",),
        "Airflow": ("apache airflow",), "Power BI": ("powerbi",), "Tableau": (),
        "Microsoft Excel": ("ms excel", "advanced excel"), "Matplotlib": (), "Seaborn": (),
        "Plotly": (),
    },
    "Databases": {
        "MySQL": (), "PostgreSQL": ("postgres",), "MongoDB": (), "SQLite": (),
        "Redis": (), "SQL Server": ("mssql", "microsoft sql server"), "Oracle Database": ("oracle db",),
        "Firebase": (), "NoSQL": (), "Snowflake": (),
    },
    "Cloud & DevOps": {
        "AWS": ("amazon web services",), "Azure": ("microsoft azure",),
        "Google Cloud": ("gcp", "google cloud platform"), "Docker": (),
        "Kubernetes": ("k8s",), "Terraform": (), "Ansible": (), "Jenkins": (),
        "GitHub Actions": (), "CI/CD": ("cicd", "continuous integration"),
        "Linux": (), "Nginx": (), "Kafka": ("apache kafka",),
    },
    "Tools & Practices": {
        "Git": (), "GitHub": (), "GitLab": (), "Jira": (), "Agile": (), "Scrum": (),
        "Kanban": (), "Unit Testing": ("unit tests",), "Pytest": (), "Selenium": (),
        "Figma": (), "Postman": (), "Android": (), "iOS": (), "Flutter": (),
        "React Native": (),
    },
    "Soft Skills": {
        "Communication": ("communication skills",),
        "Teamwork": ("team work", "team player", "collaboration"),
        "Leadership": ("team leadership",),
        "Problem Solving": ("problem-solving",),
        "Critical Thinking": ("critical-thinking",),
        "Time Management": (), "Adaptability": ("adaptable",),
        "Creativity": ("creative thinking",), "Project Management": (),
        "Stakeholder Management": (), "Negotiation": (),
        "Presentation Skills": ("public speaking",),
        "Attention to Detail": ("detail-oriented", "detail oriented"),
    },
}

# Words that are ambiguous in running text ("excel at ...") but clear inside a
# dedicated Skills section. They are matched ONLY there.
CONTEXT_ONLY_ALIASES: dict[str, tuple[str, ...]] = {"Microsoft Excel": ("excel",)}


@functools.lru_cache(maxsize=1)
def _skill_index() -> tuple[dict[str, str], dict[str, str]]:
    """Build (alias -> canonical skill) and (canonical skill -> category) lookups once."""
    alias_to_skill: dict[str, str] = {}
    skill_to_category: dict[str, str] = {}
    for category, skills in SKILL_CATALOG.items():
        for canonical, aliases in skills.items():
            skill_to_category[canonical] = category
            for alias in (canonical, *aliases):
                alias_to_skill[alias.lower()] = canonical
    return alias_to_skill, skill_to_category


def _build_matcher(aliases: list[str]) -> re.Pattern[str]:
    """Compile ONE regex that matches any alias as a whole 'token'.

    Boundaries use look-around instead of \\b because \\b breaks on names such as
    'C++', 'C#' or '.NET'. Longest aliases come first so 'react native' beats 'react'.
    The look-behind/ahead also stop 'java' matching inside 'javascript' and
    'sql' matching inside 'mysql'.
    """
    ordered = sorted(set(aliases), key=len, reverse=True)
    body = "|".join(re.escape(alias) for alias in ordered)
    return re.compile(rf"(?<![A-Za-z0-9+#])(?:{body})(?![A-Za-z0-9+#])", re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def _main_matcher() -> re.Pattern[str]:
    return _build_matcher(list(_skill_index()[0]))


@functools.lru_cache(maxsize=1)
def _context_matcher() -> tuple[re.Pattern[str], dict[str, str]]:
    mapping = {alias: skill for skill, aliases in CONTEXT_ONLY_ALIASES.items() for alias in aliases}
    return _build_matcher(list(mapping)), mapping


def _count_skill_mentions(text: str, include_context_only: bool) -> Counter[str]:
    """Count how often each canonical skill appears in a piece of text."""
    alias_to_skill = _skill_index()[0]
    counts: Counter[str] = Counter()
    for match in _main_matcher().finditer(text):
        counts[alias_to_skill[match.group(0).lower()]] += 1
    if include_context_only:
        matcher, mapping = _context_matcher()
        for match in matcher.finditer(text):
            counts[mapping[match.group(0).lower()]] += 1
    return counts


def extract_skills(sections: dict[str, str]) -> list[SkillHit]:
    """Detect skills section by section so we can record WHERE each one appears.

    A skill found in experience/projects text is 'mentioned in a work context'.
    That is a fact about the words on the page - not proof the person is skilled.
    """
    skill_to_category = _skill_index()[1]
    mentions: dict[str, Counter[str]] = defaultdict(Counter)  # skill -> {section: count}
    for section_name, body in sections.items():
        counts = _count_skill_mentions(body, include_context_only=(section_name == "skills"))
        for skill, count in counts.items():
            mentions[skill][section_name] += count

    category_order = list(SKILL_CATALOG)
    hits = [
        SkillHit(
            name=skill,
            category=skill_to_category[skill],
            mentions=sum(per_section.values()),
            sections=sorted(per_section),
            in_skills_section="skills" in per_section,
            in_work_context=bool({"experience", "projects"} & set(per_section)),
        )
        for skill, per_section in mentions.items()
    ]
    hits.sort(key=lambda h: (category_order.index(h.category), -h.mentions, h.name.lower()))
    return hits


# =============================================================================
# 9. EXPERIENCE AND EDUCATION
# =============================================================================

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
MONTH_NUMBERS = {name: i for i, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# Matches ranges such as "Jan 2019 - Dec 2021", "2018 – 2020", "March 2020 to Present".
DATE_RANGE_RE = re.compile(
    r"(?P<sm>\b" + _MONTH + r"\.?\s+)?"
    r"\b(?P<sy>" + _YEAR + r")\b"
    r"\s*(?:-|–|—|to)\s*"
    r"(?:(?P<present>present|current|now|ongoing|today)\b"
    r"|(?P<em>\b" + _MONTH + r"\.?\s+)?\b(?P<ey>" + _YEAR + r")\b)",
    re.IGNORECASE,
)

LONG_DEGREE_RE = re.compile(
    r"\b(?:bachelors?|masters?|doctorate|diploma|associate degree|ph\.?d)\b", re.IGNORECASE
)
ABBREV_DEGREE_RE = re.compile(  # case-sensitive on purpose: "MS" (degree) vs "ms" (milliseconds)
    r"\b(?:B\.?Sc|M\.?Sc|B\.?S|M\.?S|B\.?A|M\.?A|BBA|MBA|MBBS|B\.?Eng|M\.?Eng|B\.?Tech|M\.?Tech"
    r"|BSCS|BSIT|BCS|MCS|PhD)\b"
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _month_number(token: str | None) -> int | None:
    if not token:
        return None
    return MONTH_NUMBERS[token.strip(" .").lower()[:3]]


def parse_date_ranges(text: str, today: date | None = None) -> list[DateRange]:
    """Find date ranges and convert them to month indices.

    Assumptions (documented because they make the result an ESTIMATE):
      * a missing month is treated as June (the midpoint), which is unbiased on average;
      * 'Present' ends at the current month;
      * implausible ranges (end before start, > 50 years, starting in the future) are dropped.
    """
    today = today or date.today()
    today_index = _month_index(today.year, today.month)
    ranges: list[DateRange] = []
    for match in DATE_RANGE_RE.finditer(text):
        start_year = int(match.group("sy"))
        start = _month_index(start_year, _month_number(match.group("sm")) or 6)
        if match.group("present"):
            end = today_index + 1
        else:
            end_month = _month_number(match.group("em"))
            end_year = int(match.group("ey"))
            end = _month_index(end_year, end_month) + 1 if end_month else _month_index(end_year, 6)
        end = min(end, today_index + 1)
        if start_year < 1970 or end <= start or end - start > 12 * 50 or start > today_index:
            continue
        ranges.append(DateRange(match.group(0).strip(), start, end))
    return ranges


def total_months(ranges: list[DateRange]) -> int:
    """Union of all ranges in months. Overlapping jobs are NOT double counted.

    Classic 'merge intervals' algorithm: sort by start, then extend the current
    interval while the next one overlaps it; otherwise close it and start a new one.
    """
    if not ranges:
        return 0
    intervals = sorted((r.start_month, r.end_month) for r in ranges)
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + (current_end - current_start)


def extract_education_lines(education_text: str) -> list[str]:
    """Return the original lines in the Education section that mention a degree keyword."""
    return [
        line for line in education_text.split("\n")
        if line.strip() and (LONG_DEGREE_RE.search(line) or ABBREV_DEGREE_RE.search(line))
    ]


def extract_experience_lines(experience_text: str) -> list[str]:
    """Return the original lines in the Experience section that contain a date range."""
    return [line for line in experience_text.split("\n") if DATE_RANGE_RE.search(line)]


# =============================================================================
# 10. OBSERVATIONS
# =============================================================================


def build_observations(profile: CVProfile) -> list[Observation]:
    """Plain, checkable observations about the extracted data.

    These describe what the PARSER found. 'Section not found' can mean the section
    is missing OR that it has an unusual title - the wording says so honestly.
    """
    notes: list[Observation] = []
    if profile.word_count < 150:
        notes.append(Observation(
            "warning",
            f"Only {profile.word_count} words were extracted. The file may be mostly images, "
            "or its layout may have confused text extraction."))
    elif profile.word_count > 1200:
        notes.append(Observation(
            "info",
            f"The CV is long ({profile.word_count} words). Check that every part is relevant."))

    if not profile.contact.emails:
        notes.append(Observation("warning", "No email address was detected in the text."))
    if not profile.contact.phones:
        notes.append(Observation("info", "No phone number was detected in the text."))

    for key, label in (("experience", "Experience"), ("education", "Education"), ("skills", "Skills")):
        if key not in profile.sections:
            notes.append(Observation(
                "warning",
                f"No '{label}' heading was recognised. Either the section is missing or it uses "
                "an unusual title (common titles like 'Work Experience' are recognised)."))

    if not profile.skills:
        notes.append(Observation("warning", "No skills from the built-in catalog were detected."))
    if "experience" in profile.sections and not profile.date_ranges:
        notes.append(Observation(
            "info", "The Experience section has no recognisable date ranges, so experience length "
                    "could not be estimated."))
    return notes


# =============================================================================
# 11. ANALYSIS PIPELINE
# =============================================================================


def analyze_cv(filename: str, data: bytes, today: date | None = None) -> CVProfile:
    """Run the whole Stage 1 pipeline: validate -> extract -> normalise -> detect.

    Everything happens in memory; the uploaded bytes are never written to disk.
    """
    extension = validate_upload(filename, data)
    text = normalize_text(extract_text(extension, data))
    if len(text) < MIN_TEXT_CHARS:
        hint = " Scanned (image-only) PDFs need OCR, which this lightweight version excludes." \
            if extension == "pdf" else ""
        raise TextExtractionError("Almost no readable text was found in the file." + hint)

    sections = split_into_sections(text)
    contact = extract_contact_info(text)
    skills = extract_skills(sections)

    experience_text = sections.get("experience", "")
    date_ranges = parse_date_ranges(experience_text, today)
    experience_months = total_months(date_ranges) if date_ranges else None

    checklist = [
        ("Email address", bool(contact.emails)),
        ("Phone number", bool(contact.phones)),
        ("Summary / profile section", "summary" in sections),
        ("Experience section", "experience" in sections),
        ("Education section", "education" in sections),
        ("Skills section", "skills" in sections),
        ("Dated work entries", bool(date_ranges)),
    ]

    profile = CVProfile(
        filename=filename.replace("\\", "/").split("/")[-1],
        file_type=extension,
        word_count=len(text.split()),
        char_count=len(text),
        text=text,
        name_guess=guess_name(text),
        contact=contact,
        sections=sections,
        skills=skills,
        education_lines=extract_education_lines(sections.get("education", "")),
        experience_lines=extract_experience_lines(experience_text),
        date_ranges=date_ranges,
        experience_months=experience_months,
        checklist=checklist,
    )
    profile.observations = build_observations(profile)
    return profile


def profile_to_export(profile: CVProfile) -> dict:
    """Export detected facts only. The full CV text is deliberately NOT included."""
    data = asdict(profile)
    data.pop("text", None)
    data["sections"] = sorted(k for k in profile.sections if k != "header")
    data["_note"] = ("Detected facts and heuristic estimates from rule-based parsing. "
                     "Keyword presence is not a measure of ability.")
    return data


def format_experience(months: int | None) -> str:
    if months is None:
        return "Not estimated"
    years, rest = divmod(months, 12)
    return f"~{years}y {rest}m"


# =============================================================================
# 12. STREAMLIT USER INTERFACE
# =============================================================================

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1180px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.1rem; color: #fff;
       background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);}
.hero-title {font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2;}
.hero-sub {margin-top: .35rem; opacity: .93; font-size: 1.02rem;}
.chip {display: inline-block; padding: .22rem .7rem; margin: .18rem .25rem .18rem 0;
       border-radius: 999px; font-size: .86rem; background: rgba(79,70,229,.12);
       border: 1px solid rgba(79,70,229,.35);}
.tag {display: inline-block; padding: .08rem .5rem; border-radius: 6px; font-size: .74rem;
      font-weight: 600; letter-spacing: .03em; text-transform: uppercase; margin-right: .35rem;}
.tag-fact {background: rgba(16,185,129,.16); color: #059669;}
.tag-estimate {background: rgba(245,158,11,.18); color: #b45309;}
.tag-note {background: rgba(100,116,139,.18); color: #475569;}
</style>
"""


def tag(kind: str, label: str) -> str:
    """Small coloured label. 'fact' = detected, 'estimate' = heuristic."""
    return f'<span class="tag tag-{kind}">{html.escape(label)}</span>'


def render_hero() -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-title">📄 {APP_TITLE}</div>'
        '<div class="hero-sub">Upload a CV to see what can be reliably detected in it. '
        "Transparent rules, no data stored, no paid APIs.</div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Sidebar: uploader, clear button and privacy notes. Returns the uploaded file or None."""
    with st.sidebar:
        st.header("Upload CV")
        nonce = st.session_state.get("uploader_nonce", 0)
        uploaded = st.file_uploader(
            "PDF, DOCX or TXT (max 5 MB)", type=list(ALLOWED_EXTENSIONS), key=f"uploader_{nonce}"
        )
        if st.button("🗑️ Clear CV from memory"):
            for key in ("cv_digest", "cv_profile"):
                st.session_state.pop(key, None)
            st.session_state["uploader_nonce"] = nonce + 1  # new key => empty uploader
            st.rerun()
        st.divider()
        st.subheader("🔒 Privacy")
        st.caption(
            "Your file is processed in memory only and never written to disk or sent to any "
            "external service. It disappears when you clear it or close the tab."
        )
        st.caption(f"Version: {APP_VERSION}")
    return uploaded


def get_profile(uploaded) -> CVProfile | None:
    """Analyse the upload once per distinct file (SHA-256 digest), not on every UI rerun.

    Streamlit re-executes this script after each click. We keep the result in
    st.session_state (server memory for this browser session only). We do NOT use
    st.cache_data, because that would keep CV contents cached across users/sessions.
    """
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("cv_digest") == digest:
        return st.session_state.get("cv_profile")
    try:
        profile = analyze_cv(uploaded.name, data)
    except CVAnalyzerError as exc:
        st.error(f"⚠️ {exc}")
        return None
    except Exception:  # unexpected bug: show a generic message, never CV content
        st.error("⚠️ Unexpected error while analysing this file. Please try another file.")
        return None
    st.session_state["cv_digest"] = digest
    st.session_state["cv_profile"] = profile
    return profile


def render_empty_state() -> None:
    st.info("👈 Upload a CV in the sidebar to begin.")
    col1, col2, col3 = st.columns(3)
    with col1.container(border=True):
        st.markdown("**1. Upload**")
        st.caption("PDF, DOCX or TXT. Validated for size, type and safety first.")
    with col2.container(border=True):
        st.markdown("**2. Detect**")
        st.caption("Contact details, sections, skills, education and dated work entries.")
    with col3.container(border=True):
        st.markdown("**3. Review**")
        st.caption("Every result is labelled as detected fact or estimate.")


def render_overview(profile: CVProfile) -> None:
    sections_found = [name for name in profile.sections if name != "header"]
    cols = st.columns(4)
    cols[0].metric("Words extracted", f"{profile.word_count:,}")
    cols[1].metric("Sections recognised", len(sections_found))
    cols[2].metric("Skills detected", len(profile.skills))
    cols[3].metric("Experience (estimate)", format_experience(profile.experience_months))
    st.markdown(
        tag("fact", "Detected") + "counted directly from your text. "
        + tag("estimate", "Estimate") + "inferred from date ranges; can be wrong.",
        unsafe_allow_html=True,
    )

    st.subheader("Structure checklist")
    found = sum(ok for _, ok in profile.checklist)
    total = len(profile.checklist)
    st.progress(found / total, text=f"{found} of {total} expected items found")
    left, right = st.columns(2)
    for index, (label, ok) in enumerate(profile.checklist):
        (left if index % 2 == 0 else right).markdown(f"{'✅' if ok else '⬜'} {label}")
    st.caption("This checks whether the parser FOUND items. It is not a quality score for the CV.")

    st.subheader("Observations")
    if not profile.observations:
        st.success("No parsing concerns were raised.")
    for note in profile.observations:
        (st.warning if note.level == "warning" else st.info)(note.message)

    st.subheader("Skills by category")
    if profile.skills:
        counts = Counter(hit.category for hit in profile.skills)
        frame = pd.DataFrame({"Category": list(counts), "Skills detected": list(counts.values())})
        st.bar_chart(frame.set_index("Category"))
    else:
        st.info("No skills detected, so there is nothing to chart.")

    st.download_button(
        "⬇️ Download detected facts (JSON)",
        data=json.dumps(profile_to_export(profile), indent=2, default=str),
        file_name="cv_detected_facts.json",
        mime="application/json",
    )
    st.caption("The export excludes the full CV text but may contain your contact details.")


def render_profile_tab(profile: CVProfile) -> None:
    # st.text() shows text literally (no Markdown/HTML), so CV content cannot inject formatting.
    st.markdown(tag("estimate", "Estimate") + "Name guess (first short capitalised line)",
                unsafe_allow_html=True)
    st.text(profile.name_guess or "Not detected")

    st.markdown(tag("fact", "Detected") + "Contact details", unsafe_allow_html=True)
    contact = profile.contact
    rows = [("Email", contact.emails), ("Phone", contact.phones), ("LinkedIn", contact.linkedin),
            ("GitHub", contact.github), ("Other links", contact.other_links)]
    for label, values in rows:
        with st.container(border=True):
            st.caption(label)
            st.text("\n".join(values) if values else "Not detected")
    st.caption("Links are shown as plain text and are not clickable, for safety.")


def render_skills_tab(profile: CVProfile) -> None:
    if not profile.skills:
        st.info("No skills from the catalog were detected.")
        return
    st.markdown(tag("fact", "Detected") + "Skills whose names appear in the CV text.",
                unsafe_allow_html=True)
    st.warning(
        "A skill appearing in a CV is not evidence of ability, and a missing keyword does not "
        "mean the person lacks the skill. Treat this as a text inventory, not an assessment."
    )
    by_category: dict[str, list[SkillHit]] = defaultdict(list)
    for hit in profile.skills:
        by_category[hit.category].append(hit)
    for category, hits in by_category.items():
        st.markdown(f"**{html.escape(category)}**")
        chips = "".join(f'<span class="chip">{html.escape(h.name)}</span>' for h in hits)
        st.markdown(chips, unsafe_allow_html=True)

    st.subheader("Evidence table")
    table = pd.DataFrame([{
        "Skill": h.name, "Category": h.category, "Mentions": h.mentions,
        "Listed in Skills section": "Yes" if h.in_skills_section else "No",
        "Mentioned in Experience/Projects": "Yes" if h.in_work_context else "No",
    } for h in profile.skills])
    st.dataframe(table, hide_index=True)


def render_experience_tab(profile: CVProfile) -> None:
    st.markdown(tag("estimate", "Estimate") + "Total experience from dated entries",
                unsafe_allow_html=True)
    st.metric("Estimated experience", format_experience(profile.experience_months))
    st.caption("Overlapping roles are merged, missing months are assumed to be June, and "
               "'Present' means today. Gaps, part-time work and unlabeled dates are not understood.")

    st.markdown(tag("fact", "Detected") + "Date ranges found in the Experience section",
                unsafe_allow_html=True)
    if profile.date_ranges:
        for item in profile.date_ranges:
            st.text(item.raw)
    else:
        st.info("No date ranges found.")

    st.markdown(tag("fact", "Detected") + "Experience lines containing dates", unsafe_allow_html=True)
    st.text("\n".join(profile.experience_lines) if profile.experience_lines else "None found")

    st.markdown(tag("fact", "Detected") + "Education lines mentioning a degree keyword",
                unsafe_allow_html=True)
    st.text("\n".join(profile.education_lines) if profile.education_lines else "None found")


def render_text_tab(profile: CVProfile) -> None:
    st.caption("Use this to check what the extractor actually read. If text is garbled or "
               "missing here, every later result will suffer.")
    options = ["Full text"] + [name for name in profile.sections]
    choice = st.selectbox("View", options)
    body = profile.text if choice == "Full text" else profile.sections[choice]
    st.text_area("Extracted text", value=body, height=420, disabled=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()
    uploaded = render_sidebar()
    if uploaded is None:
        render_empty_state()
        return

    with st.spinner("Analysing CV..."):
        profile = get_profile(uploaded)
    if profile is None:
        return

    st.markdown(f"**File:** `{profile.filename}` · {profile.file_type.upper()} · "
                f"{profile.char_count:,} characters")
    tabs = st.tabs(["Overview", "Profile & Contact", "Skills", "Experience & Education", "Text Preview"])
    with tabs[0]:
        render_overview(profile)
    with tabs[1]:
        render_profile_tab(profile)
    with tabs[2]:
        render_skills_tab(profile)
    with tabs[3]:
        render_experience_tab(profile)
    with tabs[4]:
        render_text_tab(profile)


# =============================================================================
# 13. SELF-TESTS  (run with: python app.py --selftest)
# =============================================================================

SAMPLE_CV = """Jane Doe
jane.doe@example.com | +1 (415) 555-0134 | linkedin.com/in/janedoe | github.com/janedoe

Summary
Analyst who likes to excel at communication and teamwork.

Skills
Python, SQL, Excel, Tableau, C++, C#, Node.js, Power BI

Experience
Data Analyst, Acme Corp  Jan 2019 – Dec 2021
- Built dashboards in Power BI using SQL and Python.
Senior Analyst, Beta Ltd  2022 - Present
- Led a team of 4 analysts; strong problem solving.

Education
BSc Computer Science, State University 2014 - 2018
"""


def run_self_tests() -> int:
    """Small, fast regression tests. Returns the number of failures."""
    failures = 0

    def check(name: str, condition: bool) -> None:
        nonlocal failures
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")
        failures += 0 if condition else 1

    def raises(func, *args) -> bool:
        try:
            func(*args)
        except CVAnalyzerError:
            return True
        return False

    today = date(2026, 11, 1)
    profile = analyze_cv("cv.txt", SAMPLE_CV.encode("utf-8"), today)
    names = {hit.name: hit for hit in profile.skills}

    check("email detected", profile.contact.emails == ["jane.doe@example.com"])
    check("exactly one phone, year ranges ignored",
          len(profile.contact.phones) == 1 and "555" in profile.contact.phones[0])
    check("linkedin and github detected", bool(profile.contact.linkedin and profile.contact.github))
    check("name guessed", profile.name_guess == "Jane Doe")
    check("core sections found",
          {"summary", "skills", "experience", "education"} <= set(profile.sections))
    check("special-character skills (C++, C#, Node.js)",
          {"C++", "C#", "Node.js", "Power BI"} <= set(names))
    check("'Excel' counted only inside Skills section",
          "Microsoft Excel" in names and names["Microsoft Excel"].sections == ["skills"])
    check("Python seen in skills AND work context",
          names["Python"].in_skills_section and names["Python"].in_work_context)
    check("soft skills detected", {"Communication", "Teamwork", "Problem Solving"} <= set(names))

    mixed = _count_skill_mentions("I use JavaScript and MySQL", False)
    check("no false 'Java' inside JavaScript, no 'SQL' inside MySQL",
          set(mixed) == {"JavaScript", "MySQL"})
    check(".NET detected", ".NET" in _count_skill_mentions("Built apps with C++, C# and .NET", False))

    check("month-precise range = 36 months",
          total_months(parse_date_ranges("Jan 2019 – Dec 2021", today)) == 36)
    check("overlapping ranges are merged (not double counted)",
          total_months(parse_date_ranges("2019 - 2021 and 2020 - 2022", today)) == 36)
    check("future start dates are ignored", parse_date_ranges("2040 - 2041", today) == [])
    check("experience estimate >= 36 months",
          profile.experience_months is not None and profile.experience_months >= 36)
    check("education line detected", any("BSc" in line for line in profile.education_lines))
    check("export excludes full text", "text" not in profile_to_export(profile))

    check("empty file rejected", raises(validate_upload, "cv.pdf", b""))
    check("wrong extension rejected", raises(validate_upload, "cv.exe", b"data"))
    check("fake PDF rejected", raises(validate_upload, "cv.pdf", b"hello world"))
    check("fake DOCX rejected", raises(validate_upload, "cv.docx", b"hello world"))
    check("oversized file rejected",
          raises(validate_upload, "cv.txt", b"a" * (MAX_FILE_BYTES + 1)))
    check("almost-empty text rejected", raises(analyze_cv, "cv.txt", b"too short"))

    print(f"\n{failures} failure(s)." if failures else "\nAll self-tests passed.")
    return failures


# =============================================================================
# 14. ENTRY POINT
# =============================================================================
# `streamlit run app.py` executes this file with __name__ == "__main__", so we
# use a command-line flag to choose between the app and the self-tests.

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if run_self_tests() else 0)
    main()
