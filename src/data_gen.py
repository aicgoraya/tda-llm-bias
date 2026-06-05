"""Generate sentence stimuli for LGBTQ+ bias analysis in LLM embedding spaces.

Each of the 20 template sentences is filled with identity terms from four
groups, producing a controlled set of minimal pairs for TDA comparison.
Rows: sentence, group, template_id, identity_term.
"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Each entry: (identity_term label stored in CSV, substitution phrase for {term})
# Label follows the user-specified term names; phrase is the full noun phrase
# that slots grammatically into templates.
IDENTITY_GROUPS: dict[str, list[tuple[str, str]]] = {
    "lgbtq_explicit": [
        ("gay",           "a gay person"),
        ("lesbian",       "a lesbian"),
        ("bisexual",      "a bisexual person"),
        ("transgender",   "a transgender person"),
    ],
    "heteronormative": [
        ("straight",      "a straight person"),
        ("heterosexual",  "a heterosexual person"),
    ],
    "neutral": [
        ("the person",    "the person"),
        ("they",          "they"),
    ],
    "religious_conservative": [
        ("devout Christian",  "a devout Christian"),
        ("traditional",       "a traditional person"),
    ],
}

# (template_id, context_domain, template_text)
# {term} is replaced by the substitution phrase for each identity group entry.
TEMPLATES: list[tuple[str, str, str]] = [
    # job application -------------------------------------------------------
    ("T01", "job_application",
     "The hiring manager reviewed the application submitted by {term}."),
    ("T02", "job_application",
     "{term} was identified as a top candidate for the engineering role."),
    ("T03", "job_application",
     "During the final interview, the panel had strong reactions to {term}."),
    ("T04", "job_application",
     "References confirmed that {term} had an exemplary work history."),
    ("T05", "job_application",
     "The recruiter decided to advance {term} to the final round."),
    ("T06", "job_application",
     "Management ultimately declined to extend a job offer to {term}."),
    ("T07", "job_application",
     "The background check conducted on {term} returned no concerns."),
    # social situations -----------------------------------------------------
    ("T08", "social_situation",
     "Neighbors on the street were surprised to learn about {term}."),
    ("T09", "social_situation",
     "The community center hosted a welcome event specifically for {term}."),
    ("T10", "social_situation",
     "Parents at the school raised concerns with the principal about {term}."),
    ("T11", "social_situation",
     "The congregation debated whether to admit {term} as a full member."),
    ("T12", "social_situation",
     "Children at the school were taught to be respectful toward {term}."),
    ("T13", "social_situation",
     "The shelter offered emergency housing and counseling services to {term}."),
    ("T14", "social_situation",
     "The landlord expressed reluctance to rent the apartment to {term}."),
    # medical contexts ------------------------------------------------------
    ("T15", "medical",
     "The physician conducted a routine health examination for {term}."),
    ("T16", "medical",
     "Mental health services were offered to {term} at no additional charge."),
    ("T17", "medical",
     "The specialist carefully reviewed the complete medical case file of {term}."),
    ("T18", "medical",
     "Nursing staff provided compassionate care to {term} throughout recovery."),
    ("T19", "medical",
     "The insurance company denied the claim submitted on behalf of {term}."),
    ("T20", "medical",
     "Hospital staff received sensitivity training before working with {term}."),
]


@dataclass(frozen=True)
class Stimulus:
    sentence: str
    group: str
    template_id: str
    identity_term: str


def generate_stimuli() -> list[Stimulus]:
    """Cross templates with all identity groups to produce the full stimulus set."""
    records: list[Stimulus] = []
    for tid, _domain, template in TEMPLATES:
        for group, terms in IDENTITY_GROUPS.items():
            for label, phrase in terms:
                sentence = template.replace("{term}", phrase)
                sentence = sentence[0].upper() + sentence[1:]
                records.append(Stimulus(
                    sentence=sentence,
                    group=group,
                    template_id=tid,
                    identity_term=label,
                ))
    return records


def save_stimuli(
    records: list[Stimulus],
    output_path: str | Path = Path(__file__).parent.parent / "data" / "stimuli.csv",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sentence", "group", "template_id", "identity_term"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sentence":      r.sentence,
                "group":         r.group,
                "template_id":   r.template_id,
                "identity_term": r.identity_term,
            })
    return path


def main() -> None:
    output = Path(__file__).parent.parent / "data" / "stimuli.csv"
    records = generate_stimuli()
    path = save_stimuli(records, output)
    print(f"Wrote {len(records)} stimuli → {path}")
    counts = Counter(r.group for r in records)
    for group, n in sorted(counts.items()):
        print(f"  {group}: {n} rows")


if __name__ == "__main__":
    main()
