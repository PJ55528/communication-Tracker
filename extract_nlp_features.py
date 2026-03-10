import os
import glob
import json
import numpy as np
import pandas as pd

# NLP libraries
import textstat
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# language tool can be slow; we'll catch exceptions around it
try:
    import language_tool_python
except ImportError:
    language_tool_python = None


# helper functions ----------------------------------------------------------

def load_spacy_model(name="en_core_web_sm"):
    """Load a spaCy model, downloading it if necessary."""
    try:
        return spacy.load(name)
    except OSError:
        # model not present, try to download
        from spacy.cli import download

        download(name)
        return spacy.load(name)


def compute_parse_depth(root_token):
    """Recursively compute depth of the subtree rooted at `root_token`."""
    if not list(root_token.children):
        return 1
    return 1 + max(compute_parse_depth(child) for child in root_token.children)


def get_subordinate_clause_rate(doc):
    """Count tokens that begin subordinate clauses (simple heuristic).

    We look for a small set of lowercased tokens that are often used to
    introduce subordinate clauses.  The rate is normalized by number of
    sentences so it roughly reflects how often the student embeds extra
    information in a sentence rather than using multiple short sentences.
    """
    markers = {"because", "although", "since", "unless", "which", "that", "when", "while"}
    if len(list(doc.sents)) == 0:
        return 0.0
    count = sum(1 for tok in doc if tok.lower_ in markers)
    return count / len(list(doc.sents))


def extract_features_for_text(text, nlp, bert_model, rubric_embeddings):
    """Compute all the requested features for a single transcript string."""
    rows = {}

    # basic counts
    words = text.split()
    total_words = len(words)
    rows["word_count"] = total_words

    # ------------------------------------------------------------------
    # Group 1: Grammar error rate (rubric: fluency/grammar)
    # ------------------------------------------------------------------
    rows["grammar_error_count"] = 0
    rows["grammar_error_rate"] = 0.0
    if language_tool_python is not None:
        try:
            tool = language_tool_python.LanguageTool("en-US")
            matches = tool.check(text)
            # filter out punctuation-only issues (whisper transcripts lack punctuation)
            grammar_matches = [m for m in matches if m.ruleIssueType != "PUNCTUATION"]
            rows["grammar_error_count"] = len(grammar_matches)
            if total_words > 0:
                rows["grammar_error_rate"] = len(grammar_matches) / total_words
        except Exception:
            # fail gracefully--leave counts at zero
            pass

    # ------------------------------------------------------------------
    # Group 2: Sentence complexity (rubric: structured answers / depth)
    # ------------------------------------------------------------------
    rows["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)

    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        # compute depth of the dependency tree for this sentence
        depths.append(compute_parse_depth(sent.root))
    rows["avg_parse_depth"] = np.mean(depths) if depths else 0.0
    rows["subordinate_clause_rate"] = get_subordinate_clause_rate(doc)

    # ------------------------------------------------------------------
    # Group 3: BERT features (rubric: coherence & domain vocabulary)
    # ------------------------------------------------------------------
    sent_texts = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if sent_texts:
        sent_embs = bert_model.encode(sent_texts)
        # compute cosine similarity between consecutive sentences
        sims = []
        for i in range(len(sent_embs) - 1):
            sims.append(cosine_similarity([sent_embs[i]], [sent_embs[i + 1]])[0][0])
        rows["coherence_score"] = float(np.mean(sims)) if sims else 0.0
    else:
        rows["coherence_score"] = 0.0

    # full-document embedding
    full_emb = bert_model.encode(text)
    rows["bert_embedding"] = full_emb

    # rubric relevance scores
    for key, crit_emb in rubric_embeddings.items():
        sim = cosine_similarity([full_emb], [crit_emb])[0][0]
        rows[key] = float(sim)

    return rows


# -----------------------------------------------------------------------------
# main entrypoint
# -----------------------------------------------------------------------------

def main():
    # folder containing JSONs; adjust as needed or assume cwd
    transcript_folder = os.getcwd()
    pattern = os.path.join(transcript_folder, "*.json")
    files = glob.glob(pattern)

    # filter out non-transcript files heuristically
    transcripts = [f for f in files if "transcript" in os.path.basename(f).lower()]

    records = []
    embeddings = []

    # prepare models once
    nlp = load_spacy_model()
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")

    # precompute rubric criterion embeddings
    rubric_texts = {
        "relevance_technical_vocab": "The student demonstrates technical knowledge and uses domain-specific vocabulary",
        "relevance_fluency": "The student speaks fluently with correct grammar and varied vocabulary",
        "relevance_structure": "The student gives structured answers with clear transitions between points",
    }
    rubric_embeddings = {k: bert_model.encode(v) for k, v in rubric_texts.items()}

    for path in transcripts:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # try to read student/session fields
        student = data.get("student") or os.path.splitext(os.path.basename(path))[0]
        session = data.get("session", "")

        # combine segments
        if "segments" in data and isinstance(data["segments"], list):
            joined = " ".join(seg.get("text", "") for seg in data["segments"])
        else:
            joined = data.get("text", "")

        feats = extract_features_for_text(joined, nlp, bert_model, rubric_embeddings)
        feats["student"] = student
        feats["session"] = session
        records.append(feats)
        embeddings.append(feats["bert_embedding"])

    # build dataframe of scalar features
    # remove embedding column from records to avoid arrays in df
    for rec in records:
        rec.pop("bert_embedding", None)

    df = pd.DataFrame.from_records(records)

    # save outputs
    df.to_csv("nlp_features.csv", index=False)
    np.save("bert_embeddings.npy", np.vstack(embeddings))

    print("=== Feature summary ===")
    print(df)


if __name__ == "__main__":
    main()
