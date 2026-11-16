# Running a batch of questions

This disposable helper runs the existing retrieval CLI once per question. It does
not change the app or grade its answers. Write your own questions for Step 3.

## Prepare questions

Create `questions.txt` in Week 1 with one complete question per line. Any number
of questions is accepted, starting at one. Blank lines are ignored; all other
lines are questions, so omit headings, comments, and numbering unless you want
them sent to the app. Keep expected answers in separate notes.

## Run

From the repository root:

```bash
cd "00_Technical_Challenges/Week 1"
uv run python run_questions.py questions.txt
```

The app uses its usual `.env` and API configuration. Each question runs in a
separate process using the same Python interpreter as the runner. The runner
always launches the app from Week 1, regardless of your starting directory.

Optional settings:

```bash
uv run python run_questions.py questions.txt -k 5 --timeout 600
uv run python run_questions.py questions.txt --corpus data/your_document.txt
uv run python run_questions.py --help
```

Question-file and explicit corpus paths are relative to your current directory.
Without `--corpus`, there must be exactly one `.txt` in Week 1's `data/` folder.
The timeout defaults to 300 seconds per question. Failed or timed-out questions
are recorded and the runner continues; there are no automatic retries.

## Read the report

Each run creates a new report in `logs/evaluations/`, using a UTC timestamp and
unique suffix. Existing reports are never overwritten. Use `--output-dir PATH`
to select another directory. The default logs directory is already Git-ignored;
copy reports elsewhere deliberately if you want to preserve them in Git.

Reports include the inputs, answers, source previews, diagnostics, exit statuses,
and durations in question order. Stdout and stderr are stored separately within
each question's section; they are not a chronological merged transcript. Complete
results are flushed after each question. Ctrl+C leaves a partial report, though
output from the interrupted question may not be retained.

Fill in the manual PASS/FAIL and failure-stage fields after reviewing each answer.
A successful exit status does not mean the answer is correct. The runner exits
with status 1 if any question fails to execute or times out, and 130 if interrupted.

The CLI's `--show-chunks` prints only up to 400 characters per retrieved chunk.
Inspect the complete chunks separately when those previews do not explain a
failure. Every question still rebuilds the corpus embeddings and makes API
requests; caching remains deferred. Keep the corpus, code, and settings unchanged
while a batch is running, and use the same question file for before/after runs.
