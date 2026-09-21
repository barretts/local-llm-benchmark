# Measured findings report

The final, reviewed report is [local-coding-agent-benchmark-report.pdf](local-coding-agent-benchmark-report.pdf). It covers the original 64K search, separate user-authorized 16K trials, Bonsai, speculative decoding, engine availability, provisional picks, and measurement limits.

The PDF has 18 pages and embeds exact copies of both evidence snapshots:

- [original-audit.json](evidence/original-audit.json): the original export's 50 configurations and 932 attempts, including versions, hashes, launch arguments, coding outcomes, and diagnostic timings.
- [focused-audit.json](evidence/focused-audit.json): later 16K and Bonsai trials, family results, tool failures, memory probes, and MTP/DFlash findings.
- [pdf-validation.json](evidence/pdf-validation.json): reviewed PDF SHA256, page count, attachment checks, factual review, and visual validation.

The combined durable benchmark snapshot contained 63 registered configurations and 1,093 attempt records. These include screening, retries, and interrupted attempts, rather than 1,093 distinct coding tasks. No configuration qualified as the complete 64K winner or as a completed 16K agent winner.

To regenerate on the original host, copy the two audit snapshots into `tmp/pdfs/`, provide the report generator's private Python dependencies, and run `scripts/build-findings-report.py`. Its Windows font and path assumptions are documented in the repository README. Full raw streams, resource logs, grading artifacts, and the checkpoint database remain in the original local workspace.
