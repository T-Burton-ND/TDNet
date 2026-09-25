---
type: reference
up: "[[Home_TDNet]]"
---

# Wiki Schema — TDNet

> Conventions and workflows for LLM maintenance of this wiki.
> Based on the [llm-wiki pattern](https://github.com/tobi/llm-wiki).

## Purpose

This wiki is a persistent, compounding knowledge base. The LLM writes and maintains all pages. The human curates sources, directs analysis, and asks questions. Knowledge is compiled once and kept current, not re-derived every session.

**Write discipline.** Treat the wiki as durable, queryable memory, not a summary. Your work succeeds only if a future agent, reading only these pages, can recover a source's specific claims -- the exact quantities (with their units), the named systems and methods, and the conditions -- and see how they connect to other pages. Preserve those particulars as first-class, retrievable structure (their own pages or clearly-marked sections, cross-linked), not merely as passing mentions folded under general concepts. A page that keeps the general ideas but loses the specifics has failed, however tidy it looks.

## Source of Truth

**Raw sources** (immutable — read but never modify):
- Project code and documentation
- Data files and results
- External references

**The wiki** (LLM-owned — create, update, cross-reference, maintain):
- All `.md` files in this directory

## Page Format

Every content page should include:

1. **Title** — `# Page Name` as H1
2. **Opening line** — One sentence summarizing what this page is about
3. **Body** — Tables, prose, code blocks as appropriate. Concise reference style.
4. **Cross-references** — `See also:` line at the bottom with `[Display Name](Page-Name)` links

## Frontmatter

Every page gets standard YAML frontmatter:

```markdown
---
type: concept | entity | source-summary | synthesis | analysis | decision | index | comparison | untyped
up: "[[Parent-Page]]"
tags: [topic-a, topic-b]
---
```

**Required fields**:
- `type:` — what kind of page this is (use `untyped` if unsure)
- `up:` — parent page in the hierarchy (usually a category page or index)

**Optional typed edges** (add when the relationship is clear):
- `source:` — literature or raw source this page summarizes
- `extends:` — concept or page this builds upon
- `supports:` / `criticizes:` — claim or page this provides evidence for or against
- `related:` — lateral connection (prefer specific edges above when possible)

**Rules**:
- Every page gets frontmatter — no exceptions
- Use `type: untyped` rather than skipping frontmatter entirely
- Cross-references in frontmatter use `[[Page-Name]]` wikilink format for Obsidian compatibility
- Cross-references in body text use `[Display](Page-Name)` format for GitHub wiki compatibility
- The frontmatter feeds the knowledge graph pipeline for SPARQL queries

## Page types

Most page types (`concept`, `entity`, `synthesis`, etc.) have no required structure beyond the page format. Two query-driven types do, because they exist to capture content that would otherwise be dropped on the floor mid-session.

### `analysis`

A query-driven assessment or evaluation. Use when the user asks a synthesis question (`why X`, `compare A and B`, `should we ...`) and the answer is fileable.

**Required sections**:
1. **Question** — the synthesis question being answered, verbatim or close to it
2. **Context** — what background drove the question
3. **Analysis** — the body of the assessment
4. **Conclusion** — the bottom line
5. **Open follow-ups** — what is unresolved

**Required frontmatter**: `derived_from:` listing the source pages synthesised (one or more wikilinks). Without it, the analysis is unprovenanced.

### `decision`

A design choice with rationale. Use when the project picks one option over alternatives and the reasoning should outlast the decision.

**Required sections**:
1. **Question** — the choice being made
2. **Options considered** — each option with pros / cons
3. **Decision** — what was chosen
4. **Rejected alternatives** — what was not chosen, and why
5. **Revisit triggers** — conditions under which the choice should be re-opened

**Required frontmatter**: `decided_at: YYYY-MM-DD`. **Optional**: `superseded_by: [[Page]]` once a later decision replaces this one.

## Naming Convention

- Use `Title-Case-Hyphenated.md` for page files (e.g., `Neural-Embeddings.md`)
- Navigation files are namespaced: `index_TDNet.md`, `log_TDNet.md`, etc.

## Special Files

### index_TDNet.md
- Catalog of every page with one-line descriptions
- Organized by category
- **Update on every ingest**

### log_TDNet.md
- Append-only chronological record
- Format: `## [YYYY-MM-DD] verb | Subject` (verbs: ingest, query, lint, update, create)
- The first bullet of every entry is the `- by:` attribution line (see Log Entry Attribution)
- Then 2-5 bullet points describing the operation
- **Append on every operation**, and commit each entry on its own (see Log Entry Attribution)

### Home_TDNet.md
- Human-facing entry point. Project description + a `## Categories` section mirroring the top-level categories in `index_TDNet.md`, with 1-3 representative links per category. **Not** a comprehensive catalog; that is the Index's job.
- **Update when a new top-level category emerges in the Index, or when a page lands that is significant enough to be one of its category's representative links.** Routine page additions inside an existing category: Index-only, no Home update needed.
- Distinct from `Home.md` (the GitHub-wiki redirect below), which is never edited.

### Home.md
- GitHub wiki redirect only — do not edit
- Real home page is [Home_TDNet](Home_TDNet)

## Cross-Referencing

- Use `[Display Text](Page-Name)` format in body text (GitHub wiki style)
- Use `[[Page-Name]]` wikilinks in frontmatter edge fields (Obsidian graph)
- Every page should have at least 2 inbound and 2 outbound links
- Link concepts on first mention within a page
- Bidirectional: if A links to B, B should link back to A

## Edges as Interface Operations

A typed edge is not a label that says "these two things are related." It is an **interface contract** that defines an *operation* the agent should perform when it traverses the edge. Each edge type has an expected behavior, an expected target type, and a semantic commitment that shapes how downstream retrieval and reasoning should handle it.

| Edge | Inverse | What it licenses the agent to do |
|---|---|---|
| `extends:` | `extendedBy` | Inherit semantic context from the parent. Treat the parent's claims as background assumptions for the current page. |
| `supports:` | `supportedBy` | Evidence aggregation. When traversing this edge, expect to combine claims; consistency across supporting pages is desirable. |
| `criticizes:` | `criticizedBy` | Contradiction detection. Expect an unresolved tension that should trigger conflict-resolution logic before combining evidence. |
| `source:` | (none — external) | Grounding check. The target is an external source; verify that any cited claim traces back to the source. |
| `up:` | (none — implicit) | Parent / breadcrumb. Navigate upward in the hierarchy for category context. |
| `related:` | (none — symmetric) | Fallback — no specific operation contract. Prefer a more specific edge type where one applies. |

**Inverses are materialised by the KG, not authored.** Inverse predicates exist so that a knowledge-graph build can traverse a typed edge in either direction; the build emits the inverse triple automatically from each forward assertion. **Agents do not write `extendedBy:`, `supportedBy:`, etc. in source documents.** When a back-reference would help a reader navigate, add it at the body level (typically in the target page's See also section), not as a frontmatter inverse. See [Edge-Types](Edge-Types) for the full 16-predicate vocabulary.

Practical implications:

- Different edge types license different retrievals. A knowledge-graph build consumes these typed edges to produce a queryable graph; the typing is what makes structural queries useful.
- Edge types have domain and range. `extends:` points at a concept or theory; `source:` points at an external resource. Violating the domain/range breaks the operation.
- Populate edge fields with the most specific type you can justify. Treat `related:` as a fallback. Over time, `related:` uses should become rarer as the edge vocabulary fits the work.

## Inline body annotations (Variant 1)

The same predicates from `extends:` / `supports:` / `criticizes:` etc. in frontmatter can be applied inline to body links. The form is a content link followed by a parenthesised italicised link to the [Edge-Types](Edge-Types) page anchor:

```markdown
This claim extends the framing in [Theory X](Theory-X) ([*extends*](Edge-Types#extends)).
```

Two links in the rendered output: the content link is the normal cross-reference (emits a `mentions` edge), and the parenthesised italicised predicate link is the carrier (adds the typed edge from the source page to the target). The predicate link is filtered out of `mentions` by the KG extractor so the Edge-Types page does not become a spurious hub.

**Frontmatter versus inline.** Two granularities, both processed by the KG.

- **Frontmatter** asserts a page-level relationship: the entire page `extends:` X.
- **Inline (Variant 1)** asserts a per-mention relationship: this specific paragraph's reference to X carries the typed edge.

Use frontmatter when the whole page relates to the target that way. Use inline when only one particular reference in one paragraph carries the relationship, or when the same target appears multiple times with different rhetorical positions.

**Agent judgment, not heuristic.** Annotations are added by reading the prose: in this specific context, is there a clear typed-edge predicate that captures the relationship between the source page and the target page? Apply the most specific predicate that fits. Default to no annotation when uncertain. Sparse-accurate beats dense-speculative; a spurious typed-edge claim distorts downstream queries.

**When to annotate**: the predicate is clearly the most-specific fit, the relationship is a per-mention claim, the target is a real wiki page (not an external URL, not a fragment-only reference), and the annotation would be informative to a future reader and queryable by the KG.

**When not to annotate**: multiple predicates plausibly fit and none clearly wins; the relationship is too vague to commit to (default `mentions` is fine); the link target is a fragment (`Page-Name#section`). Do not pair a fragment-targeted link with an inline annotation; the annotation asserts a page-level relationship and the fragment implies a sub-page target, so the two are incoherent together.

**Where the vocabulary lives.** [Edge-Types](Edge-Types) lists the 16 forward predicates with one-line definitions; each section is the anchor target for the parenthesised carrier (e.g. `[*partOf*](Edge-Types#partOf)` resolves to the `## partOf` heading on that page).

## Topology vs Content (when to use the KG)

Two distinct retrieval shapes, each suited to a different question:

- **Topology questions** — *what connects to what*. Multi-hop relationships, concept chains, parent/child rollups, hub detection ("which pages cite this finding?"). Answer by following typed edges and `[[links]]` across pages; this is the query shape a knowledge-graph build over the typed edges serves when one is available.
- **Content questions** — *what does this page actually say*. Definitions, prose claims, specific numbers, source quotations. Use a direct file read or grep.

The right pattern: use the KG to discover *where* to look (which pages connect to the topic), then file tools to read *what* the chosen pages say. Reserve grep for non-wiki code or for content searches that span many files.

## Log Entry Attribution

The wiki is a shared memory across a team. Every log entry records who performed the operation, so provenance is answerable on the page itself and not only through `git blame`.

**The `by:` field.** The first bullet of every `log_TDNet.md` entry is an attribution line:

```
- by: <human> via <agent>
```

- `<human>` is the value of `git config user.name` in the wiki repo at the time of the operation. Read it; do not invent it. If it does not match the identity that commits the entry, that is a bug to fix, not a value to guess.
- `<agent>` is the coding assistant the operation ran under, for example `claude-code` or `cursor`.

**One commit per log entry.** Never bundle multiple log operations into a single commit. Each append to `log_TDNet.md` is committed on its own: commit the page and index changes first, then the log entry as its own commit. This keeps `git blame` on the log file a faithful per-entry record, one entry mapping to one commit and one author.

**Two records, one source of truth.** The `by:` field is the human-readable copy; git history is the verifiable record. They should always agree. If they disagree, trust git and correct the field.

## Operations

### Ingest (new work completed)

When new sources, experiments, or milestones arrive:

1. Read the source material
2. Discuss key takeaways with the user
3. Create new pages or update existing ones (with frontmatter)
4. Update cross-references on all affected pages
5. Update `index_TDNet.md`
6. Append to `log_TDNet.md`

A single ingest typically touches 5-15 pages.

### Query (answering questions)

1. Read `index_TDNet.md` to find relevant pages
2. Read those pages and synthesize an answer
3. If the answer is valuable and reusable, offer to file it as a new page

### Lint (health check)

Periodically check for:
- Orphan pages (no inbound links)
- Dead links (links to non-existent pages)
- Stale claims (superseded by newer work)
- Missing pages (concepts mentioned but lacking their own page)
- Missing cross-references
- **Pages missing frontmatter** — add it based on page content (infer type, parent, tags)
- **Pages with `type: untyped`** — review and assign a proper type if now obvious

## When to Update

- **Always**: After completing significant work (experiments, milestones, key findings)
- **Always**: When findings contradict or supersede previous ones
- **Often**: When analytical questions produce reusable answers
- **Periodically**: Lint pass every few sessions

## When NOT to Update

- Routine debugging or code fixes (git history is enough)
- Temporary analysis that won't be referenced again
- Speculative plans that haven't been executed

## Git Workflow

After wiki updates:
1. Stage changed files by name
2. Commit with descriptive message
3. Do NOT push unless the user requests it

## Evolution

Update this schema as the project's needs change. It's a living document.
