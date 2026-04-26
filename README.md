# Hardware to AI — Learning Blog

A semiconductor engineer's notes on AI infrastructure.  
Live at: **https://anirudhs16.github.io/ai-systems**

---

## Setup (one-time, 10 minutes)

### 1. Create the repo on GitHub
- Go to github.com → New repository
- Name it exactly: `ai-systems` (or `anirudhs16.github.io` if you want it as your root site)
- Set to **Public**
- Do NOT initialize with README (you have this one)

### 2. Push this folder
```bash
cd anirudh-blog
git init
git add .
git commit -m "initial blog setup"
git branch -M main
git remote add origin https://github.com/anirudhs16/ai-systems.git
git push -u origin main
```

### 3. Enable GitHub Pages
- Go to repo → Settings → Pages
- Source: **Deploy from a branch**
- Branch: `main` / `/ (root)`
- Save
- Site is live in ~2 minutes at `https://anirudhs16.github.io/ai-systems`

### 4. Replace anirudhs16
- `_config.yml` → update LinkedIn and GitHub URLs
- Both post files → update the GitHub links at the bottom
- `about.md` → update links

---

## Weekly workflow (Friday, under 30 min)

```
1. Copy TEMPLATE-weekly-post.md → rename to YYYY-MM-DD-topic-slug.md   (2 min)
2. Fill in the 5 sections using your week's learning notes              (20 min)
3. git add . && git commit -m "week N: topic name" && git push          (2 min)
4. Copy the Hook paragraph → paste as LinkedIn post opener              (5 min)
```

That is it. One commit. One post. Green square on the calendar.

---

## Repo structure

```
ai-systems/
├── _config.yml              ← site settings (edit once)
├── index.md                 ← homepage (don't touch)
├── about.md                 ← your bio (edit once)
├── _posts/
│   ├── TEMPLATE-weekly-post.md          ← copy this every week
│   ├── 2026-04-11-flashattention-memory-layout.md
│   ├── 2026-04-18-kv-cache-memory-arithmetic.md
│   └── YYYY-MM-DD-your-next-post.md
└── README.md
```

---

## Post naming convention

```
YYYY-MM-DD-short-slug-no-spaces.md
```

Examples:
- `2026-04-25-gpu-architecture-first-benchmarks.md`
- `2026-05-02-vllm-pagedattention-virtual-memory.md`
- `2026-05-09-speculative-decoding-concept.md`

Date = the Friday you publish it.

---

## Commit message convention

Keeps your git log readable and your green squares meaningful:

```
week 4: gpu architecture + first ttft benchmarks
week 5: vllm pagedattention and benchmark script
week 5: benchmark script v1 — single vs concurrent requests
week 6: speculative decoding concept post
```

Aim for **2 commits per week minimum**: one for the blog post, one for whatever code/experiment you ran that week in your benchmark repo.

---

## The LinkedIn → Blog workflow

Every LinkedIn post is a shortened version of the blog post.

| LinkedIn post | Blog post |
|--------------|-----------|
| Hook paragraph only | Full post with formula, table, hardware parallel |
| ~300 words | ~800–1200 words |
| Publish Friday | Publish same day, link in comments |
| Optimised for emotion / curiosity | Optimised for depth / credibility |

The LinkedIn post drives people to the blog. The blog is what hiring managers and technical recruiters find when they search your name.
