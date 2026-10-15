# Dynamo — Landing Page

Marketing landing page for **Dynamo** (`agent_visual_harness`), a headless, production-ready AI Visual & Video Harness in Python for AI coding agents, browser automation and computer-use sandboxes. The page is written for two audiences: developers (DX, code, architecture) and enterprise buyers (security, scale, governance).

## Design system
- **Palette:** Obsidian background `#0A0B0D`, Volt lime primary `#C8FF3D`, Signal cyan secondary `#4FD8FF`, plus red, amber and neutral greys.
- **Type:** Geist for the UI, Geist Mono for code and telemetry, and Instrument Serif italic for emphasis words.
- **Background:** a subtle grid with a soft glow, and a dark terminal-style look.

## Completed features
- Sticky navigation that becomes a menu on mobile
- A hero section with a **live simulated session**. It shows the Xvfb window, a REC timer, a moving cursor, click rings, a running telemetry feed and a keyframe strip from the token reducer.
- A row of tools Dynamo works with: Python, Playwright, Docker, K8s, GitHub Actions, X11, FFmpeg
- A problem section with 3 pain points
- An architecture section with the 5 modules (display, recorder, processor, token_reducer, harness) and their specs
- An SDK section with tabbed code samples (agent.py, CLI, Dockerfile, GitHub Actions) and a copy button
- An **interactive token reduction calculator**. The numbers it shows are rough estimates, not measurements.
- An artifacts section (MP4, GIF, VLM frames, telemetry.jsonl)
- A table comparing Dynamo with screenshots only and DIY ffmpeg scripts
- An enterprise section (zero data egress, fleet scale, audit, CPU-only, model-agnostic, support)
- An FAQ that opens and closes
- A demo request form with checks on required fields. It saves entries to the `demo_requests` table.
- Click-to-copy install command, fade-in on scroll, and reduced motion for users who turn it off

## Entry points
- `/index.html` — single-page site. Section anchors: `#pipeline-section`, `#sdk-section`, `#tokens-section`, `#artifacts-section`, `#compare-section`, `#enterprise-section`, `#faq-section`, `#demo-section`
- `POST tables/demo_requests` — creates a demo request

## Data model
`demo_requests`: `id`, `name`, `email`, `company`, `team_size` (1-10 / 11-50 / 51-200 / 201-1000 / 1000+), `use_case` (Coding agents / Browser automation / QA / Computer-use sandboxes / Eval & red-teaming / Other), `message`.

## Files
- `index.html` — page markup
- `css/style.css` — design system and responsive styles
- `js/main.js` — hero simulation, code tabs, calculator, form, reveal animations

## Not yet implemented / next steps
- Real docs and GitHub links (currently internal anchors)
- Real recorded demo GIF/MP4 in place of the simulated hero
- Customer logos, testimonials and benchmark numbers once available
- Pricing page and an admin view for demo requests. A static site cannot properly protect an admin page, so this needs a real backend with authentication.
- Analytics

## Deployment
Use the **Publish tab** to publish.
