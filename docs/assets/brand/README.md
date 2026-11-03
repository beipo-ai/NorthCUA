# NorthCUA brand assets

Use the blue Beipo mark and **NorthCUA** wordmark from the latest
*NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale*.
The public runtime is the **NorthCUA Harness**. Commands, imports and environment
variables use `northcua` / `NORTHCUA_`; **NexAU** is the separate framework dependency.

## Files and paper alignment

- [northcua-mark.svg](northcua-mark.svg) is a byte-identical copy of the paper's original blue Beipo mark.
- [northcua-logo.svg](northcua-logo.svg) is the transparent horizontal lockup, with the paper's dark lettering.
- [northcua-logo-dark.svg](northcua-logo-dark.svg) changes only the wordmark fill to white; the blue mark is unchanged.

The wordmark uses **TeX Gyre Adventor Regular**. Its eight exact glyph paths,
per-letter positions and baseline were extracted from the first-page header of
[the distributed report](../../paper/main.pdf), not retyped with a substitute font.
The mark-to-wordmark gap and relative dimensions retain that header's geometry.
The complete lockup is uniformly scaled with surrounding clear space to a
600 x 160 canvas. No embedded font, remote font, raster image, script or external
resource is needed to render either SVG. The retired cursor logo and previous
Hanken Grotesk lockup are not distributed here.

[manifest.json](manifest.json) identifies the exact report snapshot, source mark,
extraction geometry and checksums of the distributed assets. The report itself
was not modified to construct these files.

## Font provenance and licenses

The report embeds TeX Gyre Adventor Regular, part of the GUST e-foundry's
[TeX Gyre collection](https://www.gust.org.pl/projects/e-foundry/tex-gyre).
Only the eight rendered outline shapes are used in the logo; no font program is
redistributed. The [GUST Font License](GUST-FONT-LICENSE.txt) and its referenced
[LaTeX Project Public License 1.3c](LPPL-1.3c.txt) are retained for attribution
and provenance. The original font has not been modified or renamed as a new font.

## Usage

GitHub READMEs select the light or dark lockup with GitHub's
`#gh-light-mode-only` and `#gh-dark-mode-only` image/link fragments. These track
the reader's GitHub theme, including when it differs from the operating system's
appearance. Do not replace them with a `picture` element using
`prefers-color-scheme`: that follows the system appearance instead of GitHub's
selected theme. For other renderers, select the appropriate SVG explicitly.
Preserve the mark's original blue and use the appropriate lettering for the background.
Keep the full lockup proportional and leave clear space around it. Avoid adding
another large project-name heading directly beneath it.

The brand slogan is **Worlds in mind. Wonders at your fingertips.**
In Chinese: **心中万象，指间成形。** Place it below the lockup without repeating
the project name. Historical filenames and recorded evidence hashes retain their
original identities; current research citations use the distributed report's
actual NorthCUA title and authors.
