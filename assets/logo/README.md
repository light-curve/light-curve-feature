# Logo

Copies of artwork from [`light-curve/branding`](https://github.com/light-curve/branding),
where the Illustrator sources and every other variant live. Do not edit them here — change
them upstream and copy the result across, so the two stay byte-identical.

Designed by [Anastasiia Voloshina](https://www.behance.net/anastasvoloshi2), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| File | Used by |
|---|---|
| `wordmark-for-{light,dark}-bg.svg` | the banner at the top of `README.md` |
| `mark-adaptive.svg` | `html_logo_url` and `html_favicon_url`, the rustdoc sidebar and tab icon |

The pair is picked between by a `<picture>` element, which follows the reader's theme.
`mark-adaptive.svg` carries both purples and switches internally on `prefers-color-scheme`,
which is all a file loaded through `<img>` can do — it is an isolated document that the page
around it cannot style.
