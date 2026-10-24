# Dataset

| Path | Content | In git |
|---|---|---|
| `raw/pages/*.json` | 1,096 Vinmec diabetes pages, one per article: title, URL, date, sections (heading, paragraphs, bullets) | no (copyrighted text; rebuild with `python kg/crawl.py`) |
| `raw/index.csv` | every checked candidate page with its mention counts and whether it was kept | yes |
| `processed/entities.csv` | `name, label, threshold` | yes |
| `processed/triples.csv` | `subject, relation, object, support, source, example_urls` | yes |
| `processed/articles.csv` | `url, title, page_type, date_published, about` | yes |
| `processed/mentions.csv` | `url, entity`: every entity mentioned anywhere on a page | yes |

Source: public articles of https://www.vinmec.com (`/vie/bai-viet/`, `/vie/benh/`), collected through the sitemap at 1 request per second. `processed/` is rebuilt with `python kg/extract.py`.
