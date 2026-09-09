# Docs generator replacement

This PR replaces the accepted API-only spike with a small API generator plus an explicit guide publisher so model-card and reproduction readers keep working after the route move.

The intended reader is the maintainer deciding whether to merge [issue #322](https://github.com/creative-graphic-design/design-generators/issues/322) and fund the later legacy-script cleanup.

## Outcome

The API generator discovers non-private Python modules from the root uv workspace members and writes only docs/api/.

scripts/publish_doc_guides.py is the separate declarative copy step, with a small publish function that writes model and library guides under docs/models/ and docs/libraries/.

The publisher copies README.md, REPRODUCING.md, and TRAINING.md when present, adds the required docs frontmatter, writes literate-nav SUMMARY.md files, and leaves mkdocs.yml responsible for the authored nav.

The publisher rewrites only repository-root-relative Markdown links beginning with models/, lib/, or docs/; local links, external links, fragments, and unrelated paths remain unchanged.

The site landing page and models overview are hand-written under docs/, so the replacement no longer copies the root README, invents model badges, or mutates authored navigation.

The hand-written docs/models.md overview now links to the copied model guides and the API pages, so its checkpoint and training links do not point at removed /api/ guide routes.

Zensical has no native redirect plugin in the selected compatibility surface, so the old guide URLs are listed below for the release announcement and any hosting-layer redirect decision.

## Route collisions

The five old training-guide URLs collided with generated training API sections: cgb-dm, dlt, layout-dm, layout-flow, and layoutdiffusion.

The API pages retain their /api/models/<pkg>/training/ routes, while the reader-facing guides move to /models/<pkg>/training/; the collision list below proves both routes exist with different content.

## API exposure review

The 93-URL spike inventory contained seven concrete model test or vendor-only pages, so discovery excludes the basenames testing, vendor_parity, vendor_state, and vendor_state_dict.

The model exclusions are flex_dm.testing, layout_fid.testing, layoutprompter.vendor_parity, postero.vendor_parity, housegan.vendor_state_dict, layout_detr.vendor_state, and ltnet.vendor_state_dict; the same testing rule also drops the three shared-library helpers laygen.agents.testing, laygen.common.testing, and posgen.common.testing.

No src/scripts module was found, and no other new-only module had a name that identified it as a test helper or vendor adapter; conversion, data, training, and parity modules remain visible because they contain package APIs used by contributors.

## Proof

The clean-head legacy site has 310 page URLs; the final replacement site has 397 page URLs, with 65 old-only URLs and 152 new-only URLs.

The final generator discovers 319 API modules and writes 321 API Markdown files including api/index.md and api/SUMMARY.md; the publisher writes 32 README pages, 28 reproduction pages, 5 training pages, and 2 guide summaries.

The strict build passes with Zensical 0.0.60 after running uv run --group docs python scripts/gen_api_pages.py, uv run --group docs python scripts/publish_doc_guides.py, and uv run --group docs zensical build --strict --clean -f mkdocs.yml.

site/index.html contains architecture/, training-reproduction/, conventions/, getting-started/, models/, libraries/, and the API subtree.

Focused generator and publisher tests pass 5/5, Ruff passes, and the frontmatter test covers copied guides as well as authored docs pages.

## CI and author workflow

CI now runs both small pre-build scripts before the strict Zensical build, and changes to either script trigger the documentation job.

Authors keep adding hand-written pages and nav entries in docs/ and mkdocs.yml; package authors keep README.md, REPRODUCING.md, and TRAINING.md beside each workspace member and the publisher exposes them in the guide tree.

The API configuration follows the [mkdocstrings automatic code reference recipe](https://mkdocstrings.github.io/recipes/), and native Zensical support follows its [MkDocs-plugin compatibility page](https://zensical.org/docs/compatibility/mkdocs/plugins/).

## URL diff

Old-only URLs (65)
- /api/libraries/: Old synthetic group landing page replaced by the explicit API subtree and its local literate-nav summary.
- /api/libraries/laygen/agents/testing/: Removed after exposure review: this module is a test helper and the replacement excludes the testing basename from API discovery.
- /api/libraries/laygen/common/testing/: Removed after exposure review: this module is a test helper and the replacement excludes the testing basename from API discovery.
- /api/libraries/laygen/package/: Old synthetic package landing page moved to the native section page at /api/libraries/laygen/.
- /api/libraries/posgen/common/testing/: Removed after exposure review: this module is a test helper and the replacement excludes the testing basename from API discovery.
- /api/libraries/posgen/package/: Old synthetic package landing page moved to the native section page at /api/libraries/posgen/.
- /api/libraries/traingen-parity/package/: Old synthetic package landing page moved to the native section page at /api/libraries/traingen-parity/.
- /api/libraries/traingen/package/: Old synthetic package landing page moved to the native section page at /api/libraries/traingen/.
- /api/models/: Old synthetic group landing page replaced by the explicit API subtree and its local literate-nav summary.
- /api/models/basnet/package/: Old synthetic package landing page moved to the native section page at /api/models/basnet/.
- /api/models/basnet/reproducing/: Reader-facing reproduction guide relocated to /models/basnet/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/cgb-dm/package/: Old synthetic package landing page moved to the native section page at /api/models/cgb-dm/.
- /api/models/cgb-dm/reproducing/: Reader-facing reproduction guide relocated to /models/cgb-dm/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/coarse-to-fine/package/: Old synthetic package landing page moved to the native section page at /api/models/coarse-to-fine/.
- /api/models/coarse-to-fine/reproducing/: Reader-facing reproduction guide relocated to /models/coarse-to-fine/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/dlt/package/: Old synthetic package landing page moved to the native section page at /api/models/dlt/.
- /api/models/dlt/reproducing/: Reader-facing reproduction guide relocated to /models/dlt/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/ds-gan/package/: Old synthetic package landing page moved to the native section page at /api/models/ds-gan/.
- /api/models/ds-gan/reproducing/: Reader-facing reproduction guide relocated to /models/ds-gan/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/flex-dm/package/: Old synthetic package landing page moved to the native section page at /api/models/flex-dm/.
- /api/models/flex-dm/reproducing/: Reader-facing reproduction guide relocated to /models/flex-dm/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/housegan/package/: Old synthetic package landing page moved to the native section page at /api/models/housegan/.
- /api/models/housegan/reproducing/: Reader-facing reproduction guide relocated to /models/housegan/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/lace/package/: Old synthetic package landing page moved to the native section page at /api/models/lace/.
- /api/models/lace/reproducing/: Reader-facing reproduction guide relocated to /models/lace/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layousyn/package/: Old synthetic package landing page moved to the native section page at /api/models/layousyn/.
- /api/models/layousyn/reproducing/: Reader-facing reproduction guide relocated to /models/layousyn/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-action/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-action/.
- /api/models/layout-action/reproducing/: Reader-facing reproduction guide relocated to /models/layout-action/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-corrector/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-corrector/.
- /api/models/layout-corrector/reproducing/: Reader-facing reproduction guide relocated to /models/layout-corrector/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-detr/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-detr/.
- /api/models/layout-detr/reproducing/: Reader-facing reproduction guide relocated to /models/layout-detr/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-dm/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-dm/.
- /api/models/layout-dm/reproducing/: Reader-facing reproduction guide relocated to /models/layout-dm/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-fid/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-fid/.
- /api/models/layout-fid/reproducing/: Reader-facing reproduction guide relocated to /models/layout-fid/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-flow/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-flow/.
- /api/models/layout-flow/reproducing/: Reader-facing reproduction guide relocated to /models/layout-flow/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layout-gpt/package/: Old synthetic package landing page moved to the native section page at /api/models/layout-gpt/.
- /api/models/layout-gpt/reproducing/: Reader-facing reproduction guide relocated to /models/layout-gpt/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layoutdiffusion/package/: Old synthetic package landing page moved to the native section page at /api/models/layoutdiffusion/.
- /api/models/layoutdiffusion/reproducing/: Reader-facing reproduction guide relocated to /models/layoutdiffusion/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layoutformerpp/package/: Old synthetic package landing page moved to the native section page at /api/models/layoutformerpp/.
- /api/models/layoutformerpp/reproducing/: Reader-facing reproduction guide relocated to /models/layoutformerpp/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layoutganpp/package/: Old synthetic package landing page moved to the native section page at /api/models/layoutganpp/.
- /api/models/layoutganpp/reproducing/: Reader-facing reproduction guide relocated to /models/layoutganpp/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layoutprompter/package/: Old synthetic package landing page moved to the native section page at /api/models/layoutprompter/.
- /api/models/layoutprompter/reproducing/: Reader-facing reproduction guide relocated to /models/layoutprompter/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/layoutvae/package/: Old synthetic package landing page moved to the native section page at /api/models/layoutvae/.
- /api/models/layoutvae/reproducing/: Reader-facing reproduction guide relocated to /models/layoutvae/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/ltnet/package/: Old synthetic package landing page moved to the native section page at /api/models/ltnet/.
- /api/models/ltnet/reproducing/: Reader-facing reproduction guide relocated to /models/ltnet/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/parse-then-place/package/: Old synthetic package landing page moved to the native section page at /api/models/parse-then-place/.
- /api/models/parse-then-place/reproducing/: Reader-facing reproduction guide relocated to /models/parse-then-place/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/posterllama/package/: Old synthetic package landing page moved to the native section page at /api/models/posterllama/.
- /api/models/posterllama/reproducing/: Reader-facing reproduction guide relocated to /models/posterllama/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/posterllava/package/: Old synthetic package landing page moved to the native section page at /api/models/posterllava/.
- /api/models/posterllava/reproducing/: Reader-facing reproduction guide relocated to /models/posterllava/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/postero/package/: Old synthetic package landing page moved to the native section page at /api/models/postero/.
- /api/models/postero/reproducing/: Reader-facing reproduction guide relocated to /models/postero/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/ralf/package/: Old synthetic package landing page moved to the native section page at /api/models/ralf/.
- /api/models/ralf/reproducing/: Reader-facing reproduction guide relocated to /models/ralf/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.
- /api/models/smarttext/package/: Old synthetic package landing page moved to the native section page at /api/models/smarttext/.
- /api/models/smarttext/reproducing/: Reader-facing reproduction guide relocated to /models/smarttext/reproducing/; the complete old-to-new mapping appears below because Zensical has no native redirect plugin.

New-only URLs (153)
- /api/SUMMARY/: Navigation-only API SUMMARY.md emitted for literate-nav.
- /api/libraries/laygen/agents/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/libraries/laygen/common/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/libraries/laygen/nn/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/libraries/laygen/pipelines/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/libraries/laygen/schedulers/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/libraries/posgen/common/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/libraries/traingen/lightning/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/data/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/config/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/datamodule/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/dataset/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/lightning_module/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/losses/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/parity/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/cgb-dm/training/seed/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/coarse-to-fine/geometry/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/callbacks/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/config/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/datamodule/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/dataset/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/lightning_module/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/losses/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/parity/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/dlt/training/seed/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/flex-dm/data_specs/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/flex-dm/masking/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/flex-dm/tf_checkpoint/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/housegan/datasets/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/housegan/visualization/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-action/data/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/config/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/datamodule/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/dataset/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/lightning_module/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/losses/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/parity/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-dm/training/seed/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-fid/metrics/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/sampling/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/config/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/datamodule/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/dataset/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/lightning_module/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/losses/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/parity/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-flow/training/seed/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-gpt/exemplars/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-gpt/parser/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-gpt/prompts/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layout-gpt/types/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/conditioning/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/labels/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/sampling/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/config/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/datamodule/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/dataset/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/lightning_module/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/losses/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/parity/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/seed/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutdiffusion/training/vocab/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutformerpp/geometry/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutformerpp/serialization/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutganpp/bbox/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/arrays/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/data/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/parsing/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/records/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/selection/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/serialization/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/layoutprompter/similarity/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/posterllama/postprocessing/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/posterllava/generation_posterllava/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/postero/enums/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/postero/exemplars/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/postero/parser/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/postero/prompts/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/postero/schemas/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/postero/serialization/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/ralf/datasets/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/smarttext/candidate_generation/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /api/models/smarttext/color/: API page from a non-private module file under a workspace member src tree; the reviewed test and vendor-only basenames are excluded.
- /architecture/: Hand-written page restored under the authored MkDocs nav.
- /libraries/SUMMARY/: Navigation-only guide SUMMARY.md emitted for literate-nav.
- /libraries/laygen/: Reader-facing library README copied to /libraries/.
- /libraries/posgen/: Reader-facing library README copied to /libraries/.
- /libraries/traingen-parity/: Reader-facing library README copied to /libraries/.
- /libraries/traingen/: Reader-facing library README copied to /libraries/.
- /models/SUMMARY/: Navigation-only guide SUMMARY.md emitted for literate-nav.
- /models/basnet/: Reader-facing model README copied to /models/basnet/.
- /models/basnet/reproducing/: Reader-facing reproduction guide copied to /models/basnet/reproducing/.
- /models/cgb-dm/: Reader-facing model README copied to /models/cgb-dm/.
- /models/cgb-dm/reproducing/: Reader-facing reproduction guide copied to /models/cgb-dm/reproducing/.
- /models/cgb-dm/training/: Reader-facing training guide copied to /models/cgb-dm/training/.
- /models/coarse-to-fine/: Reader-facing model README copied to /models/coarse-to-fine/.
- /models/coarse-to-fine/reproducing/: Reader-facing reproduction guide copied to /models/coarse-to-fine/reproducing/.
- /models/dlt/: Reader-facing model README copied to /models/dlt/.
- /models/dlt/reproducing/: Reader-facing reproduction guide copied to /models/dlt/reproducing/.
- /models/dlt/training/: Reader-facing training guide copied to /models/dlt/training/.
- /models/ds-gan/: Reader-facing model README copied to /models/ds-gan/.
- /models/ds-gan/reproducing/: Reader-facing reproduction guide copied to /models/ds-gan/reproducing/.
- /models/flex-dm/: Reader-facing model README copied to /models/flex-dm/.
- /models/flex-dm/reproducing/: Reader-facing reproduction guide copied to /models/flex-dm/reproducing/.
- /models/housegan/: Reader-facing model README copied to /models/housegan/.
- /models/housegan/reproducing/: Reader-facing reproduction guide copied to /models/housegan/reproducing/.
- /models/lace/: Reader-facing model README copied to /models/lace/.
- /models/lace/reproducing/: Reader-facing reproduction guide copied to /models/lace/reproducing/.
- /models/layousyn/: Reader-facing model README copied to /models/layousyn/.
- /models/layousyn/reproducing/: Reader-facing reproduction guide copied to /models/layousyn/reproducing/.
- /models/layout-action/: Reader-facing model README copied to /models/layout-action/.
- /models/layout-action/reproducing/: Reader-facing reproduction guide copied to /models/layout-action/reproducing/.
- /models/layout-corrector/: Reader-facing model README copied to /models/layout-corrector/.
- /models/layout-corrector/reproducing/: Reader-facing reproduction guide copied to /models/layout-corrector/reproducing/.
- /models/layout-detr/: Reader-facing model README copied to /models/layout-detr/.
- /models/layout-detr/reproducing/: Reader-facing reproduction guide copied to /models/layout-detr/reproducing/.
- /models/layout-dm/: Reader-facing model README copied to /models/layout-dm/.
- /models/layout-dm/reproducing/: Reader-facing reproduction guide copied to /models/layout-dm/reproducing/.
- /models/layout-dm/training/: Reader-facing training guide copied to /models/layout-dm/training/.
- /models/layout-fid/: Reader-facing model README copied to /models/layout-fid/.
- /models/layout-fid/reproducing/: Reader-facing reproduction guide copied to /models/layout-fid/reproducing/.
- /models/layout-flow/: Reader-facing model README copied to /models/layout-flow/.
- /models/layout-flow/reproducing/: Reader-facing reproduction guide copied to /models/layout-flow/reproducing/.
- /models/layout-flow/training/: Reader-facing training guide copied to /models/layout-flow/training/.
- /models/layout-gpt/: Reader-facing model README copied to /models/layout-gpt/.
- /models/layout-gpt/reproducing/: Reader-facing reproduction guide copied to /models/layout-gpt/reproducing/.
- /models/layoutdiffusion/: Reader-facing model README copied to /models/layoutdiffusion/.
- /models/layoutdiffusion/reproducing/: Reader-facing reproduction guide copied to /models/layoutdiffusion/reproducing/.
- /models/layoutdiffusion/training/: Reader-facing training guide copied to /models/layoutdiffusion/training/.
- /models/layoutformerpp/: Reader-facing model README copied to /models/layoutformerpp/.
- /models/layoutformerpp/reproducing/: Reader-facing reproduction guide copied to /models/layoutformerpp/reproducing/.
- /models/layoutganpp/: Reader-facing model README copied to /models/layoutganpp/.
- /models/layoutganpp/reproducing/: Reader-facing reproduction guide copied to /models/layoutganpp/reproducing/.
- /models/layoutprompter/: Reader-facing model README copied to /models/layoutprompter/.
- /models/layoutprompter/reproducing/: Reader-facing reproduction guide copied to /models/layoutprompter/reproducing/.
- /models/layoutvae/: Reader-facing model README copied to /models/layoutvae/.
- /models/layoutvae/reproducing/: Reader-facing reproduction guide copied to /models/layoutvae/reproducing/.
- /models/ltnet/: Reader-facing model README copied to /models/ltnet/.
- /models/ltnet/reproducing/: Reader-facing reproduction guide copied to /models/ltnet/reproducing/.
- /models/parse-then-place/: Reader-facing model README copied to /models/parse-then-place/.
- /models/parse-then-place/reproducing/: Reader-facing reproduction guide copied to /models/parse-then-place/reproducing/.
- /models/posterllama/: Reader-facing model README copied to /models/posterllama/.
- /models/posterllama/reproducing/: Reader-facing reproduction guide copied to /models/posterllama/reproducing/.
- /models/posterllava/: Reader-facing model README copied to /models/posterllava/.
- /models/posterllava/reproducing/: Reader-facing reproduction guide copied to /models/posterllava/reproducing/.
- /models/postero/: Reader-facing model README copied to /models/postero/.
- /models/postero/reproducing/: Reader-facing reproduction guide copied to /models/postero/reproducing/.
- /models/ralf/: Reader-facing model README copied to /models/ralf/.
- /models/ralf/reproducing/: Reader-facing reproduction guide copied to /models/ralf/reproducing/.
- /models/smarttext/: Reader-facing model README copied to /models/smarttext/.
- /models/smarttext/reproducing/: Reader-facing reproduction guide copied to /models/smarttext/reproducing/.
- /plans/docs-generator-replacement/: Local planning document present under docs/ but intentionally unlisted from the authored nav.

## Guide-page parity proof

- /api/models/basnet/reproducing/ -> /models/basnet/reproducing/: present in the final site.
- /api/models/cgb-dm/reproducing/ -> /models/cgb-dm/reproducing/: present in the final site.
- /api/models/coarse-to-fine/reproducing/ -> /models/coarse-to-fine/reproducing/: present in the final site.
- /api/models/dlt/reproducing/ -> /models/dlt/reproducing/: present in the final site.
- /api/models/ds-gan/reproducing/ -> /models/ds-gan/reproducing/: present in the final site.
- /api/models/flex-dm/reproducing/ -> /models/flex-dm/reproducing/: present in the final site.
- /api/models/housegan/reproducing/ -> /models/housegan/reproducing/: present in the final site.
- /api/models/lace/reproducing/ -> /models/lace/reproducing/: present in the final site.
- /api/models/layousyn/reproducing/ -> /models/layousyn/reproducing/: present in the final site.
- /api/models/layout-action/reproducing/ -> /models/layout-action/reproducing/: present in the final site.
- /api/models/layout-corrector/reproducing/ -> /models/layout-corrector/reproducing/: present in the final site.
- /api/models/layout-detr/reproducing/ -> /models/layout-detr/reproducing/: present in the final site.
- /api/models/layout-dm/reproducing/ -> /models/layout-dm/reproducing/: present in the final site.
- /api/models/layout-fid/reproducing/ -> /models/layout-fid/reproducing/: present in the final site.
- /api/models/layout-flow/reproducing/ -> /models/layout-flow/reproducing/: present in the final site.
- /api/models/layout-gpt/reproducing/ -> /models/layout-gpt/reproducing/: present in the final site.
- /api/models/layoutdiffusion/reproducing/ -> /models/layoutdiffusion/reproducing/: present in the final site.
- /api/models/layoutformerpp/reproducing/ -> /models/layoutformerpp/reproducing/: present in the final site.
- /api/models/layoutganpp/reproducing/ -> /models/layoutganpp/reproducing/: present in the final site.
- /api/models/layoutprompter/reproducing/ -> /models/layoutprompter/reproducing/: present in the final site.
- /api/models/layoutvae/reproducing/ -> /models/layoutvae/reproducing/: present in the final site.
- /api/models/ltnet/reproducing/ -> /models/ltnet/reproducing/: present in the final site.
- /api/models/parse-then-place/reproducing/ -> /models/parse-then-place/reproducing/: present in the final site.
- /api/models/posterllama/reproducing/ -> /models/posterllama/reproducing/: present in the final site.
- /api/models/posterllava/reproducing/ -> /models/posterllava/reproducing/: present in the final site.
- /api/models/postero/reproducing/ -> /models/postero/reproducing/: present in the final site.
- /api/models/ralf/reproducing/ -> /models/ralf/reproducing/: present in the final site.
- /api/models/smarttext/reproducing/ -> /models/smarttext/reproducing/: present in the final site.
- /api/models/cgb-dm/training/ -> /models/cgb-dm/training/: present in the final site; this is one of the five resolved training collisions.
- /api/models/dlt/training/ -> /models/dlt/training/: present in the final site; this is one of the five resolved training collisions.
- /api/models/layout-dm/training/ -> /models/layout-dm/training/: present in the final site; this is one of the five resolved training collisions.
- /api/models/layout-flow/training/ -> /models/layout-flow/training/: present in the final site; this is one of the five resolved training collisions.
- /api/models/layoutdiffusion/training/ -> /models/layoutdiffusion/training/: present in the final site; this is one of the five resolved training collisions.

## Follow-up cleanup

Delete scripts/gen_ref_pages.py in the cleanup PR after this PR merges and the maintainer has announced the guide URL moves.

Remove the obsolete semantic-blank-line baseline entry for scripts/gen_ref_pages.py when that cleanup deletes the file; the README checker already uses gen_api_pages.py.

Decide whether the hosting layer should redirect the 28 old reproduction URLs and the five old training URLs; Zensical itself provides no native redirect configuration in this setup.

Consider excluding navigation-only SUMMARY.md output and unlisted docs/plans/ pages from the published artifact if a later Zensical release adds a supported option.
