# Layout-Corrector

Transformers/diffusers-style Layout-Corrector wrapper for the ECCV 2024 model. This package is a composite pipeline layered on the converted `layout-dm` package and reuses LayoutDM tokenization, conditioning, and scheduler behavior.

The original implementation is vendored at `vendor/layout-corrector` and remains read-only. Released weights are expected from the original Google Drive starter kit; conversion scripts download under `.cache/`, never under `vendor/`.
