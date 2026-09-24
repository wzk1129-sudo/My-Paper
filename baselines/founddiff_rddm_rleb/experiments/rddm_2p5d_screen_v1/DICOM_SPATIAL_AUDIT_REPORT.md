# Mayo DICOM spatial metadata audit

Result: **PASS**. Pixel data were not read or modified.

| Patient | Dose | Slices | Series | Filename order | Spacing (mm) | Max gap deviation (mm) | NPY paths | Status |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| L067 | full_1mm | 560 | 1 | decreasing | 0.8000 | 0.0000 | 560 | PASS |
| L067 | quarter_1mm | 560 | 1 | decreasing | 0.8000 | 0.0000 | 560 | PASS |
| L143 | full_1mm | 585 | 1 | decreasing | 0.8000 | 0.0000 | 585 | PASS |
| L143 | quarter_1mm | 585 | 1 | decreasing | 0.8000 | 0.0000 | 585 | PASS |
| L333 | full_1mm | 610 | 1 | decreasing | 0.8000 | 0.0000 | 610 | PASS |
| L333 | quarter_1mm | 610 | 1 | decreasing | 0.8000 | 0.0000 | 610 | PASS |

| Patient | Matched positions | Max full/quarter difference (mm) | Pairing |
| --- | ---: | ---: | --- |
| L067 | 560 | 0.0 | PASS |
| L143 | 585 | 0.0 | PASS |
| L333 | 610 | 0.0 | PASS |

NPY index correspondence is established by the deterministic preprocessing source and existence of the expected indexed files; NPY binary provenance cannot be checked without pixel comparison.

The 1.0 mm DICOM SliceThickness and 0.8 mm center spacing are compatible with overlapping reconstructions; the directory name is not a center-spacing assertion. SpacingBetweenSlices is absent in these headers. All 1755 indexed full/quarter pairs have zero measured positional difference.
