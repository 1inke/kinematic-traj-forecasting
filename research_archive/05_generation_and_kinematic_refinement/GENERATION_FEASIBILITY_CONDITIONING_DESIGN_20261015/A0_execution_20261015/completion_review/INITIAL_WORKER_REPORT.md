# A0 analytic envelope results

Decision: STOP_FIXED_BOTTOM4_OPPORTUNITY

All20 and bottom4 are relaxed oracle bounds, not safe/model optima. No training or new baseline.

|split/group|n|native good|bottom4 UB good|all20 UB good|
|---|---:|---:|---:|---:|
|FIT/overall|18909|10858|11000|14996|
|FIT/stable|10767|7414|7433|9473|
|FIT/transition|3417|1507|1543|2415|
|FIT/change|4725|1937|2024|3108|
|outer/overall|1035|625|631|834|
|outer/stable|604|434|435|538|
|outer/transition|178|81|83|130|
|outer/change|253|110|113|166|
|inner/overall|1856|932|936|1403|
|inner/stable|978|601|602|833|
|inner/transition|381|165|165|282|
|inner/change|497|166|169|288|
|original_CAL/overall|2879|1490|1511|2149|
|original_CAL/stable|1535|994|996|1308|
|original_CAL/transition|644|253|264|423|
|original_CAL/change|700|243|251|418|

FIT-change gains: bottom4 1.841269841pp, all20 24.78306878pp. Integer comparisons control the gate, not rounded display.

A1 only if fixed-bottom4 FIT bound reaches5pp, after its own actual preflight. B/V never automatically bypass A0/A1. Samecandidate ADE<=.2 and FDE<=.4; untouched16 exact. Existing failed branches and old sources/models/results preserved.
