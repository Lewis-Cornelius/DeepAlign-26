| Evidence | Pairs | MAE | Median AE | AR@50 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Teacher route | 24 | 18.9 ms | 0.0 ms | 98.4% | Target is reachable with dense supervised timing. |
| Best learned student | 8 | 112.0 ms | 31.2 ms | 69.2% | Student learns local timing but not robust route control. |
| Best AR/p90 probe | 8 | 113.2 ms | 30.8 ms | 69.9% | Similar learned-decoder ceiling with slightly higher AR. |
| Coarse-to-fine fused | 8 | 116.9 ms | 28.4 ms | 67.4% | Guided decoder improves some pairs but hurts others. |
| Oracle decoder selector | 8 | 91.8 ms | n/a | 72.1% | Decoder complementarity helps but remains far from target. |
