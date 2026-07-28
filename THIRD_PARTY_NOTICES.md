# Third-party notices

## COCO API

The files `src/evaluation_script/coco.py` and
`src/evaluation_script/cocoeval.py` contain code from the COCO API.

Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

Source: <https://github.com/cocodataset/cocoapi/blob/master/license.txt>

## KAIST dataset materials

The repository does not redistribute the full KAIST Multispectral Pedestrian
Dataset. The following retained assets are treated as KAIST dataset materials:

- `src/evaluation_script/KAIST_annotation.json`, retained for compatibility
  with the public KAIST evaluator;
- `assets/qualitative_montage.png`, an adapted qualitative comparison containing
  selected KAIST frames.

The official KAIST repository identifies the dataset license as
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
These two assets are therefore distributed under those terms rather than the
root software license.

Dataset source: <https://github.com/SoonminHwang/rgbt-ped-detection>

Reference: Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and In So Kweon.
“Multispectral Pedestrian Detection: Benchmark Dataset and Baselines.” CVPR,
2015.

## Other upstream projects

YOLOv5 and DeformCAT provenance and licenses are documented in
[`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md). Their AGPL-compatible source is
distributed under the repository's root AGPL-3.0 license.
