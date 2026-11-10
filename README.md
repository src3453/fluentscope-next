# FluentScope Next

FluentScope Next is a real-time audio oscilloscope and spectrum analyzer for audio capture devices exposed by PortAudio.

- **Scope** shows triggered waveforms with stereo display options, level meters, and automatic gain control.
- **Spectrum** shows the frequency spectrum and a scrolling waterfall.
- **Lissajous** plots audio in X/Y mode for inspecting stereo phase relationships.
- **CRT rendering** adds a beam with adjustable focus, persistence, and bloom to the Scope and Lissajous views.

## Install and run

The commands below use Python 3.12 and Windows PowerShell.

```powershell
git clone https://github.com/src3453/fluentscope-next.git
cd fluentscope-next
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fluentscope_next.py
```

The last command lists available input devices and exits. Run again with the index of a microphone, line input, or loopback device listed on your system. Replace `0` with your device index.

```powershell
.\.venv\Scripts\python.exe fluentscope_next.py 0
```

You can also pass a unique substring of the device name or use `"exact:Device Name"` to match the full name. If PyAudio needs to build from source, install the PortAudio development headers for your platform first.

To enable CRT rendering, install ModernGL and use a graphics driver with OpenGL 3.3 core support. SciPy is optional and enables polyphase resampling for the CRT trace.

```powershell
.\.venv\Scripts\python.exe -m pip install moderngl scipy
```

## Controls

| Key | Action |
| --- | --- |
| F1 | Open the controls menu |
| Tab or 1–3 | Cycle or select Scope, Spectrum, and Lissajous |
| S | Toggle the spectrum overlay |
| R / M | Cycle the trigger or stereo mode |
| A / H | Cycle scope gain control or toggle gain peak hold |
| J / K / D | Toggle the scope grid, fixed-gain dBFS labels, or AGC dBFS labels |
| L / W / I | Cycle the X/Y mode, swap axes, or change axis inversion |
| C | Toggle the CRT beam when OpenGL rendering is available |
| T / Shift+T | Cycle themes forward or backward |

Settings are saved to `fluentscope_settings.json` beside the script.

## License

Copyright © 2026 src3453. FluentScope Next is distributed under the [MIT License](LICENSE). The upstream notices below apply to the respective third-party material.

### Dependencies

| Component | License and upstream notices |
| --- | --- |
| Python | [Python Software Foundation License](https://docs.python.org/3/license.html) |
| NumPy | [BSD 3-Clause and bundled component notices](https://numpy.org/doc/stable/license.html) |
| pygame | [LGPL 2.1](https://github.com/pygame/pygame/blob/2.6.1/docs/LGPL.txt) and [bundled library notices](https://github.com/pygame/pygame/tree/2.6.1/docs/licenses) |
| SDL 2 | [zlib](https://github.com/libsdl-org/SDL/blob/SDL2/LICENSE.txt) |
| PyAudio | [MIT](https://pypi.org/project/PyAudio/) |
| PortAudio | [MIT](https://github.com/PortAudio/portaudio/blob/master/LICENSE.txt) |
| ModernGL (optional) | [MIT](https://github.com/moderngl/moderngl/blob/main/LICENSE) |
| glcontext (via ModernGL) | [MIT](https://github.com/moderngl/glcontext/blob/main/LICENSE) |
| SciPy (optional) | [BSD 3-Clause and bundled component notices](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |

### Upstream code and design credits

- The correlation trigger is based on [Corrscope](https://github.com/corrscope/corrscope) (BSD 2-Clause).
- The CRT beam model draws on [Hansi Raber's Oscilloscope](https://github.com/kritzikratzi/Oscilloscope) (MIT) and [woscope's beam-rendering write-up](https://m1el.github.io/woscope-how/) ([woscope code under MIT](https://github.com/m1el/woscope/blob/master/LICENSE)).

<details>
<summary>Corrscope — BSD 2-Clause</summary>

```text
BSD 2-Clause License

Copyright (c) 2018-2020+, nyanpasu64
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

</details>

<details>
<summary>Oscilloscope — MIT</summary>

```text
MIT License

Copyright (c) 2017 hansi raber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

<details>
<summary>woscope — MIT</summary>

```text
The MIT License

Copyright (c) 2015 Igor Null <m1el.2027@gmail.com>, Chad von Nau <chad@vonnau.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

</details>
