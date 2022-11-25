#! /usr/bin/env python3
from aicspylibczi import CziFile
import argparse
import math
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import re
import sys

def strip_zen_nonsense(h):
    """
    Zen adds a lot of weird stuff to a CSV when it saves it out.
    Here we remove it.
    """
    h = h.strip('"')
    start = h.find("::")
    start = 0 if start < 0 else start + 2
    end = h.find("!!")
    if end < 0:
        end = len(h)
    return h[start:end]

class CsvLoader:
    def __init__(self, fh):
        rows = fh.read().splitlines()
        if len(rows) == 0:
            self.headers = None
            self.rows = []
            return
        self.headers = [strip_zen_nonsense(h) for h in rows[0].split(',')]
        self.rows = [r for r in rows[1:] if r.strip(',"')]

    def headersStart(self, cshs):
        """
        Returns True iff cshs (a comma-separated list of headers) matches
        the leftmost actual headers.
        """
        if not self.headers:
            return False
        hs = cshs.split(',')
        if len(self.headers) < len(hs):
            return False
        return hs == self.headers[0:len(hs)]

    def rowCount(self):
        return len(self.rows)

    def generateRows(self):
        for r in self.rows:
            yield [c.strip('"') for c in r.split(',')]


class CziImageFile(object):
    def __init__(self, path):
        czi = CziFile(path)
        self.czi = czi
        self.bbox = czi.get_mosaic_bounding_box()
        sx = czi.meta.find('Metadata/Scaling/Items/Distance[@Id="X"]/Value')
        sy = czi.meta.find('Metadata/Scaling/Items/Distance[@Id="Y"]/Value')
        if sx == None or sy == None:
            raise Exception("No pixel size in metadata")
        self.xscale = float(sx.text) * 1e6
        self.yscale = float(sy.text) * 1e6
        focus_actions = czi.meta.findall(
            'Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/FocusAction'
        )
        if focus_actions:
            def focus_action_success(e):
                r = e.find('Result')
                return r != None and r.text == 'Success'
            self.surface = [
                [ float(a.find(tag).text) for tag in ['X', 'Y', 'ResultPosition'] ]
                for a in focus_actions
                if focus_action_success(a)
            ]
        else:
            overall_z_element = (
                czi.meta.find('Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/RegionsSetup/SampleHolder/TileRegions/TileRegion/Z')
                or czi.meta.find('Metadata/HardwareSetting/ParameterCollection[@Id="MTBFocus"]/Position')
            )
            self.surface = [float(overall_z_element.text)]
        self.imageBits = int(czi.meta.find('Metadata/Information/Image/ComponentBitCount').text)

    def copyImagePortion(self, out: Image, x : int, y : int, source : (int, int, int, int), scale : float):
        pil_type = np.uint8
        data = self.czi.read_mosaic(region=source, scale_factor=1/scale, C=0)
        pil_bits = 8 * pil_type(0).nbytes
        rescaled = np.floor(np.minimum(
            np.multiply(data[0], 2**(pil_bits - self.imageBits)),
            2**pil_bits - 1
        ))
        img = np.asarray(rescaled.astype(pil_type))
        pil = Image.fromarray(img)
        out.paste(pil, (x, y))

    def createScaledImage(self, scale, flip=False):
        width = math.floor(self.bbox.w / scale + 0.5)
        height = math.floor(self.bbox.h / scale + 0.5)
        r = Image.new(mode='RGB', size=(width, height))
        max_chunk = 200
        xc = math.floor(width / max_chunk)
        yc = math.floor(height / max_chunk)
        xs = math.ceil(width / xc)
        ys = math.ceil(height / yc)
        for xi in range(xc):
            for yi in range(yc):
                x = xi*xs
                y = yi*ys
                x_source = x * scale
                y_source = y * scale
                w_source = min(xs * scale, self.bbox.w - x_source)
                h_source = min(ys * scale, self.bbox.h - y_source)
                sbox = (
                    int(self.bbox.x + x_source),
                    int(self.bbox.y + y_source),
                    int(w_source),
                    int(h_source)
                )
                self.copyImagePortion(r, x, y, sbox, scale)
        if flip:
            return r.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        return r

    def toImagePoint(self, scale, sx, sy, flip=False):
        x = sx - self.bbox.x
        if flip:
            x = self.bbox.w - x
        return (
            math.floor(x / scale + 0.5),
            math.floor((sy - self.bbox.y) / scale + 0.5)
        )

    def loadPois(self, fh, m):
        csv = CsvLoader(fh)
        if not csv.headersStart('type,x,y,z,name'):
            raise Exception('Could not understand csv file')
        pois = []
        regs = []
        for (t,x,y,z,name,*extras) in csv.generateRows():
            x = float(x)
            y = float(y)
            if m:
                (x,y) = transform(m, x, y)
            if t == 'i':
                if len(name) != 0:
                    pois.append((x / self.xscale, y / self.yscale, name))
            elif t == 'r':
                regs.append((x / self.xscale, y / self.yscale))
        return (pois, regs)

def drawPoint(image, x, y, label, fn):
    size = 11
    font_size = 20
    width = 3
    colour = (200,10,10)
    draw = ImageDraw.Draw(image)
    fn(draw, size, colour, width)
    if label:
        font = ImageFont.truetype(font='arial.ttf', size=font_size)
        draw.text((x + size, y + size), label, fill=colour, anchor='lm', font=font)

def addPoi(image, x, y, label=None):
    def drawPoi(draw, size, colour, width):
        size *= 1.4
        draw.ellipse([x - size, y - size, x + size, y + size], outline=colour, fill=None, width=width)
    drawPoint(image, x, y, label, drawPoi)

def addRegPoint(image, x, y, label):
    def drawPoi(draw, size, colour, width):
        mid_size = size * 0.25
        for (sx,sy) in [(1,1), (1,-1), (-1,-1), (-1,1)]:
            draw.line(
                [x + sx * size, y + sy * size, x + sx * mid_size, y + sy * mid_size],
                fill=colour, width=width
            )
    drawPoint(image, x, y, label, drawPoi)

def transform(m, x, y):
    if m:
        return (
            x * m[0][0] + y * m[0][1] + m[0][2],
            x * m[1][0] + y * m[1][1] + m[1][2]
        )
    return (x, y)

parser = argparse.ArgumentParser(
    description= 'czi2jpg: '
    + 'a command-line tool for converting a CZI (Carl Zeiss Image) '
    + 'to JPG, optionally adding points of interest annotations.'
)
parser.add_argument(
    '-a',
    '--annotations',
    help='input CSV file for points of interest',
    required=False,
    dest='poi_file',
    metavar='POI-CSV',
    type=argparse.FileType('r')
)
parser.add_argument(
    '-r',
    '--reg',
    help='input CSV file registration matrix to multiply annotations by',
    required=False,
    dest='reg_file',
    metavar='REG-CSV',
    type=argparse.FileType('r')
)
parser.add_argument(
    '-d',
    '--downsample',
    help='downsampling factor',
    dest='factor',
    required=False,
    default=1,
    type=float,
)
parser.add_argument(
    '-f',
    '--flip',
    help='flip the image (left/right)',
    action='store_true'
)
parser.add_argument(
    help='input CZI file',
    dest='input',
    metavar='INPUT_CZI',
)
parser.add_argument(
    help='output JPG file',
    dest='output',
    metavar='OUTPUT_JPG',
)
options = parser.parse_args()

czi = CziImageFile(options.input)
scale = float(options.factor)
out = czi.createScaledImage(scale, options.flip)

poi_file = options.poi_file
if poi_file:
    matrix_file = options.reg_file
    reg = None
    if matrix_file:
        mfh = CsvLoader(matrix_file)
        reg = []
        for (x,y,t) in mfh.generateRows():
            reg.append([float(x),float(y),float(t)])
    (pois, regs) = czi.loadPois(poi_file, reg)
    nameRe = re.compile(r'([0-9]+)[^0-9]*$')
    for (sx,sy,name) in pois:
        (x,y) = czi.toImagePoint(scale, sx, sy, options.flip)
        nameResult = nameRe.search(name)
        if nameResult:
            addPoi(out, x, y, nameResult.group(1))
    regCount = 0
    for (sx,sy) in regs:
        (x,y) = czi.toImagePoint(scale, sx, sy, options.flip)
        regCount += 1
        addRegPoint(out, x, y, str(regCount))

out.save(options.output)
