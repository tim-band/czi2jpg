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

    def headersAre(self, cshs):
        """
        Returns True iff cshs (a comma-separated list of headers) matches
        the actual headers.
        """
        if not self.headers:
            return False
        hs = cshs.split(',')
        return hs == self.headers

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

    def copyImagePortion(self, out: Image, x : int, y : int, source : (int, int, int, int), scale : float):
        data = self.czi.read_mosaic(region=source, scale_factor=1/scale, C=0)
        rescaled = np.minimum(
            np.multiply(data[0], 1.0/256.0),
            255.0
        )
        img = np.asarray(rescaled.astype(np.uint8))
        pil = Image.fromarray(img)
        out.paste(pil, (x, y))

    def createScaledImage(self, scale):
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
        return r

    def toImagePoint(self, scale, sx, sy):
        return (
            math.floor((sx - self.bbox.x) / scale + 0.5),
            math.floor((sy - self.bbox.y) / scale + 0.5)
        )

    def loadPois(self, fh):
        csv = CsvLoader(fh)
        if not csv.headersAre('type,x,y,z,name'):
            raise Exception('Could not understand csv file')
        pois = []
        regs = []
        for (t,x,y,z,name) in csv.generateRows():
            if t == 'i':
                if len(name) != 0:
                    pois.append((float(x) / self.xscale, float(y) / self.yscale, name))
            elif t == 'r':
                regs.append((float(x) / self.xscale, float(y) / self.yscale))
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
    '-d',
    '--downsample',
    help='downsampling factor',
    dest='factor',
    required=False,
    default=1,
    type=float,
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

czi = CziImageFile(getattr(options, 'input'))
scale = float(getattr(options, 'factor'))
out = czi.createScaledImage(scale)

poi_file = getattr(options, 'poi_file')
if poi_file:
    nameRe = re.compile(r'([0-9]+)[^0-9]*$')
    regCount = 0
    (pois, regs) = czi.loadPois(poi_file)
    for (sx,sy,name) in pois:
        (x,y) = czi.toImagePoint(scale, float(sx), float(sy))
        nameResult = nameRe.search(name)
        if nameResult:
            addPoi(out, x, y, nameResult.group(1))
    for (sx,sy) in regs:
        (x,y) = czi.toImagePoint(scale, float(sx), float(sy))
        regCount += 1
        addRegPoint(out, x, y, str(regCount))

out.save(getattr(options, 'output'))
