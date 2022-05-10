#! /usr/bin/env python3
from aicspylibczi import CziFile
import argparse
import math
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageTk
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont


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
        self.xscale = float(sx.text)
        self.yscale = float(sy.text)
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

    def loadPois(self, fh):
        csv = CsvLoader(fh)
        if not csv.headersAre('type,x,y,z,name'):
            raise Exception('Could not understand csv file')
        self.pois = []
        self.regs = []
        for (t,x,y,z,name) in csv.generateRows():
            if t == 'i':
                if len(name) != 0:
                    self.pois.append(float(x) / self.xscale, float(y) / self.yscale, name)
            elif t == 'r':
                self.regs.append(float(x) / self.xscale, float(y) / self.yscale)


def addPoi(image, x, y, label):
    pass

def addRegPoint(image, x, y, label):
    pass


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
    '-o',
    '--output',
    help='output JPG file',
    required=True,
    dest='output'
)
parser.add_argument(
    help='input CZI file',
    dest='input',
    metavar='INPUT_CZI',
)
options = parser.parse_args()

czi = CziImageFile(getattr(options, 'input'))
out = czi.createScaledImage(float(getattr(options, 'factor')))

poi_file = getattr(options, 'poi_file')
if poi_file:
    with CsvLoader(poi_file) as poi:
        nameRe = re.compile(r'([0-9]+)[^0-9]*$')
        regCount = 0
        for (t,x,y,z,name) in pos.generateRows():
            if t == 'i':
                nameResult = nameRe.search(name)
                if nameResult:
                    addPoi(out, x, y, nameResult.group(1))
            elif t == 'r':
                regCount += 1
                addRegPoint(out, x, y, regCount)

out.save(getattr(options, 'output'))
