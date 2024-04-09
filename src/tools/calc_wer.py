import sys
import os
import editdistance
import json

from whisper.normalizers import EnglishTextNormalizer


fout = sys.argv[1]
fref = sys.argv[2]

hyp_file = sys.argv[1] + '_hyp'
ref_file = sys.argv[1] + '_ref'

hyp_writer = open(hyp_file, 'w')
ref_writer = open(ref_file, 'w')


std = EnglishTextNormalizer()

ref_data = []

for line in open(fref):
    uid, sent = line.strip().split(None, 1)
    sent = std(sent)
    ref_data.append([uid, sent])
    if not sent:
        sent = ' '
    ref_writer.write(f"{sent}({uid})\n")


with open(sys.argv[1]) as fin:
    hyp_data = json.load(fin)
    assert len(hyp_data) == len(ref_data)
    for ii, sent in enumerate(hyp_data):
        uid = ref_data[ii][0]
        sent = std(sent)
        if not sent:
            sent = ' '
        hyp_writer.write(f"{sent}({uid})\n")
